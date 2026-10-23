"""Tests for transcript policy, strand coordinates, coverage and gene pooling."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import ensembl_snapshot as transport
from scripts import prepare_sequence_inputs as pipeline


def transcript(identifier="ENST1", length=10, mane=False, canonical=False):
    return {"id": identifier, "version": 2, "biotype": "protein_coding", "strand": 1,
            "seq_region_name": "1", "Parent": "ENSG1", "start": 2000, "end": 7000,
            "assembly_name": "GRCh38", "is_canonical": int(canonical),
            "MANE": [{"type": "MANE_Select"}] if mane else [], "Translation": {"length": length}}


def gene(transcripts=None):
    return {"id": "ENSG1", "version": 3, "species": "homo_sapiens", "biotype": "protein_coding",
            "assembly_name": "GRCh38", "seq_region_name": "1", "strand": 1,
            "start": 2000, "end": 10000, "Transcript": transcripts or [transcript()]}


class PolicyTests(unittest.TestCase):
    def test_mane_precedes_canonical_and_longest(self):
        candidates = [transcript("LONG", 100), transcript("CANON", 50, canonical=True), transcript("MANE", 10, mane=True)]
        selected, rule = pipeline.choose_transcript(gene(candidates))
        self.assertEqual((selected["id"], rule), ("MANE", "MANE_Select"))

    def test_canonical_precedes_longest(self):
        selected, rule = pipeline.choose_transcript(gene([transcript("LONG",100), transcript("CANON", 10, canonical=True)]))
        self.assertEqual((selected["id"], rule), ("CANON", "Ensembl_Canonical"))

    def test_longest_fallback_and_stable_tie_break_ignore_input_order(self):
        candidates = [transcript("B", 20), transcript("A",20), transcript("C",10)]
        for ordering in (candidates, list(reversed(candidates))):
            selected, rule = pipeline.choose_transcript(gene(ordering))
            self.assertEqual((selected["id"], rule), ("A", "longest_translation"))

    def test_non_coding_and_wrong_parent_transcripts_are_not_candidates(self):
        noncoding = transcript("NONCODING", 100)
        noncoding["biotype"] = "lncRNA"
        wrong_parent = transcript("WRONG", 100)
        wrong_parent["Parent"] = "ENSG_OTHER"
        self.assertEqual(pipeline.choose_transcript(gene([noncoding,wrong_parent])), (None,"no_coding_transcript"))

    def test_unique_primary_coding_mapping_only(self):
        primary = gene()
        alternate = dict(primary, id="ENSG_ALT", seq_region_name="KI270728.1")
        lookup = {"ENSG1": primary, "ENSG_ALT": alternate}
        self.assertEqual(pipeline.choose_gene(lookup,lookup)[0]["id"],"ENSG1")
        lookup["ENSG2"] = dict(primary,id="ENSG2")
        self.assertEqual(pipeline.choose_gene(lookup,lookup)[1],"ambiguous_primary_coding_gene_mapping")
        self.assertEqual(pipeline.choose_gene([],lookup)[1],"no_ensembl_gene_mapping")

    def test_promoter_strand_coordinates_and_exact_width(self):
        plus = {"strand": 1, "start": 2000, "end": 7000}
        minus = dict(plus,strand=-1)
        self.assertEqual(pipeline.promoter_coordinates(plus,10000),(464,2511,1))
        self.assertEqual(pipeline.promoter_coordinates(minus,10000),(6489,8536,-1))
        for item in (plus,minus):
            start,end,_ = pipeline.promoter_coordinates(item,10000)
            self.assertEqual(end-start+1,2048)

    def test_promoter_boundaries_are_excluded_without_padding(self):
        self.assertIsNone(pipeline.promoter_coordinates({"strand":1,"start":100,"end":1000},10000))
        self.assertIsNone(pipeline.promoter_coordinates({"strand":-1,"start":100,"end":9800},10000))

    def test_sampling_is_reproducible_bounded_and_without_duplicate_starts(self):
        region = gene()
        first = pipeline.genomic_coordinates(region,"ncbigene:1",42)
        self.assertEqual(first,pipeline.genomic_coordinates(region,"ncbigene:1",42))
        self.assertNotEqual(first,pipeline.genomic_coordinates(region,"ncbigene:1",43))
        self.assertEqual(len(set(first)),3)
        for start,end,strand in first:
            self.assertEqual(end-start+1,2048)
            self.assertGreaterEqual(start,region["start"])
            self.assertLessEqual(end,region["end"])
            self.assertEqual(strand,1)
        self.assertEqual(pipeline.genomic_coordinates(dict(region,end=2100),"g",42),[])
        self.assertEqual(len(pipeline.genomic_coordinates(dict(region,end=4047),"g",42)),1)

    def test_coverage_uses_union_not_sum_of_overlapping_samples(self):
        self.assertEqual(pipeline.union_length([(10,20),(15,30),(31,40)]),31)

    def test_quality_reports_ambiguity_and_rejects_invalid_bases(self):
        stats, reason = pipeline.sequence_quality("ACGN",0)
        self.assertEqual((stats["length_bp"],stats["ambiguous_fraction"],reason),(4,0.25,"ambiguous_sequence"))
        self.assertEqual(pipeline.sequence_quality("ACGN",0.25)[1],"")
        self.assertEqual(pipeline.sequence_quality("ACG!",1)[1],"non_iupac_sequence")
        self.assertEqual(pipeline.sequence_quality("",0)[1],"empty_sequence")

    def test_cds_qc_translation_length_and_flags(self):
        self.assertEqual(pipeline.cds_quality("ATGAAATAA",2,0)[1],"")
        self.assertEqual(pipeline.cds_quality("ATGAAAT",2,0)[1],"cds_length_not_multiple_of_three")
        self.assertEqual(pipeline.cds_quality("ATGAAATAA",5,0)[1],"cds_translation_length_mismatch")
        # Non-ATG initiation and a missing terminal stop are flags, not automatic
        # exclusions: alternative initiation and annotation conventions exist.
        self.assertEqual(pipeline.cds_quality("GTGAAA",2,0)[0]["qc_flags"],["non_ATG_start","no_terminal_stop"])


class PoolingTests(unittest.TestCase):
    def test_token_mean_ignores_padding_and_special_tokens(self):
        mean = pipeline.mean_token_embedding([[100.],[2.],[4.],[1000.]], [1,1,1,0], [1,0,0,0])
        self.assertEqual(mean,[3.])
        with self.assertRaises(ValueError):
            pipeline.mean_token_embedding([[1.]], [1], [1])
    def regions(self):
        return [dict(gene_id="g1",sequence_type="cds",label="OG",region_id="g1|cds|r0",sequence="A"*513),
                dict(gene_id="g2",sequence_type="cds",label="TSG",region_id="g2|cds|r0",sequence="ACGTAC")]

    def test_every_base_appears_once_with_no_tiny_last_window(self):
        windows = pipeline.make_windows(self.regions(),510)
        first = [row for row in windows if row["gene_id"]=="g1"]
        self.assertEqual("".join(row["sequence"] for row in first),"A"*513)
        self.assertEqual([row["length_bp"] for row in first],[256,257])
        self.assertEqual(first[0]["start_0based"],0)
        self.assertEqual(first[-1]["end_0based_exclusive"],513)
        self.assertAlmostEqual(sum(row["pool_weight"] for row in first),1)
        self.assertTrue(all(row["length_bp"]<=510 for row in windows))

    def test_many_windows_produce_one_vector_per_gene(self):
        windows = pipeline.make_windows(self.regions(),510)
        embeddings = {row["window_id"]:[1.0,2.0] for row in windows}
        pooled = pipeline.pool_gene_embeddings(windows,embeddings)
        self.assertEqual(len(pooled),2)
        self.assertEqual(pooled[0]["embedding"],[1.0,2.0])

    def test_regions_receive_equal_weight_without_mixing_input_types(self):
        regions = [dict(gene_id="g",sequence_type="genomic",label="OG",region_id="one",sequence="A"*6),
                   dict(gene_id="g",sequence_type="genomic",label="OG",region_id="two",sequence="C"*12),
                   dict(gene_id="g",sequence_type="cds",label="OG",region_id="cds",sequence="G"*6)]
        windows = pipeline.make_windows(regions,6)
        embeddings = {row["window_id"]:[0.0 if row["region_id"]=="one" else 10.0] for row in windows}
        result = pipeline.pool_gene_embeddings(windows,embeddings)
        self.assertEqual({row["sequence_type"]:row["embedding"] for row in result},{"cds":[10.0],"genomic":[5.0]})

    def test_missing_duplicate_or_nonfinite_embeddings_fail_closed(self):
        windows = pipeline.make_windows(self.regions(),510)
        vectors = {row["window_id"]:[1.0] for row in windows}
        with self.assertRaises(ValueError):
            pipeline.pool_gene_embeddings(windows,{})
        with self.assertRaises(ValueError):
            pipeline.pool_gene_embeddings(windows+[windows[0]],vectors)
        bad = dict(vectors)
        bad[windows[0]["window_id"]]=[float("nan")]
        with self.assertRaises(ValueError):
            pipeline.pool_gene_embeddings(windows,bad)
        partial = windows[1:]
        with self.assertRaises(ValueError):
            pipeline.pool_gene_embeddings(partial,{row["window_id"]:[1.0] for row in partial})


class TransportTests(unittest.TestCase):
    def test_reference_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            client = transport.SnapshotClient(temp,offline=True)
            with patch.object(client,"rest",side_effect=[{"releases":[115]},{"assembly_name":"GRCh38.p14"}]),self.assertRaises(ValueError):
                client.reference()

    def test_offline_cache_miss_and_corruption_are_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            client = transport.SnapshotClient(temp,offline=True)
            with self.assertRaisesRegex(ValueError,"cache miss"):
                client.text("https://example.org")
            spec = {"url":"https://example.org","body":None}
            key = transport.digest(json.dumps(spec,sort_keys=True).encode())
            path = Path(temp)/f"{key}.json"
            path.write_text(json.dumps({"request":spec,"response":"changed","sha256":transport.digest(b"original")}))
            with self.assertRaisesRegex(ValueError,"Corrupt"):
                client.text("https://example.org")

    def test_offline_replay_checks_and_records_response_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = {"url":"https://example.org","body":None}
            key = transport.digest(json.dumps(spec,sort_keys=True).encode())
            (Path(temp)/f"{key}.json").write_text(json.dumps({"request":spec,"response":"original","sha256":transport.digest(b"original")}))
            client = transport.SnapshotClient(temp,offline=True)
            self.assertEqual(client.text("https://example.org"),"original")
            self.assertEqual(client.used[key]["response_sha256"],transport.digest(b"original"))


class EndToEndTests(unittest.TestCase):
    def audit(self, root):
        root.mkdir()
        row = {"gene_id":"ncbigene:1","gene":"SYNTHETIC","label":"OG","included":"True","entrez_ids":"[\"1\"]","ensembl_ids":"[]"}
        with (root/"gene_manifest.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        (root/"summary.json").write_text(json.dumps({"dataset_kind":"research","counts":{"included_genes":1},"artifacts":{"gene_manifest.csv":{"sha256":transport.digest((root/"gene_manifest.csv").read_bytes())}}}))

    class FakeClient:
        used = {}
        def reference(self,refresh=False):
            return {"releases":[114]}, {"assembly_name":"GRCh38.p14","top_level_region":[{"name":name,"length":100000} for name in pipeline.CHROMOSOMES]}
        def map_entrez(self,ids):
            return {"1":{"ENSG1"}}
        def lookup_genes(self,ids):
            return {"ENSG1":gene([transcript(length=2,mane=True)])}
        def sequences(self,queries,kind):
            return {query:({"seq":"ATGAAATAA","version":2,"molecule":"dna"} if kind=="cds" else {"seq":"A"*2048,"molecule":"dna"}) for query in queries}

    def test_complete_pipeline_preserves_gene_units_and_writes_integrity_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); audit=root/"labels"; self.audit(audit)
            with redirect_stdout(io.StringIO()):
                summary=pipeline.prepare(audit,root/"output",self.FakeClient())
            self.assertEqual(summary["input_genes"],1)
            self.assertEqual(summary["paired_all_inputs_genes"],1)
            self.assertEqual(summary["inputs"]["genomic"]["genes"],1)
            self.assertEqual(summary["inputs"]["genomic"]["regions"],3)
            windows=[json.loads(line) for line in (root/"output"/"windows.jsonl").read_text().splitlines()]
            pooled=pipeline.pool_gene_embeddings(windows,{row["window_id"]:[1.] for row in windows})
            self.assertEqual(len(pooled),3)  # One gene in three separate experiments.
            for filename,metadata in summary["artifacts"].items():
                self.assertEqual(transport.digest((root/"output"/filename).read_bytes()),metadata["sha256"])
            with self.assertRaisesRegex(ValueError,"already exists"):
                pipeline.prepare(audit,root/"output",self.FakeClient())

    def test_tampered_label_manifest_and_reference_abort_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); audit=root/"labels"; self.audit(audit)
            (audit/"gene_manifest.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError,"integrity"):
                pipeline.prepare(audit,root/"output",self.FakeClient())
            self.assertFalse((root/"output").exists())

    def test_one_failed_control_region_excludes_that_gene_input_only(self):
        class FailedRegionClient(self.FakeClient):
            def sequences(self,queries,kind):
                records = super().sequences(queries,kind)
                if kind != "cds":
                    bad = next(query for query in queries if int(query.split(':')[1].split('..')[0]) >= 2000)
                    records[bad]["seq"] = "N"*2048
                return records
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); audit=root/"labels"; self.audit(audit)
            with redirect_stdout(io.StringIO()):
                summary=pipeline.prepare(audit,root/"output",FailedRegionClient())
            self.assertEqual(summary["retained_cds_genes"],1)
            self.assertEqual(summary["inputs"]["promoter"]["genes"],1)
            self.assertEqual(summary["inputs"]["genomic"]["genes"],0)
            self.assertEqual(summary["paired_all_inputs_genes"],0)

    def test_cds_version_mismatch_aborts_without_partial_output(self):
        class WrongVersionClient(self.FakeClient):
            def sequences(self,queries,kind):
                records=super().sequences(queries,kind)
                if kind=="cds":
                    for record in records.values():
                        record["version"]=999
                return records
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); audit=root/"labels"; self.audit(audit)
            with redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,"version/type"):
                pipeline.prepare(audit,root/"output",WrongVersionClient())
            self.assertFalse((root/"output").exists())

    def test_non_coding_only_cohort_is_reported_with_empty_output_schemas(self):
        class NonCodingClient(self.FakeClient):
            def lookup_genes(self,ids):
                value=gene()
                value["biotype"]="lncRNA"
                return {"ENSG1":value}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); audit=root/"labels"; self.audit(audit)
            with redirect_stdout(io.StringIO()):
                summary=pipeline.prepare(audit,root/"output",NonCodingClient())
            self.assertEqual(summary["retained_cds_genes"],0)
            self.assertEqual(summary["excluded_reasons"],{"no_primary_protein_coding_gene":1})
            self.assertEqual((root/"output"/"paired_genes.csv").read_text(),"gene_id,gene,label\n")


if __name__ == "__main__":
    unittest.main()
