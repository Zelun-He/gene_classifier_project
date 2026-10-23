"""Fetch matching Ensembl translations and run exhaustive protein homology search."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from ensembl_snapshot import SnapshotClient


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(sequence_dir, cache, output, mmseqs, offline=False, threads=4):
    sequence_dir, output = Path(sequence_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((sequence_dir / "summary.json").read_text())
    # Verify the sequence run before deriving a new artifact from it.
    for name, expected in summary["artifacts"].items():
        expected = expected["sha256"] if isinstance(expected, dict) else expected
        if sha(sequence_dir / name) != expected:
            raise ValueError(f"Sequence artifact hash mismatch: {name}")
    rows = list(csv.DictReader((sequence_dir / "gene_coverage.csv").open()))
    identifiers = sorted({x for row in rows for x in json.loads(row["candidate_ensembl_ids"])})
    client = SnapshotClient(cache, offline=offline)
    client.reference()
    lookup = client.lookup_genes(identifiers)
    retained = sorted((row for row in rows if row["status"] == "retained"), key=lambda r: r["gene_id"])
    translations = []
    for row in retained:
        matches = [t for t in lookup[row["ensembl_gene_id"]]["Transcript"]
                   if t["id"] == row["transcript_id"] and str(t["version"]) == row["transcript_version"]]
        if len(matches) != 1:
            raise ValueError("Selected transcript missing or version changed")
        translations.append(matches[0]["Translation"])
    ids = sorted({t["id"] for t in translations})
    batches = [ids[i:i + 40] for i in range(0, len(ids), 40)]
    proteins = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for batch, records in zip(batches, pool.map(
                lambda b: client.rest("/sequence/id?type=protein", {"ids": b}), batches)):
            for record in records:
                key = record.get("query", record["id"])
                if key not in batch or key in proteins:
                    raise ValueError("Unexpected/duplicate protein response")
                proteins[key] = record
    manifest, fasta = [], []
    for i, (row, translation) in enumerate(zip(retained, translations)):
        p = proteins[translation["id"]]
        sequence = p["seq"].upper()
        if (p["id"] != translation["id"] or str(p["version"]) != str(translation["version"])
                or p["molecule"] != "protein" or len(sequence) != int(row["protein_length_aa"])
                or set(sequence) - set("ACDEFGHIKLMNPQRSTVWYOU")):
            raise ValueError(f"Protein QC failed: {row['gene_id']}")
        key = f"g{i:06d}"
        manifest.append({"search_id": key, "gene_id": row["gene_id"], "gene": row["gene"],
                         "transcript_id": row["transcript_id"], "transcript_version": row["transcript_version"],
                         "protein_id": p["id"], "protein_version": p["version"], "length": len(sequence),
                         "sha256": hashlib.sha256(sequence.encode()).hexdigest()})
        fasta.append(f">{key}\n{sequence}\n")
    (output / "proteins.fasta").write_text("".join(fasta))
    (output / "protein_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    version = subprocess.check_output([str(mmseqs), "version"], text=True).strip()
    if version != "8cc5ce367b5638c4306c2d7cfc652dd099a4643f":
        raise ValueError("Use pinned MMseqs2 18-8cc5c")
    command = [str(mmseqs), "easy-search", str(output / "proteins.fasta"),
               str(output / "proteins.fasta"), str(output / "alignments.tsv"), str(output / "tmp"),
               "--prefilter-mode", "2", "--alignment-mode", "3", "-a", "1", "--min-seq-id", "0",
               "-c", "0", "-e", "1e-5", "--max-seqs", str(len(retained)),
               "--max-accept", "2147483647", "--max-rejected", "2147483647",
               "--threads", str(threads), "--format-output",
               "query,target,nident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits", "--remove-tmp-files", "1"]
    with (output / "mmseqs.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    provenance = {"ensembl_release": 114, "assembly": "GRCh38.p14", "mmseqs_version": version,
                  "mmseqs_binary_sha256": sha(mmseqs), "command": command,
                  "sequence_summary_sha256": sha(sequence_dir / "summary.json"),
                  "artifacts": {name: sha(output / name) for name in
                                ["proteins.fasta", "protein_manifest.json", "alignments.tsv", "mmseqs.log"]},
                  "responses": client.used}
    (output / "homology_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"proteins": len(retained), "alignments_sha256": sha(output / "alignments.tsv")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence-dir", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mmseqs", type=Path, required=True)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    prepare(args.sequence_dir, args.cache, args.output, args.mmseqs, args.offline, args.threads)
