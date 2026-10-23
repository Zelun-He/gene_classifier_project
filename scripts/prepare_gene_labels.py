"""Build a conservative, provenance-preserving cancer-gene label audit.

Only the Python standard library is needed. See docs/label_audit.md for policy.
"""

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile


POLICY_VERSION = "exclusive-cancer-roles-v1"
CATEGORIES = (
    "oncogene_only", "tumor_suppressor_only", "dual_role", "unresolved", "unannotated"
)
EMPTY = {"", "na", "n/a", "nan", "none", "null", "-"}
ID_HEADERS = {
    "entrez_id": {"entrez geneid", "entrez gene id", "entrez id"},
    "ensembl_id": {"ensembl gene id", "ensembl geneid", "ensembl id"},
}
MANIFEST_FIELDS = (
    "gene_id", "gene", "symbols", "entrez_ids", "ensembl_ids", "identity_status",
    "category", "label", "included", "exclusion_reasons", "observed_roles",
    "sources", "source_roles", "record_ids", "observation_count",
)


@dataclass(frozen=True)
class Observation:
    source: str
    record_id: str
    record_number: int
    gene: str
    entrez_id: str
    ensembl_id: str
    roles: tuple
    issues: tuple
    raw: dict


def json_text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def parse_bool(value):
    """Missing is unknown, never an implicit false annotation."""
    normalized = value.strip().lower()
    if normalized in EMPTY:
        return None
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"unrecognized boolean {value!r}")


def cosmic_roles(value):
    if value.strip().lower() in EMPTY:
        return (), ("missing_role_annotation",)
    aliases = {
        "oncogene": "OG", "og": "OG", "onc": "OG",
        "tsg": "TSG", "tumor suppressor": "TSG", "tumour suppressor": "TSG",
        "tumor suppressor gene": "TSG", "tumour suppressor gene": "TSG",
        "fusion": None,
    }
    roles, issues = set(), set()
    tokens = [" ".join(token.split()) for token in re.split(r"[,;|]", value.lower())]
    if not any(tokens):
        return (), ("missing_role_annotation",)
    for token in tokens:
        if not token:
            continue
        if token not in aliases:
            issues.add("unrecognized_role_annotation")
        elif aliases[token]:
            roles.add(aliases[token])
    return tuple(sorted(roles)), tuple(sorted(issues))


def read_id(row, kind):
    values = set()
    for header, raw in row.items():
        if " ".join(header.lower().split()) not in ID_HEADERS[kind]:
            continue
        value = raw.strip()
        if value.lower() in EMPTY or value == "0":
            continue
        if kind == "entrez_id":
            # Accept integer-valued CSV exports, but never round fractional IDs.
            if not re.fullmatch(r"[0-9]+(?:\.0+)?", value):
                raise ValueError(f"invalid Entrez ID {value!r}")
            value = value.split(".")[0].lstrip("0")
            if not value:
                continue
        else:
            if not re.fullmatch(r"ENSG[0-9]{11}(?:\.[0-9]+)?", value):
                raise ValueError(f"invalid human Ensembl gene ID {value!r}")
            value = value.split(".")[0]
        values.add(value)
    if len(values) > 1:
        raise ValueError(f"conflicting {kind} columns in one row")
    return next(iter(values), "")


def read_source(path, source):
    """Hash and parse the same bytes; retain every original data record."""
    path = Path(path)
    payload = path.read_bytes()
    reader = csv.DictReader(
        io.StringIO(payload.decode("utf-8-sig"), newline=""),
        delimiter="," if source == "cosmic" else "\t",
        strict=True,
    )
    headers = reader.fieldnames
    if not headers:
        raise ValueError(f"{source}: missing header")
    stripped = [header.strip() for header in headers]
    if len(set(stripped)) != len(stripped):
        raise ValueError(f"{source}: duplicate column names")
    required = (
        {"Gene Symbol", "Role in Cancer"} if source == "cosmic"
        else {"Hugo Symbol", "Is Oncogene", "Is Tumor Suppressor Gene"}
    )
    if not required.issubset(stripped):
        raise ValueError(f"{source}: missing columns {sorted(required - set(stripped))}")
    observations = []
    for number, raw in enumerate(reader, start=1):
        try:
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError("record has the wrong number of columns")
            row = {key.strip(): value for key, value in raw.items()}
            symbol = row["Gene Symbol" if source == "cosmic" else "Hugo Symbol"]
            gene = symbol.strip().upper()
            if not re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]*", gene):
                raise ValueError(f"missing or unsupported gene symbol {symbol!r}")
            if source == "cosmic":
                roles, issues = cosmic_roles(row["Role in Cancer"])
            else:
                oncogene = parse_bool(row["Is Oncogene"])
                suppressor = parse_bool(row["Is Tumor Suppressor Gene"])
                roles = tuple(role for role, flag in (("OG", oncogene), ("TSG", suppressor)) if flag)
                issues = ("incomplete_role_annotation",) if None in (oncogene, suppressor) else ()
            observations.append(Observation(
                source, f"{source}:{number}", number, gene,
                read_id(row, "entrez_id"), read_id(row, "ensembl_id"),
                roles, issues, raw,
            ))
        except ValueError as error:
            raise ValueError(f"{source} record {number}: {error}") from error
    if not observations:
        raise ValueError(f"{source}: no data records")
    return observations, {
        "filename": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "input_records": len(observations),
        "unique_normalized_symbols": len({row.gene for row in observations}),
        "records_without_stable_id": sum(not row.entrez_id and not row.ensembl_id for row in observations),
        "records_with_annotation_issues": sum(bool(row.issues) for row in observations),
        "explicit_dual_role_records": sum(set(row.roles) == {"OG", "TSG"} for row in observations),
    }


def group_observations(observations):
    """Connect shared symbols/IDs; quarantine ambiguous components later."""
    parent = list(range(len(observations)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, observation in enumerate(observations):
        for namespace, value in (
            ("symbol", observation.gene), ("entrez", observation.entrez_id),
            ("ensembl", observation.ensembl_id),
        ):
            if not value:
                continue
            key = (namespace, value)
            if key in seen:
                parent[root(index)] = root(seen[key])
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index, observation in enumerate(observations):
        groups[root(index)].append(observation)
    return list(groups.values())


def classify_group(observations):
    symbols = sorted({item.gene for item in observations})
    entrez = sorted({item.entrez_id for item in observations if item.entrez_id})
    ensembl = sorted({item.ensembl_id for item in observations if item.ensembl_id})
    identity_conflict = len(entrez) > 1 or len(ensembl) > 1
    has_stable_id = bool(entrez or ensembl)
    # A representative symbol, not a claim that an alias is the approved symbol.
    preferred = sorted({item.gene for item in observations if item.source == "oncokb"})
    gene = (preferred or symbols)[0]
    gene_id = f"ncbigene:{entrez[0]}" if entrez else (
        f"ensembl:{ensembl[0]}" if ensembl else f"symbol:{gene}"
    )
    if identity_conflict:
        identity = json_text({"symbols": symbols, "entrez": entrez, "ensembl": ensembl})
        gene_id = "ambiguous:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    role_sets = {frozenset(item.roles) for item in observations}
    issues = {issue for item in observations for issue in item.issues}
    reasons = []
    if identity_conflict:
        reasons.append("conflicting_stable_ids")
    if frozenset({"OG"}) in role_sets and frozenset({"TSG"}) in role_sets:
        reasons.append("opposing_exclusive_annotations")
    reasons.extend(sorted(issues))
    if reasons:
        category = "unresolved"
    elif frozenset({"OG", "TSG"}) in role_sets:
        category = "dual_role"
        reasons.append("explicit_dual_role_annotation")
    elif frozenset({"OG"}) in role_sets:
        category = "oncogene_only"
    elif frozenset({"TSG"}) in role_sets:
        category = "tumor_suppressor_only"
    else:
        category = "unannotated"
        reasons.append("no_oncogene_or_tsg_annotation")
    if not has_stable_id:
        reasons.append("missing_stable_id")
    label = {"oncogene_only": "OG", "tumor_suppressor_only": "TSG"}.get(category, "")
    included = bool(label) and not reasons
    source_roles = {}
    for source in sorted({item.source for item in observations}):
        source_roles[source] = sorted({
            "+".join(item.roles) or "none"
            for item in observations if item.source == source
        })
    return {
        "gene_id": gene_id, "gene": gene, "symbols": json_text(symbols),
        "entrez_ids": json_text(entrez), "ensembl_ids": json_text(ensembl),
        "identity_status": "conflict" if identity_conflict else ("stable_id" if has_stable_id else "symbol_only"),
        "category": category, "label": label, "included": included,
        "exclusion_reasons": json_text(reasons),
        "observed_roles": json_text(sorted({role for item in observations for role in item.roles})),
        "sources": json_text(sorted(source_roles)), "source_roles": json_text(source_roles),
        "record_ids": json_text(sorted(item.record_id for item in observations)),
        "observation_count": len(observations),
    }


def build_audit(observations):
    return sorted(
        (classify_group(group) for group in group_observations(observations)),
        key=lambda row: (row["gene"], row["gene_id"]),
    )


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_audit(output_dir, observations, manifest, sources, dataset_kind):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"output directory already exists; choose a new run directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    included = [row for row in manifest if row["included"]]
    reason_counts = Counter(
        reason for row in manifest for reason in json.loads(row["exclusion_reasons"])
    )
    summary = {
        "schema_version": 1, "policy_version": POLICY_VERSION,
        "dataset_kind": dataset_kind,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": sources,
        "policy": {
            "eligible_roles": ["oncogene_only", "tumor_suppressor_only"],
            "stable_id_required": True,
            "single_source_support_allowed": True,
            "fusion_only_is_unannotated": True,
            "blank_boolean_is_unknown": True,
            "representative_symbol": "lexicographic_with_oncokb_preference",
            "aliases": "merge_only_when_symbols_or_source_provided_stable_ids_match",
            "conflicts": "exclude_entire_connected_identity_group; no_source_priority",
        },
        "counts": {
            "input_records": len(observations),
            "identity_groups": len(manifest),
            "included_genes": len(included),
            "excluded_groups": len(manifest) - len(included),
            "categories": {category: sum(row["category"] == category for row in manifest) for category in CATEGORIES},
            "included_labels": {label: sum(row["label"] == label for row in included) for label in ("OG", "TSG")},
            "exclusion_reasons": dict(sorted(reason_counts.items())),
            "included_source_presence": dict(sorted(Counter("+".join(json.loads(row["sources"])) for row in included).items())),
            "included_label_support": dict(sorted(Counter(
                "+".join(source for source, roles in json.loads(row["source_roles"]).items() if row["label"] in roles)
                for row in included
            ).items())),
        },
        "limitations": [
            "Database annotations are not independent experimental validation.",
            "Source-provided identifiers are preserved, not verified against a current nomenclature authority.",
            "Missing identifiers and ambiguous annotations require manual resolution before inclusion.",
            "Identity groups with conflicting IDs may contain multiple biological genes; group counts are not gene counts.",
        ],
    }
    with tempfile.TemporaryDirectory(prefix=".label-audit-", dir=output_dir.parent) as temp:
        staged = Path(temp) / "audit"
        staged.mkdir()
        write_csv(staged / "gene_manifest.csv", manifest, MANIFEST_FIELDS)
        write_csv(staged / "gene_labels.csv", ({"gene": row["gene"], "label": row["label"]} for row in included), ("gene", "label"))
        write_csv(staged / "excluded_genes.csv", (row for row in manifest if not row["included"]), MANIFEST_FIELDS)
        write_csv(staged / "dual_role_genes.csv", (row for row in manifest if row["category"] == "dual_role"), MANIFEST_FIELDS)
        write_csv(staged / "unresolved_genes.csv", (row for row in manifest if row["category"] == "unresolved"), MANIFEST_FIELDS)
        with (staged / "source_annotations.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for observation in sorted(observations, key=lambda item: (item.source, item.record_number)):
                handle.write(json_text(asdict(observation)) + "\n")
        # Hash every output table so later changes are detectable.
        summary["artifacts"] = {
            path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
            for path in sorted(staged.iterdir())
        }
        (staged / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staged.rename(output_dir)
    return summary


def iso_date(value):
    try:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a date in YYYY-MM-DD format") from error
    return value


def nonempty(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be blank")
    return value.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmic", required=True, type=Path, help="COSMIC Cancer Gene Census CSV")
    parser.add_argument("--oncokb", required=True, type=Path, help="OncoKB Cancer Gene List TSV")
    for source in ("cosmic", "oncokb"):
        parser.add_argument(f"--{source}-release", required=True, type=nonempty, help="Exact source release/snapshot identifier; never inferred from filename")
        parser.add_argument(f"--{source}-retrieved-on", required=True, type=iso_date)
    parser.add_argument("--output-dir", required=True, type=Path, help="A new directory for this audit run")
    parser.add_argument("--dataset-kind", choices=("research", "synthetic"), default="research")
    args = parser.parse_args(argv)
    observations, sources = [], {}
    try:
        for source in ("cosmic", "oncokb"):
            rows, metadata = read_source(getattr(args, source), source)
            metadata.update({
                "release": getattr(args, f"{source}_release"),
                "retrieved_on": getattr(args, f"{source}_retrieved_on"),
            })
            observations.extend(rows)
            sources[source] = metadata
        summary = write_audit(args.output_dir, observations, build_audit(observations), sources, args.dataset_kind)
    except (OSError, ValueError, csv.Error, UnicodeError) as error:
        parser.exit(2, f"Label audit failed: {error}\n")
    counts = summary["counts"]
    print(f"{args.dataset_kind} audit: {counts['included_genes']} included genes, "
          f"{counts['excluded_groups']} excluded groups; saved to {args.output_dir}")
    if not counts["included_genes"]:
        print("No genes are eligible. Inspect excluded_genes.csv before proceeding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
