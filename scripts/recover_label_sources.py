"""Recover the original source exports from this repository's initial commit.

This reads already committed Git objects, not the databases' authenticated APIs.
Raw exports remain untracked. No annotations or missing dates are invented.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


SOURCE_COMMIT = "0b3b43e04fa7b613e6eec9634669cb5f05c3563f"
REPOSITORY_URL = "https://github.com/Zelun-He/gene_classifier_project"
SOURCES = {
    "cosmic": {
        "path": "gene_classfifier_project/Census_allThu May 22 21_50_23 2025.csv",
        "sha256": "5f5fbe6e68533e9f771680eb1473f6242913993f2429891c9bf48af363b7d9de",
    },
    "oncokb": {
        "path": "gene_classfifier_project/cancerGeneList.tsv",
        "sha256": "802e957942bbfbe73127824e9b1841a4973bfa978741f81b99bd8ec4353f3994",
    },
}


def read_historical_blob(repo_root, path):
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{SOURCE_COMMIT}:{path}"],
        capture_output=True,
    )
    if result.returncode:
        raise ValueError(
            f"Cannot read {path!r} from {SOURCE_COMMIT}. "
            "Use a Git clone containing the initial commit; ZIP downloads and shallow "
            "clones may lack these objects. Fetch repository history, then retry."
        )
    return result.stdout


def recover_sources(output_dir, repo_root):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    recovered, metadata = {}, {}
    for source, spec in SOURCES.items():
        payload = read_historical_blob(repo_root, spec["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != spec["sha256"]:
            raise ValueError(f"{source}: recovered bytes do not match the recorded SHA-256")
        recovered[source] = payload
        metadata[source] = {
            "filename": Path(spec["path"]).name,
            "git_path": spec["path"],
            "provenance": f"{REPOSITORY_URL}/blob/{SOURCE_COMMIT}/{spec['path'].replace(' ', '%20')}",
            "sha256": digest,
            "size_bytes": len(payload),
            "release": "unreported-original-export",
            "retrieved_on": "unknown",
        }
    manifest = {
        "repository": REPOSITORY_URL,
        "source_commit": SOURCE_COMMIT,
        "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "git show; exact original bytes; SHA-256 verified",
        "sources": metadata,
        "limitations": [
            "Repository recovery does not independently authenticate the original database downloads.",
            "Database releases and original retrieval dates are not established by a Git commit or filename.",
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".source-recovery-", dir=output_dir.parent) as temp:
        staged = Path(temp) / "sources"
        staged.mkdir()
        for source, payload in recovered.items():
            (staged / metadata[source]["filename"]).write_bytes(payload)
        (staged / "recovery_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staged.rename(output_dir)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New untracked source directory")
    args = parser.parse_args(argv)
    try:
        manifest = recover_sources(args.output_dir, Path(__file__).resolve().parents[1])
    except (OSError, ValueError) as error:
        parser.exit(2, f"Source recovery failed: {error}\n")
    print(f"Recovered {len(manifest['sources'])} original exports from {SOURCE_COMMIT} to {args.output_dir}")
    print("Release and original retrieval dates remain unreported/unknown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
