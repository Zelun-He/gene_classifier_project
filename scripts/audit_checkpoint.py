"""Audit available Git evidence without loading weights or printing credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def audit(root, revision):
    revision = git(root, "rev-parse", revision).decode().strip()
    objects = set()
    for commit in git(root, "rev-list", revision).decode().splitlines():
        objects.update(git(root, "ls-tree", "-r", "--format=%(objectname) %(path)", commit).decode().splitlines())
    datasets, artifacts = [], []
    for line in sorted(objects):
        oid, _, name = line.partition(" ")
        if not name:
            continue
        if Path(name).name in {"train.tsv", "test.tsv", "test.tst"}:
            raw = git(root, "cat-file", "blob", oid)
            datasets.append({"path": name, "git_blob": oid, "bytes": len(raw),
                             "rows": len(raw.splitlines()), "sha256": hashlib.sha256(raw).hexdigest()})
        if (Path(name).name in {"trainer_state.json", "training_args.bin", "model.safetensors", "pytorch_model.bin"}
                or "/checkpoint-916/" in name):
            artifacts.append(name)
    local = Path(root) / "models/output/checkpoint-916"
    return {"audited_revision": revision, "checkpoint": "models/output/checkpoint-916",
            "available_locally": local.exists(), "historical_training_artifacts": sorted(set(artifacts)),
            "historical_dataset_blobs": sorted(datasets, key=lambda x: x["path"]),
            "status": "exposure_history_unverifiable", "benchmark_allowed": False,
            "required_action": "restart_from_pristine_revision_pinned_pretrained_checkpoint",
            "findings": [
                "run_finetune.py loads checkpoint-916 before trainer.train(); its previous training inputs are unknown.",
                "The same loader uses test.tsv as Trainer eval_dataset, without a separate inner validation partition.",
                "Legacy dataset generation shuffles without a seed and writes sequence/label pairs without gene IDs.",
                "Recovered historical train.tsv and test.tst blobs are empty placeholders, not exposure records.",
                "Model weights, trainer state, split IDs and original training logs are absent from the audited history.",
                "Ignored/unpublished local artifacts and external training runs cannot be audited from Git.",
                "Missing evidence does not establish contamination or establish absence of contamination."],
            "initialization_requirements": ["Pristine upstream pretrained model with documented provenance",
                                            "Immutable model revision plus configuration/tokenizer/weight SHA-256 hashes",
                                            "Fresh task head and fresh optimizer/scheduler per inner and outer fit",
                                            "No task-finetuned legacy weights or continuation between folds",
                                            "Record pretraining corpus overlap separately from supervised task leakage"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--revision", required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(audit(a.repo, a.revision), indent=2, sort_keys=True) + "\n")
