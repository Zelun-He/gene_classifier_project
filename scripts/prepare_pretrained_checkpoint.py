"""Recover a pinned original DNABERT-6 MLM checkpoint with artifact integrity."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

MODEL = "zhihan1996/DNA_bert_6"
REVISION = "a79a8fd96ad172f964a4dbef3f4d7545a5034baa"
FILES = {
    "config.json": (1110, "5b1beca6b3c35e59eaa41147838dead30ae273b9"),
    "special_tokens_map.json": (112, "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5"),
    "tokenizer_config.json": (40, "9de0d36b30e859708df4d608b6f6b6d8f48e3243"),
    "vocab.txt": (28703, "cc9ad53732e21ea9180f18e1fb67a38cc5702d4c"),
    "pytorch_model.bin": (359199902, "e1688c3a3f881daff02a42b487e04612c4ec9ef33f1f02fa5f4e67d7b9865ca9"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_artifact(path, name):
    size, expected = FILES[name]
    if path.stat().st_size != size:
        raise ValueError(f"Unexpected upstream artifact size: {name}")
    if len(expected) == 64:
        actual = sha(path)
    else:
        raw = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if actual != expected:
        raise ValueError(f"Upstream artifact checksum mismatch: {name}")


def prepare(output, record_path, offline=False):
    output, record_path = Path(output), Path(record_path)
    output.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = output / name
        if not path.exists():
            if offline:
                raise ValueError(f"Missing offline checkpoint artifact: {name}")
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
            temporary = path.with_suffix(path.suffix + ".part")
            with urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
            verify_artifact(temporary, name)
            temporary.replace(path)
        verify_artifact(path, name)
        print(f"Verified {name}", flush=True)
    vocab = (output / "vocab.txt").read_text().splitlines()
    config = json.loads((output / "config.json").read_text())
    if (len(vocab) != 4101 or len(set(vocab)) != 4101 or config["model_type"] != "bert"
            or config["architectures"] != ["BertForMaskedLM"] or config.get("finetuning_task") is not None):
        raise ValueError("Expected original pretrained 6-mer BERT MLM configuration")
    record = {
        "kind": "pristine_pretrained", "task_finetuned": False, "model_id": MODEL, "revision": REVISION,
        "provenance_url": "https://github.com/jerryji1993/DNABERT#update-20250708",
        "checkpoint_url": f"https://huggingface.co/{MODEL}/tree/{REVISION}",
        "artifact_sha256": {"config": sha(output / "config.json"), "tokenizer": sha(output / "vocab.txt"),
                            "weights": sha(output / "pytorch_model.bin")},
        "files": {name: {"size_bytes": size, "sha256": sha(output / name),
                         "upstream_blob_or_lfs_oid": expected} for name, (size, expected) in FILES.items()},
        "basis": ["The authors' repository links this distribution as pretrained DNABERT-6",
                  "The pinned 2021 revision predates later custom-code/configuration changes",
                  "The original and current upstream weight files have the same LFS SHA-256",
                  "The extraction loader will require complete encoder tensors and permit only the MLM head to be discarded"],
        "pretraining_overlap": "Human reference sequences, including benchmark genes, may have appeared in unsupervised pretraining; this is not a novel-sequence holdout claim",
        "limitations": "Pristine status follows upstream provenance and artifact checks, not an independently recovered upstream training ledger",
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if record_path.exists() and record_path.read_text() != text:
        raise ValueError("Existing initialization record differs")
    record_path.write_text(text)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--record", required=True, type=Path)
    p.add_argument("--offline", action="store_true")
    a = p.parse_args()
    prepare(a.output, a.record, a.offline)
