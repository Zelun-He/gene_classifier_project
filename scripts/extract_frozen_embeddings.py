"""Extract label-independent DNABERT-6 window and gene embeddings on CPU.

Strict original checkpoint loading, no remote code, no learned head, no optimizer.
The cache can resume only when plan, input, source and runtime hashes agree.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

if __package__:
    from . import benchmark_features as f, evaluation_protocol as protocol, prepare_pretrained_checkpoint as checkpoint
else:
    import benchmark_features as f
    import evaluation_protocol as protocol
    import prepare_pretrained_checkpoint as checkpoint


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_windows(sequence_dir, split_path, plan):
    sequence_dir, split_path = Path(sequence_dir), Path(split_path)
    if f.sha(sequence_dir / "summary.json") != plan["sequence_summary_sha256"] or f.sha(split_path) != plan["split_manifest_sha256"]:
        raise ValueError("Inputs differ from frozen embedding plan")
    summary = json.loads((sequence_dir / "summary.json").read_text())
    for name, metadata in summary["artifacts"].items():
        if f.sha(sequence_dir / name) != metadata["sha256"]:
            raise ValueError(f"Sequence input artifact changed: {name}")
    splits = json.loads(split_path.read_text())
    windows = [json.loads(line) for line in (sequence_dir / "windows.jsonl").read_text().splitlines()]
    windows = sorted((w for w in windows if w["sequence_type"] == "cds"), key=lambda w: (w["gene_id"], w["start_0based"]))
    if len({w["window_id"] for w in windows}) != len(windows):
        raise ValueError("Duplicate window IDs")
    regions = f.read_fasta(sequence_dir / "sequences.fasta")
    by_gene = defaultdict(list)
    for window in windows:
        if not window["sequence"] or set(window["sequence"]) - set("ACGT"):
            raise ValueError("Invalid window DNA")
        if not 6 <= len(window["sequence"]) == window["length_bp"] <= 510:
            raise ValueError("Invalid model-window length")
        by_gene[window["gene_id"]].append(window)
    if set(by_gene) != set(splits["groups"]):
        raise ValueError("Windows and fixed gene cohort differ")
    for gene, members in by_gene.items():
        if len({w["region_id"] for w in members}) != 1:
            raise ValueError("Expected one CDS region per gene")
        region = regions[members[0]["region_id"]]
        if "".join(w["sequence"] for w in members) != region:
            raise ValueError("Windows do not reconstruct the full CDS")
        offset = 0
        for w in members:
            if w["start_0based"] != offset or w["end_0based_exclusive"] != offset + len(w["sequence"]):
                raise ValueError("Window intervals are not contiguous")
            if not np.isclose(w["pool_weight"], len(w["sequence"]) / len(region), rtol=0, atol=1e-14):
                raise ValueError("Incorrect gene pooling weight")
            offset += len(w["sequence"])
    return windows


def load_encoder(model_dir, record_path, plan):
    import torch
    import transformers
    from transformers import BertConfig, BertModel, BertTokenizer
    model_dir, record_path = Path(model_dir), Path(record_path)
    if f.sha(record_path) != plan["initialization_record_sha256"]:
        raise ValueError("Initialization provenance record changed")
    record = json.loads(record_path.read_text())
    protocol.validate_initialization(record)
    if record["model_id"] != checkpoint.MODEL or record["revision"] != checkpoint.REVISION:
        raise ValueError("Unexpected upstream initialization")
    if plan["encoder"]["model_id"] != record["model_id"] or plan["encoder"]["revision"] != record["revision"]:
        raise ValueError("Plan and initialization disagree")
    for name in checkpoint.FILES:
        checkpoint.verify_artifact(model_dir / name, name)
        if f.sha(model_dir / name) != record["files"][name]["sha256"]:
            raise ValueError("Checkpoint record/file mismatch")
    if torch.__version__ != plan["software"]["torch"] or transformers.__version__ != plan["software"]["transformers"]:
        raise ValueError("Install pinned requirements-embeddings.txt in a separate environment")
    torch.set_num_threads(plan["encoder"]["threads"])
    torch.manual_seed(plan["seed"])
    torch.use_deterministic_algorithms(True)
    config = BertConfig.from_json_file(str(model_dir / "config.json"))
    if (config.hidden_size, config.num_hidden_layers, config.vocab_size, config.max_position_embeddings) != (768, 12, 4101, 512):
        raise ValueError("Unexpected original DNABERT-6 architecture")
    config._attn_implementation = plan["encoder"]["attention_implementation"]
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a plain upstream tensor state dictionary")
    if any(not (k.startswith("bert.") or k.startswith("cls.predictions.")) for k in state):
        raise ValueError("Unexpected non-encoder/non-MLM tensors")
    has_pooler = any(k.startswith("bert.pooler.") for k in state)
    model = BertModel(config, add_pooling_layer=has_pooler)
    encoder_state = {k[len("bert."):]: v for k, v in state.items() if k.startswith("bert.")}
    dropped_buffers = []
    for name in ("embeddings.position_ids", "embeddings.token_type_ids"):
        if name in encoder_state and name not in model.state_dict():
            expected = torch.arange(config.max_position_embeddings).unsqueeze(0) if name.endswith("position_ids") else torch.zeros((1, config.max_position_embeddings), dtype=torch.int64)
            if not torch.equal(encoder_state[name], expected):
                raise ValueError("Unexpected historical position/token-type buffer")
            del encoder_state[name]
            dropped_buffers.append(name)
    model.load_state_dict(encoder_state, strict=True)
    if any(t.dtype != torch.float32 for t in model.parameters()):
        raise ValueError("Expected original float32 encoder parameters")
    model.eval()
    model.requires_grad_(False)
    tokenizer = BertTokenizer(vocab_file=str(model_dir / "vocab.txt"), do_lower_case=False, model_max_length=512)
    kmers = ["".join(k) for k in itertools.product("ACGT", repeat=6)]
    mapped = tokenizer.convert_tokens_to_ids(kmers)
    if len(set(mapped)) != 4096 or tokenizer.unk_token_id in mapped or len(tokenizer) != 4101:
        raise ValueError("Tokenizer is not a complete one-token-per-6-mer vocabulary")
    return model, tokenizer, {"encoder_parameters": sum(t.numel() for t in model.parameters()),
                              "loaded_encoder_tensor_count": len(encoder_state), "pooler_loaded_but_not_used": has_pooler,
                              "discarded_mlm_head_tensors": sorted(k for k in state if k.startswith("cls.predictions.")),
                              "validated_historical_buffers": dropped_buffers, "strict_load": True,
                              "all_parameters_frozen": all(not t.requires_grad for t in model.parameters())}


def parameter_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return h.hexdigest()


def encode_windows(windows, tokenizer):
    ids, audit = [], []
    for w in windows:
        sequence = w["sequence"]
        kmers = [sequence[i:i + 6] for i in range(len(sequence) - 5)]
        encoded = tokenizer(" ".join(kmers), add_special_tokens=True, truncation=False,
                            return_attention_mask=True, return_special_tokens_mask=True)
        expected = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(kmers) + [tokenizer.sep_token_id]
        if encoded["input_ids"] != expected or len(expected) > 512 or tokenizer.unk_token_id in expected:
            raise ValueError("Tokenizer mismatch, unknown token or truncation needed")
        if sum(encoded["special_tokens_mask"]) != 2 or sum(encoded["attention_mask"]) != len(expected):
            raise ValueError("Unexpected model token masks")
        ids.append(encoded)
        audit.append({"window_id": w["window_id"], "gene_id": w["gene_id"], "bases": len(sequence),
                      "model_tokens": len(expected), "real_kmers": len(kmers), "unknown_tokens": 0,
                      "pool_weight": w["pool_weight"]})
    return ids, audit


def token_mean(hidden, attention, special):
    import torch
    if hidden.ndim != 3 or attention.shape != hidden.shape[:2] or special.shape != attention.shape:
        raise ValueError("Hidden states and token masks must align")
    if not torch.all((attention == 0) | (attention == 1)) or not torch.all((special == 0) | (special == 1)):
        raise ValueError("Invalid masks")
    valid = attention.bool() & ~special.bool()
    counts = valid.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("No real sequence tokens")
    pooled = torch.where(valid.unsqueeze(-1), hidden, 0).sum(dim=1) / counts.unsqueeze(-1)
    if not torch.isfinite(pooled).all():
        raise ValueError("Nonfinite window embedding")
    return pooled


def forward_batch(model, tokenizer, encoded):
    import torch
    batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
    special = batch.pop("special_tokens_mask")
    with torch.inference_mode():
        output = model(**batch, return_dict=True, output_attentions=False, output_hidden_states=False)
        vector = token_mean(output.last_hidden_state, batch["attention_mask"], special)
    return vector.cpu().numpy().astype(np.float32, copy=False)


def pool_genes(windows, vectors):
    if vectors.ndim != 2 or len(windows) != len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Window vectors must exactly align")
    if len({w["window_id"] for w in windows}) != len(windows):
        raise ValueError("Duplicate windows")
    genes = sorted({w["gene_id"] for w in windows})
    sums = {g: np.zeros(vectors.shape[1], dtype=np.float64) for g in genes}
    weights = defaultdict(float)
    for w, vector in zip(windows, vectors):
        if not np.isfinite(w["pool_weight"]) or w["pool_weight"] <= 0:
            raise ValueError("Invalid pool weight")
        sums[w["gene_id"]] += vector.astype(np.float64) * w["pool_weight"]
        weights[w["gene_id"]] += w["pool_weight"]
    if any(not np.isclose(v, 1, rtol=0, atol=1e-10) for v in weights.values()):
        raise ValueError("Incomplete gene window weights")
    return genes, np.vstack([sums[g] for g in genes]).astype(np.float32)


def extract(sequence_dir, split_path, model_dir, record_path, plan_path, output):
    import torch
    import transformers
    output, plan_path = Path(output), Path(plan_path)
    plan = json.loads(plan_path.read_text())
    e = plan["encoder"]
    if (e["frozen"] is not True or e["dropout"] is not False or e["truncation"] is not False
            or e["remote_code"] is not False or e["precision"] != "float32" or e["device"] != "cpu"
            or e["k"] != 6 or e["layer"] != "last" or plan["encoder_finetuning"] is not False):
        raise ValueError("Only predeclared frozen float32 CPU last-layer inference is supported")
    fixed_policy = {"window_policy": "existing_balanced_nonoverlapping_510bp_cds_windows",
                    "token_pooling": "last_hidden_state_mean_excluding_padding_and_special_tokens",
                    "gene_pooling": "raw_window_length_weighted_mean_float64_accumulation_then_float32",
                    "dimension": 768, "l2_normalization": False, "attention_implementation": "sdpa"}
    if any(e[key] != value for key, value in fixed_policy.items()) or e["threads"] < 1 or e["batch_size"] < 1:
        raise ValueError("Unsupported embedding extraction policy")
    if np.__version__ != plan["software"]["numpy"]:
        raise ValueError("NumPy version differs from frozen plan")
    windows = load_windows(sequence_dir, split_path, plan)
    model, tokenizer, loading = load_encoder(model_dir, record_path, plan)
    initial_parameter_hash = parameter_sha(model)
    encoded, audit = encode_windows(windows, tokenizer)
    receipt = {"plan_sha256": f.sha(plan_path), "initialization_record_sha256": f.sha(record_path),
               "windows_sha256": f.sha(Path(sequence_dir) / "windows.jsonl"),
               "source_sha256": {path.name: f.sha(path) for path in
                                  [Path(__file__), Path(checkpoint.__file__), Path(f.__file__), Path(protocol.__file__)]},
               "software": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
               "parameter_sha256_before": initial_parameter_hash, "loading": loading,
               "window_ids_sha256": f.object_sha([w["window_id"] for w in windows]), "encoder": e}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "extraction_identity.json").exists():
        if json.loads((output / "extraction_identity.json").read_text()) != receipt:
            raise ValueError("Existing extraction cache belongs to different inputs/code/runtime")
    else:
        write_json(output / "extraction_identity.json", receipt)
    blocks_dir = output / "blocks"
    blocks_dir.mkdir(exist_ok=True)
    blocks, checksums = [], {}
    start_time = time.monotonic()
    size = e["batch_size"]
    for start in range(0, len(windows), size):
        stop = min(start + size, len(windows))
        path = blocks_dir / f"{start:06d}.npz"
        sidecar = path.with_suffix(".sha256")
        if path.exists() and sidecar.exists():
            if sidecar.read_text().strip() != f.sha(path):
                raise ValueError("Cached embedding block checksum mismatch")
            with np.load(path, allow_pickle=False) as block:
                if block["ids"].tolist() != [w["window_id"] for w in windows[start:stop]]:
                    raise ValueError("Cached window IDs changed")
                vectors = block["embeddings"].copy()
                if vectors.shape != (stop - start, e["dimension"]) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid cached window embeddings")
        else:
            vectors = forward_batch(model, tokenizer, encoded[start:stop])
            temporary = path.with_suffix(".part")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, ids=np.array([w["window_id"] for w in windows[start:stop]]), embeddings=vectors)
            temporary.replace(path)
            sidecar.write_text(f.sha(path) + "\n")
        blocks.append(vectors)
        checksums[path.name] = f.sha(path)
        if (start // size) % 20 == 0 or stop == len(windows):
            print(json.dumps({"windows_done": stop, "windows_total": len(windows), "elapsed_seconds": round(time.monotonic() - start_time, 1)}), flush=True)
    vectors = np.vstack(blocks)
    genes, pooled = pool_genes(windows, vectors)
    final_hash = parameter_sha(model)
    if final_hash != initial_parameter_hash or model.training or any(t.requires_grad for t in model.parameters()):
        raise ValueError("Frozen encoder state changed during extraction")
    # Padding invariance is a numerical compatibility check, not model tuning.
    shortest, longest = int(np.argmin([len(x["input_ids"]) for x in encoded])), int(np.argmax([len(x["input_ids"]) for x in encoded]))
    alone = forward_batch(model, tokenizer, [encoded[shortest]])[0]
    padded = forward_batch(model, tokenizer, [encoded[shortest], encoded[longest]])[0]
    padding_error = float(np.max(np.abs(alone - padded)))
    if padding_error > 5e-5:
        raise ValueError("Unexpected padding-sensitive pooled representation")
    np.savez_compressed(output / "gene_embeddings.npz", gene_ids=np.array(genes), embeddings=pooled)
    write_json(output / "tokenization_audit.json", audit)
    write_json(output / "block_hashes.json", checksums)
    receipt.update({"status": "completed", "genes": len(genes), "windows": len(windows), "embedding_dimension": pooled.shape[1],
                    "parameter_sha256_after": final_hash, "parameters_unchanged": True,
                    "padding_invariance_max_absolute_error": padding_error, "unknown_tokens": 0,
                    "maximum_tokens_including_special": max(r["model_tokens"] for r in audit),
                    "gene_matrix_sha256": f.array_sha(pooled), "window_matrix_sha256": f.array_sha(vectors),
                    "artifacts": {name: f.sha(output / name) for name in
                                  ["gene_embeddings.npz", "tokenization_audit.json", "block_hashes.json", "extraction_identity.json"]}})
    write_json(output / "extraction.json", receipt)
    print(json.dumps({"completed": True, "genes": len(genes), "windows": len(windows), "dimension": pooled.shape[1]}), flush=True)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence-dir", required=True, type=Path)
    p.add_argument("--splits", default=Path("evaluation/v1/splits.json"), type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--initialization-record", default=Path("evaluation/dnabert6_initialization_v1.json"), type=Path)
    p.add_argument("--plan", default=Path("evaluation/frozen_encoder_plan_v1.json"), type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    extract(a.sequence_dir, a.splits, a.model_dir, a.initialization_record, a.plan, a.output)
