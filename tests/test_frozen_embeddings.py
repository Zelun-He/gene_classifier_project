"""Tokenizer/pooling compatibility without downloading a model in the test suite."""
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import extract_frozen_embeddings as encoder
from scripts import prepare_sequence_inputs as sequence
from scripts import prepare_pretrained_checkpoint as checkpoint

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


class PoolingTests(unittest.TestCase):
    def test_gene_pooling_matches_existing_sequence_policy(self):
        windows = [{"window_id": "a0", "gene_id": "a", "pool_weight": .4, "sequence_type": "cds", "label": "OG"},
                   {"window_id": "a1", "gene_id": "a", "pool_weight": .6, "sequence_type": "cds", "label": "OG"},
                   {"window_id": "b0", "gene_id": "b", "pool_weight": 1., "sequence_type": "cds", "label": "TSG"}]
        vectors = np.array([[1, 2], [3, 4], [10, 20]], dtype=np.float32)
        ids, actual = encoder.pool_genes(windows, vectors)
        expected = sequence.pool_gene_embeddings(windows, {w["window_id"]: v.tolist() for w, v in zip(windows, vectors)})
        self.assertEqual(ids, [r["gene_id"] for r in expected])
        np.testing.assert_allclose(actual, [r["embedding"] for r in expected], rtol=1e-7)
        self.assertEqual(actual.dtype, np.float32)

    def test_missing_duplicate_and_nonfinite_windows_fail(self):
        windows = [{"window_id": "a", "gene_id": "g", "pool_weight": .5}]
        with self.assertRaises(ValueError):
            encoder.pool_genes(windows, np.ones((1, 2)))
        with self.assertRaises(ValueError):
            encoder.pool_genes(windows * 2, np.ones((2, 2)))
        windows[0]["pool_weight"] = 1.
        with self.assertRaises(ValueError):
            encoder.pool_genes(windows, np.array([[np.nan]]))

    def test_checkpoint_pins_are_pretrained_mlm_artifacts(self):
        self.assertEqual(checkpoint.REVISION, "a79a8fd96ad172f964a4dbef3f4d7545a5034baa")
        self.assertEqual(checkpoint.FILES["pytorch_model.bin"][0], 359199902)
        self.assertNotIn("checkpoint-916", checkpoint.MODEL)
        record = json.loads((Path(__file__).resolve().parents[1] / "evaluation/dnabert6_initialization_v1.json").read_text())
        self.assertEqual(record["artifact_sha256"]["weights"], checkpoint.FILES["pytorch_model.bin"][1])


@unittest.skipUnless(HAS_TORCH, "Install optional requirements-embeddings.txt for encoder tests")
class TorchPoolingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_real_token_mean_excludes_specials_and_nan_padding(self):
        import torch
        hidden = torch.tensor([[[99., 99.], [1., 2.], [3., 4.], [77., 77.], [float("nan"), float("nan")]]])
        attention, special = torch.tensor([[1, 1, 1, 1, 0]]), torch.tensor([[1, 0, 0, 1, 1]])
        actual = encoder.token_mean(hidden, attention, special)
        np.testing.assert_array_equal(actual.numpy(), [[2., 3.]])
        with self.assertRaises(ValueError):
            encoder.token_mean(hidden, attention, torch.ones_like(special))

    def test_complete_sixmer_tokenizer_never_truncates_510_bases(self):
        from transformers import BertTokenizer
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.txt"
            tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + ["".join(k) for k in itertools.product("ACGT", repeat=6)]
            path.write_text("\n".join(tokens) + "\n")
            tokenizer = BertTokenizer(vocab_file=str(path), do_lower_case=False, model_max_length=512)
            windows = [{"window_id": "full", "gene_id": "a", "sequence": "ACGT" * 127 + "AC", "pool_weight": 1.},
                       {"window_id": "short", "gene_id": "b", "sequence": "ACGTACGTAC", "pool_weight": 1.}]
            encoded, audit = encoder.encode_windows(windows, tokenizer)
            self.assertEqual(len(encoded[0]["input_ids"]), 507)
            self.assertEqual(audit[0]["real_kmers"], 505)
            self.assertEqual(sum(r["unknown_tokens"] for r in audit), 0)
            batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
            self.assertEqual(int(batch["attention_mask"][1].sum()), 7)

    def test_tiny_frozen_bert_is_padding_invariant_and_does_not_change_parameters(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizer
        torch.manual_seed(42)
        model = BertModel(BertConfig(vocab_size=4101, hidden_size=16, num_hidden_layers=1,
                                    num_attention_heads=2, intermediate_size=32, max_position_embeddings=512))
        model.eval(); model.requires_grad_(False)
        before = encoder.parameter_sha(model)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.txt"
            path.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] +
                                      ["".join(k) for k in itertools.product("ACGT", repeat=6)]) + "\n")
            tokenizer = BertTokenizer(vocab_file=str(path), do_lower_case=False, model_max_length=512)
            windows = [{"window_id": "a", "gene_id": "a", "sequence": "ACGTACGT", "pool_weight": 1.},
                       {"window_id": "b", "gene_id": "b", "sequence": "ACGT" * 30, "pool_weight": 1.}]
            encoded, _ = encoder.encode_windows(windows, tokenizer)
            alone = encoder.forward_batch(model, tokenizer, encoded[:1])
            padded = encoder.forward_batch(model, tokenizer, encoded)[:1]
            np.testing.assert_allclose(alone, padded, rtol=1e-5, atol=1e-6)
            self.assertEqual(before, encoder.parameter_sha(model))
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
