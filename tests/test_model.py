"""Model invariants: initialisation scale, causality, the KV cache, the context
guard, and seed reproducibility.
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from glassbox import data
from glassbox.model import ModelConfig, build_model
from glassbox.train import document_windows

from . import fixtures


class UntrainedLoss(unittest.TestCase):
    """An untrained model must be exactly as surprised as a uniform guess.

    With tied input/output embeddings, ``nn.Embedding``'s default N(0, 1) makes
    the untrained cross-entropy roughly twice ln(vocab). Every later "the loss
    went down" claim is measured from this starting point, so if it is wrong the
    whole curve is meaningless.
    """

    TOLERANCE = 0.05

    def test_cross_entropy_is_within_0_05_of_ln_vocab(self) -> None:
        tok = fixtures.tokenizer()
        cfg = fixtures.model_config()
        model = build_model(cfg, seed=fixtures.SEED)
        model.eval()
        rows = [
            w.ids
            for w in document_windows(tok, data.split_documents("train"), cfg.context)
        ][:64]
        total, counted = 0.0, 0
        with torch.no_grad():
            for row in rows:
                ids = torch.tensor([row])
                logits = model(ids[:, :-1])["logits"]
                total += float(
                    F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        ids[:, 1:].reshape(-1),
                        reduction="sum",
                    )
                )
                counted += ids.size(1) - 1
        loss = total / counted
        uniform = math.log(len(tok))
        self.assertLessEqual(
            abs(loss - uniform),
            self.TOLERANCE,
            "untrained loss {:.4f} is not within {} of ln({}) = {:.4f}".format(
                loss, self.TOLERANCE, len(tok), uniform
            ),
        )

    def test_init_std_is_small_enough_to_matter(self) -> None:
        """The default N(0,1) failure mode, reproduced so the fix is visible."""
        cfg = fixtures.model_config()
        model = build_model(cfg, seed=fixtures.SEED)
        std = float(model.token_embedding.weight.detach().std())
        self.assertLess(std, 0.05, "embedding init is far wider than init_std")
        self.assertAlmostEqual(std, cfg.init_std, delta=0.01)

    def test_parameter_count_is_what_the_course_reports(self) -> None:
        model = build_model(fixtures.model_config(), seed=fixtures.SEED)
        self.assertEqual(model.num_parameters, 3048)


class Causality(unittest.TestCase):
    def test_changing_a_future_token_leaves_earlier_logits_bit_identical(self) -> None:
        tok = fixtures.tokenizer()
        model = fixtures.trained_model()
        ids = torch.tensor([tok.encode("pip has two red stones.", bos=True)])
        alt = ids.clone()
        original = int(alt[0, -1])
        alt[0, -1] = (original + 5) % len(tok)
        self.assertNotEqual(int(alt[0, -1]), original)
        with torch.no_grad():
            a = model(ids)["logits"]
            b = model(alt)["logits"]
        self.assertTrue(
            torch.equal(a[:, :-1], b[:, :-1]),
            "earlier positions changed by up to {} when a later token was "
            "edited".format(float((a[:, :-1] - b[:, :-1]).abs().max())),
        )
        # The edited position itself MUST change, or the test is vacuous.
        self.assertGreater(float((a[:, -1] - b[:, -1]).abs().max()), 0.0)

    def test_attention_assigns_exactly_zero_to_future_positions(self) -> None:
        tok = fixtures.tokenizer()
        model = fixtures.trained_model()
        ids = torch.tensor([tok.encode("pip has two", bos=True)])
        cap = model(ids, capture=True)["capture"]
        for layer in range(model.cfg.layers):
            weights = cap.records["block{}.attention".format(layer)]["values"][0]
            for head_index, head in enumerate(weights):
                for i, row in enumerate(head):
                    with self.subTest(layer=layer, head=head_index, query=i):
                        self.assertEqual(row[i + 1 :], [0.0] * (len(row) - i - 1))
                        self.assertAlmostEqual(sum(row), 1.0, delta=1e-6)


class KVCache(unittest.TestCase):
    """Caching changes how much work the forward pass repeats, not what it
    computes. The comparison is relative to the logit scale: an absolute
    threshold passes on an untrained model and then fails on a trained one
    purely because the logits grew."""

    TOLERANCE = 1e-5

    def test_incremental_decode_matches_the_full_forward_pass(self) -> None:
        tok = fixtures.tokenizer()
        model = fixtures.trained_model()
        ids = torch.tensor([tok.encode("pip has two red stones.", bos=True)])
        with torch.no_grad():
            full = model(ids)["logits"]
            past, steps = None, []
            for i in range(ids.size(1)):
                out = model(ids[:, i : i + 1], past=past, use_cache=True)
                past = out["past"]
                steps.append(out["logits"][:, -1])
            incremental = torch.stack(steps, dim=1)
        self.assertEqual(full.shape, incremental.shape)
        worst = float((full - incremental).abs().max())
        scale = max(float(full.abs().max()), 1e-12)
        self.assertLessEqual(
            worst / scale,
            self.TOLERANCE,
            "cached and uncached logits differ by {:.3e} on a scale of "
            "{:.3e} (relative {:.3e})".format(worst, scale, worst / scale),
        )

    def test_cache_length_grows_by_one_per_step(self) -> None:
        tok = fixtures.tokenizer()
        model = fixtures.trained_model()
        ids = torch.tensor([tok.encode("pip has ", bos=True)])
        with torch.no_grad():
            out = model(ids, use_cache=True)
            past = out["past"]
            self.assertEqual(len(past), model.cfg.layers)
            self.assertEqual(past[0][0].size(2), ids.size(1))
            for expected in range(ids.size(1) + 1, ids.size(1) + 4):
                out = model(torch.tensor([[5]]), past=past, use_cache=True)
                past = out["past"]
                self.assertEqual(past[0][0].size(2), expected)
                self.assertEqual(past[0][1].size(2), expected)


class ContextGuard(unittest.TestCase):
    def test_an_over_long_sequence_raises(self) -> None:
        cfg = fixtures.model_config()
        model = build_model(cfg, seed=fixtures.SEED)
        too_long = torch.zeros(1, cfg.context + 1, dtype=torch.long)
        with self.assertRaises(ValueError) as caught:
            model(too_long)
        self.assertIn(str(cfg.context), str(caught.exception))

    def test_a_sequence_of_exactly_context_length_is_allowed(self) -> None:
        cfg = fixtures.model_config()
        model = build_model(cfg, seed=fixtures.SEED)
        out = model(torch.zeros(1, cfg.context, dtype=torch.long))
        self.assertEqual(out["logits"].shape, (1, cfg.context, cfg.vocab_size))

    def test_the_cache_offset_counts_towards_the_guard(self) -> None:
        """A cached prefix plus a new token must not be able to exceed context."""
        cfg = fixtures.model_config()
        model = build_model(cfg, seed=fixtures.SEED)
        with torch.no_grad():
            out = model(
                torch.zeros(1, cfg.context, dtype=torch.long), use_cache=True
            )
            with self.assertRaises(ValueError):
                model(torch.zeros(1, 1, dtype=torch.long), past=out["past"],
                      use_cache=True)

    def test_context_is_long_enough_for_the_longest_document(self) -> None:
        tok = fixtures.tokenizer()
        longest = max(
            len(tok.encode(d.text, bos=True, eos=True)) for d in data.documents()
        )
        self.assertLessEqual(longest, fixtures.model_config().context)

    def test_width_must_divide_heads(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(vocab_size=45, width=8, heads=3)


class SeedReproducibility(unittest.TestCase):
    def test_the_same_seed_gives_bit_identical_weights(self) -> None:
        cfg = fixtures.model_config()
        a = build_model(cfg, seed=42)
        b = build_model(cfg, seed=42)
        self.assertEqual(fixtures.weight_hash(a), fixtures.weight_hash(b))
        for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(na, nb)
            self.assertTrue(torch.equal(pa, pb), "{} differs".format(na))

    def test_a_different_seed_gives_different_weights(self) -> None:
        cfg = fixtures.model_config()
        a = build_model(cfg, seed=42)
        c = build_model(cfg, seed=43)
        self.assertNotEqual(
            fixtures.weight_hash(a),
            fixtures.weight_hash(c),
            "seed 42 and seed 43 produced identical weights, so the seed is "
            "not actually wired to initialisation",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
