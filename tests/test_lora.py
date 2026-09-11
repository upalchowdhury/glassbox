"""LoRA: the properties that make "low rank adapter" mean something.

Conventions under test (getting these backwards is the usual way LoRA is taught
wrongly):

    W        [out, in]   frozen
    A        [r, in]     trainable, small random init
    B        [out, r]    trainable, ZERO init
    delta_W  = (alpha / r) * (B @ A)
"""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from glassbox.lora import (
    LoRAConfig,
    LoRALinear,
    adapter_report,
    apply_lora,
    first_step_gradients,
    lora_modules,
    merge_all,
    unmerge_all,
)

from . import fixtures

RANK = 2
ALPHA = 4.0
#: Relative to the logit scale, for the same reason as the KV-cache check:
#: merging sums the update into the base weight while the adapter path sums it
#: separately, and float32 gives the two orders slightly different round-off.
TOLERANCE = 1e-5


def _probe(tok):
    return torch.tensor([tok.encode("pip has two red stones.", bos=True)])


def _loss(model, ids):
    logits = model(ids[:, :-1])["logits"]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))


class AdapterAttachment(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = fixtures.tokenizer()
        self.ids = _probe(self.tok)
        self.model = fixtures.trained_model()
        self.model.eval()
        with torch.no_grad():
            self.before = self.model(self.ids)["logits"].clone()
        self.info = apply_lora(self.model, LoRAConfig(rank=RANK, alpha=ALPHA))

    def test_the_adapter_is_an_exact_no_op_at_initialisation(self) -> None:
        with torch.no_grad():
            after = self.model(self.ids)["logits"]
        self.assertTrue(
            torch.equal(self.before, after),
            "attaching an adapter changed the output by {} - B is not zero".format(
                float((self.before - after).abs().max())
            ),
        )

    def test_b_starts_at_exactly_zero_and_a_does_not(self) -> None:
        modules = lora_modules(self.model)
        self.assertTrue(modules, "no adapters were attached")
        for name, m in modules:
            with self.subTest(module=name):
                self.assertTrue(torch.all(m.lora_B == 0.0))
                self.assertFalse(torch.all(m.lora_A == 0.0))
                self.assertEqual(tuple(m.lora_A.shape), (RANK, m.base.in_features))
                self.assertEqual(tuple(m.lora_B.shape), (m.base.out_features, RANK))
                self.assertEqual(float(m.delta_weight.abs().max()), 0.0)
                self.assertEqual(m.scaling, ALPHA / RANK)

    def test_only_the_adapters_are_trainable(self) -> None:
        trainable = [
            n for n, p in self.model.named_parameters() if p.requires_grad
        ]
        self.assertTrue(trainable)
        for name in trainable:
            with self.subTest(parameter=name):
                self.assertTrue(
                    name.endswith("lora_A") or name.endswith("lora_B"),
                    "{} is trainable but is not an adapter".format(name),
                )
        cfg = fixtures.model_config()
        # Two matrices (A and B) per adapter, one adapter per target per block;
        # A is [r, width] and B is [width, r], so each contributes r * width.
        per_adapter = 2 * RANK * cfg.width
        expected = per_adapter * len(LoRAConfig().targets) * cfg.layers
        self.assertEqual(self.info["trainable_parameters"], expected)
        self.assertEqual(self.info["trainable_parameters"], 128)
        self.assertEqual(self.info["frozen_parameters"], 3048)
        self.assertAlmostEqual(self.info["trainable_fraction"], 128 / 3176, places=6)

    def test_the_attached_targets_are_q_and_v_in_every_block(self) -> None:
        self.assertEqual(
            self.info["attached_to"],
            ["blocks.0.q", "blocks.0.v", "blocks.1.q", "blocks.1.v"],
        )

    def test_applying_twice_does_not_stack_adapters(self) -> None:
        again = apply_lora(self.model, LoRAConfig(rank=RANK, alpha=ALPHA))
        self.assertEqual(again["attached_to"], [])
        self.assertEqual(len(lora_modules(self.model)), 4)

    def test_rank_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            LoRALinear(nn.Linear(4, 4, bias=False), rank=0, alpha=1.0)

    def test_an_unknown_target_is_rejected(self) -> None:
        model = fixtures.trained_model()
        with self.assertRaises(KeyError):
            apply_lora(model, LoRAConfig(rank=RANK, alpha=ALPHA, targets=("nope",)))


class FirstStepGradients(unittest.TestCase):
    def test_a_has_exactly_zero_gradient_while_b_does_not(self) -> None:
        tok = fixtures.tokenizer()
        ids = _probe(tok)
        model = fixtures.trained_model()
        apply_lora(model, LoRAConfig(rank=RANK, alpha=ALPHA))
        reports = first_step_gradients(model, _loss(model, ids))
        self.assertEqual(len(reports), 4)
        for report in reports:
            with self.subTest(module=report["module"]):
                self.assertTrue(report["B_is_zero"])
                self.assertEqual(
                    report["A_grad_norm"],
                    0.0,
                    "A has gradient {} on the first step; it is multiplied by "
                    "B = 0 so it must be exactly zero".format(report["A_grad_norm"]),
                )
                self.assertIsNotNone(report["B_grad_norm"])
                self.assertGreater(report["B_grad_norm"], 0.0)

    def test_a_acquires_a_gradient_once_b_is_non_zero(self) -> None:
        """The asymmetry is about the first step only, not about A being inert."""
        tok = fixtures.tokenizer()
        ids = _probe(tok)
        model = fixtures.trained_model()
        apply_lora(model, LoRAConfig(rank=RANK, alpha=ALPHA))
        with torch.no_grad():
            for _, m in lora_modules(model):
                m.lora_B.add_(0.1)
        reports = first_step_gradients(model, _loss(model, ids))
        for report in reports:
            with self.subTest(module=report["module"]):
                self.assertFalse(report["B_is_zero"])
                self.assertGreater(report["A_grad_norm"], 0.0)


class TrainedAdapters(unittest.TestCase):
    """One short adapter-only training run, shared by the tests below.

    Every test works on its own deep copy of the trained model, so a test that
    merges adapters or perturbs a weight cannot leak into another and the class
    does not depend on method execution order.
    """

    STEPS = 30

    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = fixtures.tokenizer()
        cls.ids = _probe(cls.tok)
        model = fixtures.trained_model()
        cls.snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
        apply_lora(model, LoRAConfig(rank=RANK, alpha=ALPHA))
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.05
        )
        model.train()
        for _ in range(cls.STEPS):
            loss = _loss(model, cls.ids)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        cls.template = model

    def setUp(self) -> None:
        self.model = copy.deepcopy(self.template)
        self.model.eval()

    def test_the_adapters_actually_moved(self) -> None:
        """Otherwise every test below would pass trivially."""
        for name, m in lora_modules(self.model):
            with self.subTest(module=name):
                self.assertFalse(torch.all(m.lora_B == 0.0))
                self.assertGreater(float(m.delta_weight.abs().max()), 0.0)

    def test_base_weights_are_bit_identical_after_adapter_training(self) -> None:
        current = dict(self.model.named_parameters())
        compared = 0
        for name, saved in self.snapshot.items():
            # Attaching an adapter renames the wrapped Linear's weight to
            # '<module>.base.weight'.
            for candidate in (name, name.replace(".weight", ".base.weight")):
                if candidate in current:
                    compared += 1
                    with self.subTest(parameter=name):
                        self.assertTrue(
                            torch.equal(saved, current[candidate].detach()),
                            "{} changed by {}".format(
                                name,
                                float(
                                    (saved - current[candidate].detach()).abs().max()
                                ),
                            ),
                        )
                    break
        self.assertEqual(compared, len(self.snapshot))

    def test_merge_equals_unmerge_within_float32_tolerance(self) -> None:
        with torch.no_grad():
            unmerged = self.model(self.ids)["logits"].clone()
        scale = max(float(unmerged.abs().max()), 1e-12)
        merge_all(self.model)
        with torch.no_grad():
            merged = self.model(self.ids)["logits"].clone()
        relative = float((unmerged - merged).abs().max()) / scale
        self.assertLessEqual(
            relative,
            TOLERANCE,
            "merged and unmerged logits differ relatively by {:.3e} on a scale "
            "of {:.3e}".format(relative, scale),
        )
        for _, m in lora_modules(self.model):
            self.assertTrue(m.merged)
        unmerge_all(self.model)
        with torch.no_grad():
            roundtrip = float(
                (unmerged - self.model(self.ids)["logits"]).abs().max()
            ) / scale
        self.assertLessEqual(roundtrip, TOLERANCE)
        for _, m in lora_modules(self.model):
            self.assertFalse(m.merged)

    def test_merging_twice_is_idempotent(self) -> None:
        merge_all(self.model)
        with torch.no_grad():
            once = self.model(self.ids)["logits"].clone()
        merge_all(self.model)
        with torch.no_grad():
            twice = self.model(self.ids)["logits"].clone()
        self.assertTrue(torch.equal(once, twice))

    def test_the_update_has_at_most_r_non_zero_singular_values(self) -> None:
        for report in adapter_report(self.model):
            with self.subTest(module=report["module"]):
                singular = report["delta_singular_values"]
                self.assertEqual(len(singular), min(report["base_shape"]))
                self.assertLessEqual(
                    report["delta_matrix_rank"],
                    RANK,
                    "update realised rank {} with singular values {}".format(
                        report["delta_matrix_rank"], singular
                    ),
                )
                # Beyond r the values are float32 round-off, not structure.
                ordered = sorted(singular, reverse=True)
                largest = ordered[0]
                self.assertGreater(largest, 0.0)
                for value in ordered[RANK:]:
                    self.assertLess(value, largest * 1e-4)

    def test_delta_weight_matches_the_stated_convention(self) -> None:
        for name, m in lora_modules(self.model):
            with self.subTest(module=name):
                expected = (ALPHA / RANK) * (m.lora_B @ m.lora_A)
                self.assertTrue(torch.equal(m.delta_weight, expected))
                self.assertEqual(
                    tuple(m.delta_weight.shape), tuple(m.base.weight.shape)
                )

    def test_the_merged_weight_is_base_plus_delta(self) -> None:
        for name, m in lora_modules(self.model):
            base_before = m.base.weight.detach().clone()
            delta = m.delta_weight.detach().clone()
            m.merge()
            with self.subTest(module=name):
                self.assertTrue(
                    torch.allclose(
                        m.base.weight.detach(), base_before + delta, atol=1e-6
                    )
                )
            m.unmerge()
            self.assertTrue(
                torch.allclose(m.base.weight.detach(), base_before, atol=1e-6)
            )

    def test_a_merged_adapter_has_no_adapter_branch_left_in_the_forward_pass(
        self,
    ) -> None:
        """After merging, the module is arithmetically an ordinary Linear."""
        merge_all(self.model)
        for name, m in lora_modules(self.model):
            x = torch.randn(3, m.base.in_features)
            with self.subTest(module=name), torch.no_grad():
                self.assertTrue(torch.equal(m(x), m.base(x)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
