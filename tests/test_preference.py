"""Preference learning: the DPO reference identity, and the reward model.

The load-bearing assertion is the identity: when the policy IS its frozen
reference, every log-ratio is zero, the margin is exactly zero, and the loss is
exactly ln 2. If that drifts, the reference is wired wrong and every DPO number
downstream is measured against the wrong baseline.
"""

from __future__ import annotations

import copy
import math
import unittest

import torch
import torch.nn.functional as F

from glassbox import data
from glassbox.preference import (
    DPOConfig,
    RewardConfig,
    RewardModel,
    batch_sequence_logprobs,
    dpo_terms,
    train_dpo,
    train_reward_model,
)

from . import fixtures

#: ln 2 as float32 arithmetic actually produces it. -logsigmoid(0) in float32 is
#: 0.6931471824645996; math.log(2) is 0.6931471805599453. The 1.9e-9 gap is the
#: dtype, not the implementation, so the assertion names the float32 value.
LN2_FLOAT32 = float(-F.logsigmoid(torch.tensor(0.0)))


class DPOReferenceIdentity(unittest.TestCase):
    BETA = 0.1

    def setUp(self) -> None:
        self.tok = fixtures.tokenizer()
        base = fixtures.trained_model()
        self.policy = copy.deepcopy(base)
        self.reference = copy.deepcopy(base)
        self.policy.eval()
        self.reference.eval()
        self.terms = dpo_terms(
            self.policy, self.reference, self.tok, data.PREFERENCE_PAIRS, self.BETA
        )

    def test_the_margin_is_exactly_zero(self) -> None:
        margin = self.terms["margin"]
        self.assertEqual(
            float(margin.abs().max()),
            0.0,
            "with policy == reference the largest margin is {}".format(
                float(margin.abs().max())
            ),
        )
        self.assertTrue(torch.equal(margin, torch.zeros_like(margin)))

    def test_the_loss_is_exactly_ln_2(self) -> None:
        loss = float(self.terms["loss"])
        self.assertEqual(
            loss,
            LN2_FLOAT32,
            "loss {!r} is not the float32 value of ln 2 ({!r})".format(
                loss, LN2_FLOAT32
            ),
        )
        self.assertAlmostEqual(loss, math.log(2), places=8)
        per_pair = self.terms["loss_per_pair"]
        self.assertTrue(
            torch.equal(per_pair, torch.full_like(per_pair, LN2_FLOAT32))
        )

    def test_policy_and_reference_logprobs_are_bit_identical(self) -> None:
        for key in ("chosen", "rejected"):
            with self.subTest(side=key):
                self.assertTrue(
                    torch.equal(
                        self.terms["policy_" + key], self.terms["reference_" + key]
                    )
                )

    def test_the_implied_rewards_are_zero(self) -> None:
        for key in ("implied_reward_chosen", "implied_reward_rejected"):
            with self.subTest(term=key):
                self.assertEqual(float(self.terms[key].abs().max()), 0.0)

    def test_all_six_pairs_are_scored(self) -> None:
        self.assertEqual(
            self.terms["margin"].shape, (len(data.PREFERENCE_PAIRS),)
        )
        self.assertEqual(len(data.PREFERENCE_PAIRS), 6)

    def test_a_perturbed_policy_moves_the_margin_off_zero(self) -> None:
        """The identity must be a property of equality, not of the arithmetic
        always returning zero."""
        with torch.no_grad():
            self.policy.token_embedding.weight.add_(0.05)
        terms = dpo_terms(
            self.policy, self.reference, self.tok, data.PREFERENCE_PAIRS, self.BETA
        )
        self.assertGreater(float(terms["margin"].abs().max()), 0.0)
        self.assertNotEqual(float(terms["loss"]), LN2_FLOAT32)


class BatchSequenceLogprobs(unittest.TestCase):
    def test_padding_cannot_change_another_rows_score(self) -> None:
        tok = fixtures.tokenizer()
        model = fixtures.trained_model()
        prompt = "who visits the garden?"
        alone = batch_sequence_logprobs(model, tok, [prompt], ["mira."])
        with_mate = batch_sequence_logprobs(
            model,
            tok,
            [prompt, prompt],
            [
                "mira.",
                "a person visits the garden each morning and that person is "
                "called mira.",
            ],
        )
        self.assertAlmostEqual(float(alone[0]), float(with_mate[0]), places=4)

    def test_length_mismatch_is_rejected(self) -> None:
        tok = fixtures.tokenizer()
        model = fixtures.trained_model()
        with self.assertRaises(ValueError):
            batch_sequence_logprobs(model, tok, ["a", "b"], ["one."])

    def test_scores_are_negative_and_longer_answers_score_lower(self) -> None:
        tok = fixtures.tokenizer()
        model = fixtures.trained_model()
        pair = data.PREFERENCE_PAIRS[3]  # the wordy-but-factual rejection
        scores = batch_sequence_logprobs(
            model, tok, [pair.prompt, pair.prompt], [pair.chosen, pair.rejected]
        )
        self.assertLess(float(scores[0]), 0.0)
        self.assertLess(
            float(scores[1]),
            float(scores[0]),
            "a much longer response did not accumulate a lower total logprob",
        )


class RewardModelTraining(unittest.TestCase):
    """A short Bradley-Terry run: the loss must start at ln 2 and fall."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = fixtures.tokenizer()
        cls.out = train_reward_model(
            fixtures.trained_model(),
            cls.tok,
            data.PREFERENCE_PAIRS,
            RewardConfig(steps=60),
        )

    def test_the_head_starts_at_zero_so_the_loss_starts_at_ln_2(self) -> None:
        first = self.out["history"][0]
        self.assertEqual(first["step"], 0)
        self.assertAlmostEqual(first["loss"], math.log(2), places=6)
        self.assertAlmostEqual(first["mean_margin"], 0.0, places=6)

    def test_the_loss_falls_and_the_ranking_is_learned(self) -> None:
        history = self.out["history"]
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        self.assertGreater(history[-1]["mean_margin"], 0.0)

    def test_every_pair_is_reported_with_its_reason(self) -> None:
        self.assertEqual(len(self.out["pairs"]), len(data.PREFERENCE_PAIRS))
        for row, pair in zip(self.out["pairs"], data.PREFERENCE_PAIRS):
            with self.subTest(prompt=pair.prompt):
                self.assertEqual(row["reason"], pair.reason)
                self.assertEqual(
                    row["ranked_correctly"], row["score_chosen"] > row["score_rejected"]
                )
                self.assertAlmostEqual(
                    row["margin"],
                    row["score_chosen"] - row["score_rejected"],
                    places=5,
                )

    def test_preference_is_not_correctness(self) -> None:
        """At least one rejected answer is factually true; the data must say so."""
        factual = [p for p in data.PREFERENCE_PAIRS if p.rejected_is_factual]
        self.assertTrue(
            factual,
            "no pair is marked rejected_is_factual, so the course cannot show "
            "that a ranking is about style as well as truth",
        )

    def test_an_untrained_reward_head_scores_everything_zero(self) -> None:
        rm = RewardModel(fixtures.trained_model())
        with torch.no_grad():
            scores = rm.score(
                self.tok,
                [p.prompt for p in data.PREFERENCE_PAIRS],
                [p.chosen for p in data.PREFERENCE_PAIRS],
            )
        self.assertEqual(float(scores.abs().max()), 0.0)


class DPOTraining(unittest.TestCase):
    """A short DPO run, checking the direction of what it optimises.

    The margin rising while the chosen answer's own log-probability FALLS is a
    real and widely missed property of DPO: the objective is a difference, so it
    can be satisfied by pushing the rejected answer down further than the
    chosen one.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = fixtures.tokenizer()
        policy = fixtures.trained_model()
        cls.out = train_dpo(
            policy, cls.tok, data.PREFERENCE_PAIRS, DPOConfig(steps=40)
        )

    def test_the_run_starts_at_ln_2_with_a_zero_margin(self) -> None:
        first = self.out["history"][0]
        self.assertEqual(first["step"], 0)
        self.assertAlmostEqual(first["loss"], math.log(2), places=6)
        self.assertAlmostEqual(first["margin"], 0.0, places=7)

    def test_the_loss_falls_and_the_margin_rises(self) -> None:
        history = self.out["history"]
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        self.assertGreater(history[-1]["margin"], history[0]["margin"])

    def test_the_reference_is_frozen(self) -> None:
        reference = self.out["reference"]
        for name, p in reference.named_parameters():
            with self.subTest(parameter=name):
                self.assertFalse(p.requires_grad)
        self.assertFalse(reference.training)

    def test_each_pair_reports_both_sides_against_the_reference(self) -> None:
        for row in self.out["pairs"]:
            with self.subTest(prompt=row["prompt"]):
                self.assertAlmostEqual(
                    row["implied_reward_chosen"],
                    0.1 * (row["policy_chosen"] - row["reference_chosen"]),
                    places=4,
                )
                self.assertAlmostEqual(
                    row["margin"],
                    row["implied_reward_chosen"] - row["implied_reward_rejected"],
                    places=4,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
