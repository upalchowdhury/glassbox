"""Every check in ``glassbox.checks.run_all`` must pass.

The blueprint claimed the training script "asserts causal-mask zeros, attention
row sums, MLP multiplication, an SGD update identity, a finite-difference
gradient, invariance of earlier logits to a changed future token, and assistant
loss-mask alignment". ``checks.py`` implements those; this file is what makes
them run on every test invocation, and ``test_blueprint_claims_are_all_covered``
maps each clause of that sentence onto the check that backs it, so the claim
cannot drift away from the code again.
"""

from __future__ import annotations

import unittest

from glassbox import checks

from . import fixtures

#: Every check ``run_all`` is expected to produce, in its running order.
#: Pinned so a check cannot be deleted or renamed and leave the suite reporting
#: "all passed" over a shorter list.
EXPECTED_CHECKS = (
    "causal_mask_zeros",
    "attention_rows_sum_to_one",
    "mlp_cell_is_a_dot_product",
    "future_token_invariance",
    "untrained_loss_equals_log_vocab",
    "finite_difference_matches_autograd",
    "sgd_update_identity",
    "gradient_demo_leaves_checkpoint_unchanged",
    "loss_mask_alignment",
    "kv_cache_equivalence",
    "inference_does_not_change_weights",
    "temperature_changes_the_distribution",
    "top_k_restricts_candidates",
    "lora_initial_adapter_is_a_noop",
    "lora_first_step_gradient_asymmetry",
    "lora_base_weights_stay_frozen",
    "lora_merge_equivalence",
    "lora_update_rank_is_bounded_by_r",
    "dpo_starts_at_ln2_with_zero_margin",
    "zero_variance_group_has_no_signal",
    "verifier_rejects_ambiguous_and_exploitative_output",
    "unknown_characters_map_to_unk_not_pad",
    "bpe_fitted_on_training_split_only",
    "verifiable_task_families_are_disjoint",
    "training_windows_stay_inside_one_document",
    "same_seed_reproduces_same_weights",
)

#: Each clause of the blueprint sentence, and the check that implements it.
BLUEPRINT_CLAIMS = {
    "causal-mask zeros": "causal_mask_zeros",
    "attention row sums": "attention_rows_sum_to_one",
    "MLP multiplication": "mlp_cell_is_a_dot_product",
    "an SGD update identity": "sgd_update_identity",
    "a finite-difference gradient": "finite_difference_matches_autograd",
    "invariance of earlier logits to a changed future token": "future_token_invariance",
    "assistant loss-mask alignment": "loss_mask_alignment",
}


class RunAllChecks(unittest.TestCase):
    """Runs the check battery once against the cached pretrained model."""

    results = None

    @classmethod
    def setUpClass(cls) -> None:
        model = fixtures.trained_model()
        cls.report = checks.run_all(model, fixtures.tokenizer(), fixtures.model_config())
        cls.results = {r["name"]: r for r in cls.report["results"]}

    def test_every_expected_check_ran(self) -> None:
        self.assertEqual(
            tuple(r["name"] for r in self.report["results"]),
            EXPECTED_CHECKS,
            "run_all produced a different set of checks than the suite pins",
        )

    def test_check_count_is_consistent(self) -> None:
        self.assertEqual(self.report["total"], len(EXPECTED_CHECKS))
        self.assertEqual(
            self.report["passed"] + self.report["failed"], self.report["total"]
        )

    def test_each_check_passes(self) -> None:
        for name in EXPECTED_CHECKS:
            with self.subTest(check=name):
                result = self.results[name]
                self.assertTrue(
                    result["passed"],
                    "check {!r} failed\n  description: {}\n  tolerance: {}\n"
                    "  measured: {}".format(
                        name,
                        result["description"],
                        result["tolerance"],
                        result["measured"],
                    ),
                )

    def test_all_passed_flag_agrees_with_the_individual_results(self) -> None:
        failed = [n for n, r in self.results.items() if not r["passed"]]
        self.assertEqual(failed, [], "failing checks: {}".format(failed))
        self.assertTrue(self.report["all_passed"])
        self.assertEqual(self.report["failed"], 0)

    def test_every_check_reports_measured_numbers_not_just_a_boolean(self) -> None:
        """A check that reports no numbers cannot be audited."""
        for name in EXPECTED_CHECKS:
            with self.subTest(check=name):
                self.assertTrue(
                    self.results[name]["measured"],
                    "check {!r} reported no measured values".format(name),
                )
                self.assertTrue(self.results[name]["description"].strip())

    def test_blueprint_claims_are_all_covered(self) -> None:
        for claim, name in BLUEPRINT_CLAIMS.items():
            with self.subTest(claim=claim):
                self.assertIn(
                    name,
                    self.results,
                    "the blueprint claims {!r} is asserted, but no check named "
                    "{!r} ran".format(claim, name),
                )
                self.assertTrue(self.results[name]["passed"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
