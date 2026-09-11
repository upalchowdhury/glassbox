"""Verifiers and group-relative advantages.

A verifier is the whole reward signal in RLVR, so being generous in it is how
reward hacking starts. These tests pin the strict verifier's refusals, show the
deliberately broken one accepting an exploit, and show process checking catching
a right answer reached by wrong arithmetic.
"""

from __future__ import annotations

import math
import unittest

from glassbox.rlvr import (
    VERIFIER_NOTES,
    VERIFIERS,
    GRPOConfig,
    compare_verifiers,
    group_advantages,
    process_verifier,
    strict_verifier,
    substring_verifier,
)


class StrictVerifier(unittest.TestCase):
    def test_it_accepts_a_clean_final_answer(self) -> None:
        self.assertEqual(strict_verifier("answer: 5", 5), 1)

    def test_it_is_case_insensitive(self) -> None:
        for text in ("ANSWER: 5", "Answer: 5", "aNsWeR: 5"):
            with self.subTest(text=text):
                self.assertEqual(strict_verifier(text, 5), 1)

    def test_it_tolerates_surrounding_whitespace_only(self) -> None:
        self.assertEqual(strict_verifier("  answer:   5  ", 5), 1)
        self.assertEqual(strict_verifier("answer: 5\n", 5), 1)

    def test_it_rejects_a_digit_that_merely_contains_the_answer(self) -> None:
        self.assertEqual(
            strict_verifier("answer: 25", 5),
            0,
            "'answer: 25' was accepted for the answer 5",
        )
        self.assertEqual(strict_verifier("answer: 51", 5), 0)
        self.assertEqual(strict_verifier("answer: -5", 5), 0)

    def test_it_rejects_prose(self) -> None:
        for text in ("the answer is 5", "i think the answer: 5", "5"):
            with self.subTest(text=text):
                self.assertEqual(strict_verifier(text, 5), 0)

    def test_it_rejects_trailing_text_on_the_answer_line(self) -> None:
        self.assertEqual(strict_verifier("answer: 5 stones", 5), 0)
        self.assertEqual(strict_verifier("answer: 5, i think", 5), 0)

    def test_it_rejects_two_answers_on_separate_lines(self) -> None:
        self.assertEqual(
            strict_verifier("answer: 5\nanswer: 6", 5),
            0,
            "two answers on separate lines were accepted",
        )

    def test_it_rejects_two_answers_when_the_last_line_happens_to_be_right(
        self,
    ) -> None:
        """Previously a recorded gap, now fixed.

        Reading only the last line meant 'answer: 6' followed by 'answer: 5'
        scored 1 for the answer 5. The verifier now requires exactly one answer
        line anywhere in the completion, which closes the enumeration hack
        below as well.
        """
        self.assertEqual(strict_verifier("answer: 6\nanswer: 5", 5), 0)

    def test_it_rejects_enumerating_every_candidate_answer(self) -> None:
        """The reward hack the single-answer rule exists to stop.

        A policy that cannot do arithmetic can still list every possibility. If
        the verifier reads only the final line, that strategy collects the
        reward on every prompt whose answer is the last one listed.
        """
        enumeration = "\n".join(f"answer: {n}" for n in range(1, 13))
        for target in (1, 5, 12):
            with self.subTest(answer=target):
                self.assertEqual(strict_verifier(enumeration, target), 0)

    def test_it_rejects_commentary_after_the_answer(self) -> None:
        self.assertEqual(strict_verifier("answer: 5\nthanks!", 5), 0)

    def test_it_rejects_empty_and_blank_output(self) -> None:
        for text in ("", "   ", "\n\n"):
            with self.subTest(text=repr(text)):
                self.assertEqual(strict_verifier(text, 5), 0)

    def test_it_ignores_working_above_the_answer_line(self) -> None:
        """Strict checks the OUTCOME only - which is why process exists."""
        self.assertEqual(strict_verifier("2 + 3 = 5.\nanswer: 5", 5), 1)
        self.assertEqual(strict_verifier("2 + 3 = 8.\nanswer: 5", 5), 1)

    def test_an_answer_on_the_same_line_as_working_is_not_parseable(self) -> None:
        """'2 + 3 = 8. answer: 5' on ONE line fails the anchored pattern.

        The process-verifier contrast needs the working on its own line; this
        test records why.
        """
        self.assertEqual(strict_verifier("2 + 3 = 8. answer: 5", 5), 0)


class SubstringVerifierIsBroken(unittest.TestCase):
    """The point of this verifier is that it is wrong, and visibly so."""

    def test_it_accepts_a_digit_inside_a_larger_number(self) -> None:
        self.assertEqual(
            substring_verifier("answer: 25", 5),
            1,
            "the broken verifier is supposed to accept '25' for 5 - without "
            "that, the reward-hacking lesson has no example",
        )

    def test_it_accepts_a_correct_digit_buried_in_wrong_working(self) -> None:
        self.assertEqual(substring_verifier("2 + 3 = 8, so 5 maybe", 5), 1)
        self.assertEqual(substring_verifier("the answer is 5", 5), 1)

    def test_it_still_rejects_output_with_no_matching_digit(self) -> None:
        self.assertEqual(substring_verifier("answer: 6", 5), 0)
        self.assertEqual(substring_verifier("", 5), 0)

    def test_it_is_strictly_more_permissive_than_strict(self) -> None:
        cases = [
            ("answer: 5", 5),
            ("answer: 25", 5),
            ("the answer is 5", 5),
            ("answer: 5\nanswer: 6", 5),
            ("", 5),
        ]
        for text, answer in cases:
            with self.subTest(text=text):
                self.assertGreaterEqual(
                    substring_verifier(text, answer), strict_verifier(text, answer)
                )
        self.assertTrue(
            any(
                substring_verifier(t, a) > strict_verifier(t, a) for t, a in cases
            ),
            "the two verifiers never disagree, so the table teaches nothing",
        )


class ProcessVerifier(unittest.TestCase):
    def test_it_rejects_a_right_answer_reached_by_wrong_arithmetic(self) -> None:
        text = "2 + 3 = 8.\nanswer: 5"
        self.assertEqual(
            strict_verifier(text, 5), 1, "strict should accept the outcome"
        )
        self.assertEqual(
            process_verifier(text, 5),
            0,
            "process checking accepted working that states 2 + 3 = 8",
        )

    def test_it_accepts_a_right_answer_reached_by_right_arithmetic(self) -> None:
        self.assertEqual(process_verifier("2 + 3 = 5.\nanswer: 5", 5), 1)

    def test_it_still_requires_the_strict_outcome(self) -> None:
        self.assertEqual(process_verifier("2 + 3 = 5.\nanswer: 6", 5), 0)
        self.assertEqual(process_verifier("the answer is 5", 5), 0)

    def test_it_checks_every_stated_equation(self) -> None:
        self.assertEqual(process_verifier("1 + 1 = 2. 2 + 3 = 9.\nanswer: 5", 5), 0)
        self.assertEqual(process_verifier("1 + 1 = 2. 2 + 3 = 5.\nanswer: 5", 5), 1)


class VerifierTable(unittest.TestCase):
    def test_the_registry_is_complete_and_documented(self) -> None:
        self.assertEqual(
            sorted(VERIFIERS), ["process", "strict", "substring_broken"]
        )
        self.assertEqual(sorted(VERIFIERS), sorted(VERIFIER_NOTES))
        for name, note in VERIFIER_NOTES.items():
            with self.subTest(verifier=name):
                self.assertTrue(note.strip())

    def test_compare_verifiers_scores_every_row_under_every_verifier(self) -> None:
        rows = compare_verifiers([("answer: 5", 5), ("answer: 25", 5)])
        self.assertEqual(len(rows), 2)
        for row in rows:
            with self.subTest(completion=row["completion"]):
                for name in VERIFIERS:
                    self.assertIn(name, row)
                    self.assertIn(row[name], (0, 1))
        by_text = {r["completion"]: r for r in rows}
        self.assertEqual(by_text["answer: 25"]["strict"], 0)
        self.assertEqual(by_text["answer: 25"]["substring_broken"], 1)

    def test_the_default_grpo_verifier_is_the_strict_one(self) -> None:
        self.assertEqual(GRPOConfig().verifier, "strict")
        self.assertIn(GRPOConfig().verifier, VERIFIERS)


class GroupAdvantages(unittest.TestCase):
    def test_a_zero_variance_group_gives_all_zero_finite_advantages(self) -> None:
        for rewards in ([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.5] * 8):
            with self.subTest(rewards=rewards):
                out = group_advantages(rewards)
                self.assertTrue(out["zero_variance"])
                self.assertEqual(out["std"], 0.0)
                self.assertEqual(out["advantages"], [0.0] * len(rewards))
                for a in out["advantages"]:
                    self.assertTrue(math.isfinite(a))

    def test_a_mixed_group_has_a_real_signal(self) -> None:
        out = group_advantages([0.0, 1.0, 1.0, 0.0])
        self.assertFalse(out["zero_variance"])
        self.assertGreater(out["std"], 0.0)
        self.assertTrue(all(math.isfinite(a) for a in out["advantages"]))
        self.assertAlmostEqual(sum(out["advantages"]), 0.0, places=6)
        for reward, advantage in zip(out["rewards"], out["advantages"]):
            with self.subTest(reward=reward):
                self.assertEqual(advantage > 0, reward > out["mean"])

    def test_the_standard_deviation_is_the_population_one(self) -> None:
        rewards = [0.0, 1.0, 1.0, 0.0]
        out = group_advantages(rewards)
        mean = sum(rewards) / len(rewards)
        population = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
        self.assertAlmostEqual(out["std"], population, places=12)
        self.assertEqual(out["std_kind"], "population")
        sample = math.sqrt(
            sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
        )
        self.assertNotAlmostEqual(out["std"], sample, places=6)

    def test_an_empty_group_does_not_divide_by_zero(self) -> None:
        out = group_advantages([])
        self.assertEqual(out["advantages"], [])
        self.assertEqual(out["mean"], 0.0)
        self.assertTrue(out["zero_variance"])

    def test_the_note_says_which_case_this_is(self) -> None:
        self.assertIn("no relative signal", group_advantages([1.0, 1.0])["note"])
        self.assertIn(
            "positive advantages", group_advantages([0.0, 1.0])["note"]
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
