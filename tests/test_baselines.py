"""The trained model must beat a bigram count model on validation.

This is the test that makes "the loss went down" mean something. Without it, a
curve that falls from a broken initialisation to slightly-better-than-guessing
reads as success: the OLD engine reached 3.0876 on validation, which LOST to
both count models, and nothing in the course said so.

The baselines are fitted on the training split only and scored on exactly the
positions the model is scored on, so the three numbers are comparable.
"""

from __future__ import annotations

import math
import unittest

from glassbox import data
from glassbox.evaluate import baseline_losses, retention
from glassbox.train import document_windows, mean_token_loss

from . import fixtures


class Baselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = fixtures.tokenizer()
        cls.cfg = fixtures.model_config()
        cls.baselines = baseline_losses(cls.tok, cls.cfg.context)
        model = fixtures.trained_model()
        cls.model_losses = retention(model, cls.tok)
        cls.untrained_losses = retention(fixtures.untrained_model(), cls.tok)

    def test_uniform_baseline_is_exactly_ln_vocab(self) -> None:
        self.assertAlmostEqual(
            self.baselines["validation"]["uniform"], math.log(len(self.tok)), places=9
        )

    def test_the_baselines_are_ordered_as_the_course_claims(self) -> None:
        val = self.baselines["validation"]
        self.assertLess(val["bigram_laplace"], val["unigram_laplace"])
        self.assertLess(val["unigram_laplace"], val["uniform"])

    def test_baselines_are_scored_on_the_same_positions_as_the_model(self) -> None:
        windows = document_windows(
            self.tok, data.split_documents("validation"), self.cfg.context
        )
        expected = sum(len(w.ids) - 1 for w in windows)
        self.assertEqual(self.baselines["validation"]["scored_positions"], expected)

    def test_the_trained_model_beats_the_bigram_baseline_on_validation(self) -> None:
        model_loss = self.model_losses["validation_prose_loss"]
        bigram = self.baselines["validation"]["bigram_laplace"]
        self.assertLess(
            model_loss,
            bigram,
            "after {} pretraining steps the model scores {:.4f} on validation "
            "but the bigram Laplace baseline scores {:.4f}: a transformer that "
            "does not beat the bigram model has not earned its parameters".format(
                fixtures.PRETRAIN_STEPS, model_loss, bigram
            ),
        )

    def test_the_trained_model_beats_the_unigram_and_uniform_baselines_too(self) -> None:
        model_loss = self.model_losses["validation_prose_loss"]
        val = self.baselines["validation"]
        self.assertLess(model_loss, val["unigram_laplace"])
        self.assertLess(model_loss, val["uniform"])

    def test_an_untrained_model_loses_to_both_count_models(self) -> None:
        """The comparison has to be able to fail, or it proves nothing."""
        untrained = self.untrained_losses["validation_prose_loss"]
        val = self.baselines["validation"]
        self.assertGreater(untrained, val["bigram_laplace"])
        self.assertGreater(untrained, val["unigram_laplace"])
        self.assertAlmostEqual(untrained, val["uniform"], delta=0.05)

    def test_training_actually_lowered_the_validation_loss(self) -> None:
        self.assertLess(
            self.model_losses["validation_prose_loss"],
            self.untrained_losses["validation_prose_loss"],
        )

    def test_validation_loss_is_higher_than_train_loss(self) -> None:
        """Stated so the suite records the gap rather than implying there is none."""
        self.assertGreater(
            self.model_losses["validation_prose_loss"],
            self.model_losses["train_prose_loss"],
        )

    def test_the_pretraining_history_is_consistent_with_the_measured_loss(self) -> None:
        history = fixtures.pretrain_history()
        final = history["final"]
        self.assertEqual(final["step"], fixtures.PRETRAIN_STEPS)
        self.assertAlmostEqual(
            final["validation"],
            self.model_losses["validation_prose_loss"],
            places=5,
        )
        self.assertAlmostEqual(
            final["train"], self.model_losses["train_prose_loss"], places=5
        )

    def test_the_first_recorded_loss_is_the_untrained_one(self) -> None:
        history = fixtures.pretrain_history()["history"]
        self.assertEqual(history[0]["step"], 0)
        self.assertAlmostEqual(
            history[0]["validation"],
            self.untrained_losses["validation_prose_loss"],
            places=5,
        )

    def test_a_best_validation_step_is_recorded(self) -> None:
        history = fixtures.pretrain_history()
        best = history["best_validation"]
        recorded = [h["validation"] for h in history["history"]]
        self.assertAlmostEqual(best["validation"], min(recorded), places=9)

    def test_mean_token_loss_is_token_weighted_not_batch_averaged(self) -> None:
        """A short row must not count as much as a long one."""
        model = fixtures.trained_model()
        short = self.tok.encode("pip.", bos=True, eos=True)
        long = self.tok.encode(
            data.split_documents("train")[0].text, bos=True, eos=True
        )
        loss_short = mean_token_loss(model, [short], self.tok.pad_id)
        loss_long = mean_token_loss(model, [long], self.tok.pad_id)
        combined = mean_token_loss(model, [short, long], self.tok.pad_id)
        n_short, n_long = len(short) - 1, len(long) - 1
        expected = (loss_short * n_short + loss_long * n_long) / (n_short + n_long)
        self.assertAlmostEqual(combined, expected, places=4)
        unweighted = (loss_short + loss_long) / 2
        self.assertNotAlmostEqual(combined, unweighted, places=4)

    def test_the_note_names_the_comparison(self) -> None:
        self.assertIn("bigram", self.baselines["note"])
        self.assertEqual(self.baselines["alpha"], 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
