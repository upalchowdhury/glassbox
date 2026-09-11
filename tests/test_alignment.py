"""Shift alignment: ``pad_to_batch``, ``sft_row`` and the loss mask.

An off-by-one here is invisible in a loss curve - the curve still goes down -
and it destroys the lesson, because the model is then being trained to predict
the token it was just shown.

The two properties that pin it:

* the first SCORED target is produced by the LAST PROMPT position, not by the
  first answer position;
* the scored targets are exactly the answer tokens plus EOS, and nothing else.
"""

from __future__ import annotations

import unittest

import torch

from glassbox import data
from glassbox.train import (
    IGNORE,
    masked_rows_to_batch,
    pad_to_batch,
    mask_explanation,
    sft_row,
)

from . import fixtures


class PadToBatch(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = fixtures.tokenizer()

    def test_targets_are_inputs_shifted_by_exactly_one(self) -> None:
        row = self.tok.encode("pip has two", bos=True, eos=True)
        x, y = pad_to_batch([row], self.tok.pad_id)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.size(1), len(row) - 1)
        for i in range(len(row) - 1):
            with self.subTest(position=i):
                self.assertEqual(int(x[0, i]), row[i])
                self.assertEqual(int(y[0, i]), row[i + 1])
        self.assertNotEqual(
            [int(v) for v in x[0]],
            [int(v) for v in y[0]],
            "inputs and targets are identical: the shift is missing",
        )

    def test_no_position_is_asked_to_predict_its_own_input(self) -> None:
        rows = [
            self.tok.encode(d.text, bos=True, eos=True)
            for d in data.split_documents("train")
        ]
        x, y = pad_to_batch(rows, self.tok.pad_id)
        scored = y != IGNORE
        same = ((x == y) & scored).sum()
        # Equality can happen legitimately ("oo" in "stones" has none, but
        # doubled letters do occur), so compare against the true count from the
        # raw rows rather than demanding zero.
        expected = sum(
            1 for row in rows for a, b in zip(row, row[1:]) if a == b
        )
        self.assertEqual(int(same), expected)

    def test_padding_is_excluded_from_the_loss(self) -> None:
        short = self.tok.encode("pip", bos=True, eos=True)
        long = self.tok.encode("pip has two red stones", bos=True, eos=True)
        x, y = pad_to_batch([short, long], self.tok.pad_id)
        self.assertEqual(x.size(1), len(long) - 1)
        scored = int((y[0] != IGNORE).sum())
        self.assertEqual(scored, len(short) - 1)
        self.assertTrue(torch.all(y[0, scored:] == IGNORE))
        self.assertTrue(torch.all(x[0, scored:] == self.tok.pad_id))

    def test_a_longer_batch_mate_cannot_change_another_rows_targets(self) -> None:
        row = self.tok.encode("pip has two", bos=True, eos=True)
        alone_x, alone_y = pad_to_batch([row], self.tok.pad_id)
        mate = self.tok.encode("mira visits the garden each morning.", bos=True)
        both_x, both_y = pad_to_batch([row, mate], self.tok.pad_id)
        n = alone_x.size(1)
        self.assertTrue(torch.equal(alone_x[0], both_x[0, :n]))
        self.assertTrue(torch.equal(alone_y[0], both_y[0, :n]))


class SFTRowAlignment(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = fixtures.tokenizer()
        self.item = data.SFT_TRAIN[0]
        self.ids, self.targets = sft_row(self.tok, self.item, supervise_eos=True)
        self.prompt_ids = self.tok.encode(
            data.format_prompt(self.item.prompt), bos=True
        )

    def test_row_shape(self) -> None:
        self.assertEqual(len(self.ids), len(self.targets))
        answer_ids = self.tok.encode(self.item.answer) + [self.tok.eos_id]
        self.assertEqual(self.ids, self.prompt_ids + answer_ids)

    def test_the_prompt_is_never_scored(self) -> None:
        self.assertEqual(
            self.targets[: len(self.prompt_ids)], [IGNORE] * len(self.prompt_ids)
        )

    def test_scored_targets_are_exactly_the_answer_plus_eos(self) -> None:
        scored = [t for t in self.targets if t != IGNORE]
        expected = self.tok.encode(self.item.answer) + [self.tok.eos_id]
        self.assertEqual(
            scored,
            expected,
            "scored {} but expected {}".format(
                [self.tok.display(t) for t in scored],
                [self.tok.display(t) for t in expected],
            ),
        )
        self.assertIn(self.tok.eos_id, scored)
        self.assertEqual(scored.count(self.tok.eos_id), 1)

    def test_first_scored_target_is_produced_by_the_last_prompt_position(self) -> None:
        """After the shift, the first scored row index must be prompt_len - 1."""
        x, y = masked_rows_to_batch([(self.ids, self.targets)], self.tok.pad_id)
        scored_positions = [int(i) for i in torch.nonzero(y[0] != IGNORE).flatten()]
        first = scored_positions[0]
        self.assertEqual(
            first,
            len(self.prompt_ids) - 1,
            "the first scored prediction comes from position {} but the last "
            "prompt position is {}".format(first, len(self.prompt_ids) - 1),
        )
        # That position's INPUT is the final prompt token ...
        self.assertEqual(int(x[0, first]), self.prompt_ids[-1])
        # ... and its TARGET is the first answer token.
        self.assertEqual(
            int(y[0, first]), self.tok.encode(self.item.answer)[0]
        )

    def test_scored_positions_are_contiguous_to_the_end_of_the_row(self) -> None:
        x, y = masked_rows_to_batch([(self.ids, self.targets)], self.tok.pad_id)
        scored = [int(i) for i in torch.nonzero(y[0] != IGNORE).flatten()]
        self.assertEqual(scored, list(range(scored[0], scored[0] + len(scored))))
        self.assertEqual(scored[-1], y.size(1) - 1)
        answer_ids = self.tok.encode(self.item.answer) + [self.tok.eos_id]
        self.assertEqual(len(scored), len(answer_ids))
        self.assertEqual([int(y[0, i]) for i in scored], answer_ids)

    def test_unsupervised_eos_drops_exactly_one_scored_token(self) -> None:
        ids, targets = sft_row(self.tok, self.item, supervise_eos=False)
        scored = [t for t in targets if t != IGNORE]
        self.assertEqual(scored, self.tok.encode(self.item.answer))
        self.assertNotIn(self.tok.eos_id, scored)
        self.assertEqual(len(self.ids) - len(ids), 1)

    def test_every_instruction_family_aligns(self) -> None:
        for name, items in data.instruction_families().items():
            for item in items:
                with self.subTest(family=name, prompt=item.prompt):
                    ids, targets = sft_row(self.tok, item, supervise_eos=True)
                    prompt_len = len(
                        self.tok.encode(data.format_prompt(item.prompt), bos=True)
                    )
                    self.assertEqual(len(ids), len(targets))
                    self.assertEqual(
                        targets[:prompt_len], [IGNORE] * prompt_len
                    )
                    self.assertEqual(
                        [t for t in targets if t != IGNORE],
                        self.tok.encode(item.answer) + [self.tok.eos_id],
                    )
                    # The unshifted target list is aligned to the ids, so from
                    # the first scored index onward the two must coincide.
                    self.assertEqual(targets[prompt_len:], ids[prompt_len:])


class MaskExplanationMatchesTheTrainedMask(unittest.TestCase):
    """What the interface renders is derived from the training row itself."""

    def test_displayed_mask_equals_the_optimised_mask(self) -> None:
        tok = fixtures.tokenizer()
        item = data.SFT_TRAIN[0]
        ids, targets = sft_row(tok, item, supervise_eos=True)
        explained = mask_explanation(tok, item, supervise_eos=True)
        x, y = masked_rows_to_batch([(ids, targets)], tok.pad_id)
        self.assertEqual(len(explained["positions"]), y.size(1))
        for position in explained["positions"]:
            i = position["position"]
            with self.subTest(position=i):
                self.assertEqual(position["input_id"], int(x[0, i]))
                if position["scored"]:
                    self.assertEqual(position["target_id"], int(y[0, i]))
                else:
                    self.assertIsNone(position["target_id"])
                    self.assertEqual(int(y[0, i]), IGNORE)
        self.assertEqual(
            explained["num_scored"], int((y != IGNORE).sum())
        )
        self.assertEqual(
            explained["num_scored"] + explained["num_ignored"], y.size(1)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
