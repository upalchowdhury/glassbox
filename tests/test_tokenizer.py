"""Tokenizer invariants.

Two of these encode bugs that were live in the previous engine:

* ``<unk>`` was id 0, which was also ``<pad>``: an out-of-alphabet character
  silently became padding and the model was trained to predict it.
* the alphabet was built from a string that included the literal text
  ``"<pad><eos>"``, so ``'<'`` and ``'>'`` became ordinary characters while the
  chat template's ``':'`` was missing from the corpus the alphabet was derived
  from - and every ``':'`` in ``"user: ...\\nassistant: "`` encoded to <pad>.
"""

from __future__ import annotations

import unittest

from glassbox import data
from glassbox.tokenizer import (
    PAD,
    SPECIAL_TOKENS,
    BPETokenizer,
    CharTokenizer,
    fit_on_training_split,
)

from . import fixtures


class CharTokenizerAlphabet(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = fixtures.tokenizer()

    def test_specials_are_distinct_ids(self) -> None:
        ids = self.tok.special_ids
        self.assertEqual(len(set(ids)), len(ids), "special tokens share an id")

    def test_unk_is_not_pad(self) -> None:
        self.assertNotEqual(
            self.tok.unk_id,
            self.tok.pad_id,
            "<unk> and <pad> are the same id: an unknown character would be "
            "trained as padding",
        )

    def test_unknown_characters_encode_to_unk(self) -> None:
        ids = self.tok.encode("çæ ")
        self.assertEqual(ids, [self.tok.unk_id] * 3)
        self.assertNotIn(self.tok.pad_id, ids)

    def test_no_angle_brackets_in_the_alphabet(self) -> None:
        """'<' and '>' appear only inside special-token NAMES, never as chars."""
        for ch in ("<", ">"):
            with self.subTest(char=ch):
                self.assertNotIn(
                    ch,
                    self.tok.vocab,
                    "{!r} is in the alphabet: the special-token names leaked "
                    "into the character set".format(ch),
                )
        # 'p', 'a' and 'd' are ordinary characters; only the brackets are not.
        self.assertEqual(self.tok.encode("<"), [self.tok.unk_id])
        self.assertEqual(self.tok.encode(">"), [self.tok.unk_id])
        self.assertEqual(
            self.tok.encode("<pad>"),
            [self.tok.unk_id]
            + self.tok.encode("pad")
            + [self.tok.unk_id],
        )

    def test_chat_template_punctuation_is_in_the_alphabet(self) -> None:
        """Every character of the chat template must be a real token.

        ':' was absent from the old alphabet, so the template separator encoded
        to <pad> and the model never saw the boundary it was supposed to learn.
        """
        template_chars = set(data.PROMPT_TEMPLATE.replace("{prompt}", ""))
        self.assertIn(":", template_chars)
        for ch in sorted(template_chars):
            with self.subTest(char=ch):
                self.assertIn(ch, self.tok.vocab)
                self.assertNotEqual(
                    self.tok.encode(ch),
                    [self.tok.pad_id],
                    "{!r} silently encodes to <pad>".format(ch),
                )
        formatted = data.format_prompt("how many red stones does pip have?")
        self.assertNotIn(self.tok.pad_id, self.tok.encode(formatted))
        self.assertNotIn(self.tok.unk_id, self.tok.encode(formatted))

    def test_every_character_the_engine_uses_is_representable(self) -> None:
        ids = self.tok.encode(data.all_text())
        self.assertNotIn(self.tok.unk_id, ids)
        self.assertNotIn(self.tok.pad_id, ids)

    def test_specials_are_never_single_characters(self) -> None:
        for tok_name in SPECIAL_TOKENS:
            self.assertGreater(len(tok_name), 1)
            self.assertEqual(self.tok.vocab.count(tok_name), 1)


class CharTokenizerRoundTrip(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = fixtures.tokenizer()

    def _texts(self):
        yield from (d.text for d in data.documents())
        for family in data.instruction_families().values():
            for item in family:
                yield data.format_prompt(item.prompt)
                yield item.answer
        for pair in data.PREFERENCE_PAIRS:
            yield pair.chosen
            yield pair.rejected
        for family in data.task_families(8).values():
            for task in family:
                yield data.format_prompt(task.prompt)

    def test_round_trip_is_exact(self) -> None:
        for text in self._texts():
            with self.subTest(text=text[:40]):
                self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_round_trip_with_bos_and_eos_strips_only_the_specials(self) -> None:
        text = "pip has two red stones."
        ids = self.tok.encode(text, bos=True, eos=True)
        self.assertEqual(ids[0], self.tok.bos_id)
        self.assertEqual(ids[-1], self.tok.eos_id)
        self.assertEqual(self.tok.decode(ids), text)
        self.assertEqual(
            self.tok.decode(ids, keep_specials=True), "<bos>" + text + "<eos>"
        )

    def test_decode_ignores_out_of_range_ids(self) -> None:
        self.assertEqual(self.tok.decode([-1, 10 ** 6]), "")

    def test_vocab_size_and_ordering(self) -> None:
        self.assertEqual(len(self.tok), len(self.tok.vocab))
        self.assertEqual(tuple(self.tok.vocab[:4]), SPECIAL_TOKENS)
        chars = self.tok.vocab[4:]
        self.assertEqual(chars, sorted(chars), "character ids are not sorted")


class BPEFitIsolation(unittest.TestCase):
    MERGES = 40

    def test_supported_fit_uses_training_documents_only(self) -> None:
        train_only = BPETokenizer.fit(
            [d.text for d in data.split_documents("train")], num_merges=self.MERGES
        )
        supported = fit_on_training_split(num_merges=self.MERGES)
        self.assertEqual(
            [m.token for m in train_only.merges],
            [m.token for m in supported.merges],
            "fit_on_training_split saw something other than the train split",
        )

    def test_fitting_on_everything_gives_a_different_vocabulary(self) -> None:
        """If this ever stops differing, the leak test below proves nothing."""
        train_only = fit_on_training_split(num_merges=self.MERGES)
        everything = BPETokenizer.fit(
            [d.text for d in data.documents()], num_merges=self.MERGES
        )
        self.assertNotEqual(
            [m.token for m in train_only.merges],
            [m.token for m in everything.merges],
            "fitting on train and on train+validation produced identical "
            "merges, so this data cannot demonstrate a tokenizer leak",
        )
        self.assertEqual(len(train_only.merges), self.MERGES)
        self.assertEqual(len(everything.merges), self.MERGES)

    def test_merge_ranks_are_dense_and_ordered(self) -> None:
        bpe = fit_on_training_split(num_merges=self.MERGES)
        self.assertEqual([m.rank for m in bpe.merges], list(range(len(bpe.merges))))
        for m in bpe.merges:
            self.assertEqual(m.token, m.left + m.right)
            self.assertGreaterEqual(m.count, 2)

    def test_bpe_specials_match_the_character_tokenizer(self) -> None:
        bpe = fit_on_training_split(num_merges=self.MERGES)
        self.assertEqual(tuple(bpe.vocab[:4]), SPECIAL_TOKENS)
        self.assertNotEqual(bpe.unk_id, bpe.pad_id)
        self.assertEqual(bpe.vocab[bpe.pad_id], PAD)

    def test_bpe_uses_fewer_tokens_than_characters_on_trained_text(self) -> None:
        bpe = fit_on_training_split(num_merges=60)
        sample = "pip has two red stones."
        self.assertLess(
            len(bpe.encode(sample)), len(fixtures.tokenizer().encode(sample))
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
