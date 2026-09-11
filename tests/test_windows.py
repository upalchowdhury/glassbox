"""Training windows must stay inside one document.

The interface claimed "training windows stay inside paragraphs" while the old
loop joined all eight paragraphs with no separator and slid a stride-7 window
across the seams, so the model was trained to predict the first character of
paragraph 4 from the last character of paragraph 3.

The test below is stronger than a substring check: it rebuilds each document's
token sequence and asserts every window is the exact slice
``ids[start : start + context + 1]`` of the document it is tagged with. A
substring test can pass on a window that spans two documents whenever the
spanning text happens to occur elsewhere.
"""

from __future__ import annotations

import unittest

from glassbox import data
from glassbox.train import document_windows

from . import fixtures


class WindowsRespectDocuments(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = fixtures.tokenizer()
        self.context = fixtures.model_config().context

    def _windows(self, split: str):
        return document_windows(
            self.tok, data.split_documents(split), self.context
        )

    def test_every_window_is_an_exact_slice_of_its_own_document(self) -> None:
        for split in ("train", "validation"):
            docs = {d.doc_id: d for d in data.split_documents(split)}
            windows = self._windows(split)
            self.assertTrue(windows, "no windows for split {!r}".format(split))
            for w in windows:
                with self.subTest(split=split, doc=w.doc_id, start=w.start):
                    self.assertIn(w.doc_id, docs)
                    full = self.tok.encode(
                        docs[w.doc_id].text, bos=True, eos=True
                    )
                    self.assertEqual(
                        w.ids,
                        full[w.start : w.start + self.context + 1],
                        "window is not a contiguous slice of {}".format(w.doc_id),
                    )

    def test_no_window_decodes_to_text_spanning_two_documents(self) -> None:
        """The weaker substring form of the same claim, kept as a second net."""
        docs = {d.doc_id: d.text for d in data.split_documents("train")}
        for w in self._windows("train"):
            decoded = self.tok.decode(w.ids)
            with self.subTest(doc=w.doc_id, start=w.start):
                self.assertIn(decoded, docs[w.doc_id])

    def test_windows_never_mix_train_and_validation(self) -> None:
        train_ids = {w.doc_id for w in self._windows("train")}
        val_ids = {w.doc_id for w in self._windows("validation")}
        self.assertEqual(train_ids & val_ids, set())
        self.assertTrue(all(i.startswith("train/") for i in train_ids))
        self.assertTrue(all(i.startswith("validation/") for i in val_ids))

    def test_every_window_starts_inside_its_document_and_has_two_tokens(self) -> None:
        for w in self._windows("train"):
            with self.subTest(doc=w.doc_id, start=w.start):
                self.assertGreaterEqual(w.start, 0)
                self.assertGreaterEqual(
                    len(w.ids), 2, "a window of one token scores nothing"
                )

    def test_the_first_window_of_each_document_begins_with_bos(self) -> None:
        by_doc = {}
        for w in self._windows("train"):
            by_doc.setdefault(w.doc_id, []).append(w)
        self.assertEqual(len(by_doc), len(data.split_documents("train")))
        for doc_id, windows in by_doc.items():
            first = min(windows, key=lambda w: w.start)
            with self.subTest(doc=doc_id):
                self.assertEqual(first.start, 0)
                self.assertEqual(first.ids[0], self.tok.bos_id)

    def test_eos_is_the_last_target_of_every_document(self) -> None:
        """EOS is a learned signal only if it is actually in the windows."""
        for doc in data.split_documents("train"):
            ids = self.tok.encode(doc.text, bos=True, eos=True)
            with self.subTest(doc=doc.doc_id):
                self.assertEqual(ids[-1], self.tok.eos_id)
        # Every window of a document that fits inside the context runs to the
        # end of that document, so EOS is the final target of each one. A
        # document longer than the context would be windowed in pieces, and
        # only the final piece would carry EOS - hence the length guard.
        lengths = {
            d.doc_id: len(self.tok.encode(d.text, bos=True, eos=True))
            for d in data.split_documents("train")
        }
        checked = 0
        for w in self._windows("train"):
            if lengths[w.doc_id] <= self.context + 1:
                checked += 1
                with self.subTest(doc=w.doc_id, start=w.start):
                    self.assertEqual(w.ids[-1], self.tok.eos_id)
        self.assertGreater(checked, 0, "no document fits inside the context")

    def test_window_count_matches_the_document_lengths(self) -> None:
        expected = sum(
            len(self.tok.encode(d.text, bos=True, eos=True)) - 1
            for d in data.split_documents("train")
        )
        self.assertEqual(len(self._windows("train")), expected)

    def test_splits_are_made_over_documents_not_characters(self) -> None:
        train = {d.text for d in data.split_documents("train")}
        validation = {d.text for d in data.split_documents("validation")}
        self.assertEqual(train & validation, set())
        self.assertEqual(len(train), 8)
        self.assertEqual(len(validation), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
