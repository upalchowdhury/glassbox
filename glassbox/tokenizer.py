"""Tokenizers: one character tokenizer for the microscope, one learned
subword tokenizer for the lesson that contrasts them.

Two things here exist specifically to stop a teaching tool from lying:

* ``CharTokenizer`` has a real ``<unk>``. The original engine mapped every
  unknown character to id 0, which was ``<pad>`` - so a typo silently became
  padding and the model was trained to predict it.
* ``BPETokenizer`` is fitted on the TRAINING SPLIT ONLY and records every
  merge it learned. Fitting a tokenizer on validation text is a real and
  easy-to-miss leak, so the fit function refuses to see anything else.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from . import data

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS: Tuple[str, ...] = (PAD, BOS, EOS, UNK)


class CharTokenizer:
    """One character = one token. The simplest thing that is still honest.

    Ids are arbitrary categories. Id 20 is not "twice" id 10, and the ordering
    carries no meaning beyond "specials first, then characters sorted".
    """

    def __init__(self, alphabet: Iterable[str] | None = None):
        text = data.all_text() if alphabet is None else "".join(alphabet)
        chars = sorted(set(text))
        # A special token is a multi-character name, never a member of the
        # alphabet, so there is nothing to filter out here. This is the bug the
        # previous engine had: it appended the literal string "<pad><eos>" to
        # the alphabet source and so taught the model about '<' and '>'.
        for c in chars:
            if len(c) != 1:  # pragma: no cover - defensive
                raise ValueError(f"alphabet entry is not one character: {c!r}")
        self.vocab: List[str] = list(SPECIAL_TOKENS) + chars
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def special_ids(self) -> Tuple[int, ...]:
        return (self.pad_id, self.bos_id, self.eos_id, self.unk_id)

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        ids = [self.stoi.get(c, self.unk_id) for c in text]
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], keep_specials: bool = False) -> str:
        out: List[str] = []
        for i in ids:
            if i < 0 or i >= len(self.vocab):
                continue
            tok = self.vocab[i]
            if tok in SPECIAL_TOKENS:
                if keep_specials:
                    out.append(tok)
                continue
            out.append(tok)
        return "".join(out)

    def display(self, token_id: int) -> str:
        """Printable name for one id, for matrix row labels."""
        tok = self.vocab[token_id]
        return {" ": "␠", "\n": "↵"}.get(tok, tok)


# --------------------------------------------------------------------------
# A minimal byte-pair-encoding tokenizer, fitted on the training split.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Merge:
    """One learned merge, with the count that justified it."""

    rank: int
    left: str
    right: str
    token: str
    count: int


class BPETokenizer:
    """Learned subword vocabulary, built by repeatedly merging the most
    frequent adjacent pair.

    Kept deliberately small and slow-but-readable: a learner should be able to
    follow the merge table and see that "stone" became one token because it
    occurred often, not because anyone listed it in advance.
    """

    def __init__(self, vocab: Sequence[str], merges: Sequence[Merge]):
        self.vocab: List[str] = list(vocab)
        self.merges: List[Merge] = list(merges)
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]
        self._ranks: Dict[Tuple[str, str], int] = {
            (m.left, m.right): m.rank for m in self.merges
        }

    def __len__(self) -> int:
        return len(self.vocab)

    @classmethod
    def fit(cls, texts: Sequence[str], num_merges: int = 60) -> "BPETokenizer":
        """Learn merges from ``texts`` only.

        Callers must pass the training split. :func:`fit_on_training_split`
        below is the supported entry point; it exists so no call site has to
        remember which documents are safe to fit on.
        """
        words: Counter = Counter()
        for t in texts:
            for w in t.replace("\n", " ").split(" "):
                if w:
                    words[w] += 1

        # Each word is a tuple of symbols; '</w>' marks a word boundary so
        # "stone" and "stones" can share a prefix without merging across words.
        corpus: Dict[Tuple[str, ...], int] = {
            tuple(list(w) + ["</w>"]): c for w, c in words.items()
        }
        alphabet = sorted({s for word in corpus for s in word})
        vocab: List[str] = list(SPECIAL_TOKENS) + alphabet
        merges: List[Merge] = []

        for rank in range(num_merges):
            pairs: Counter = Counter()
            for word, freq in corpus.items():
                for a, b in zip(word, word[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            (left, right), count = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
            if count < 2:
                break
            token = left + right
            merges.append(Merge(rank, left, right, token, count))
            if token not in vocab:
                vocab.append(token)
            corpus = {
                cls._apply_merge(word, left, right): freq
                for word, freq in corpus.items()
            }
        return cls(vocab, merges)

    @staticmethod
    def _apply_merge(
        word: Tuple[str, ...], left: str, right: str
    ) -> Tuple[str, ...]:
        out: List[str] = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i] == left and word[i + 1] == right:
                out.append(left + right)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def tokenize_word(self, word: str) -> List[str]:
        symbols: Tuple[str, ...] = tuple(list(word) + ["</w>"])
        while True:
            best = None
            for a, b in zip(symbols, symbols[1:]):
                r = self._ranks.get((a, b))
                if r is not None and (best is None or r < best[0]):
                    best = (r, a, b)
            if best is None:
                break
            symbols = self._apply_merge(symbols, best[1], best[2])
        return list(symbols)

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        ids: List[int] = [self.bos_id] if bos else []
        for word in text.replace("\n", " ").split(" "):
            if not word:
                continue
            for piece in self.tokenize_word(word):
                ids.append(self.stoi.get(piece, self.unk_id))
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], keep_specials: bool = False) -> str:
        pieces: List[str] = []
        for i in ids:
            if i < 0 or i >= len(self.vocab):
                continue
            tok = self.vocab[i]
            if tok in SPECIAL_TOKENS:
                if keep_specials:
                    pieces.append(tok)
                continue
            pieces.append(tok)
        return "".join(pieces).replace("</w>", " ").strip()


def fit_on_training_split(num_merges: int = 60) -> BPETokenizer:
    """The only supported way to build the subword tokenizer."""
    train_only = [d.text for d in data.split_documents("train")]
    return BPETokenizer.fit(train_only, num_merges=num_merges)


def leaked_subwords(bpe: BPETokenizer) -> List[str]:
    """Multi-character tokens that appear in validation text.

    Returns the overlap a learner should EXPECT: shared words like "stone"
    legitimately occur in both splits. The point of the lesson is that this
    list is evidence about the data, and that the tokenizer was still fitted
    without reading the validation split.
    """
    val = " ".join(d.text for d in data.split_documents("validation"))
    return sorted(
        {
            t
            for t in bpe.vocab
            if t not in SPECIAL_TOKENS and len(t.replace("</w>", "")) > 1
            and t.replace("</w>", "") in val
        }
    )
