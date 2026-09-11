"""Glassbox data: one small world, honest splits, explicit provenance.

Everything a learner trains on lives here. Three rules this module enforces,
because breaking them silently is the most common way a teaching tool lies:

1. Splits are made over DOCUMENTS, before any windowing. A training window can
   never straddle two documents, so the model is never asked to predict the
   first character of paragraph 4 from the last character of paragraph 3.
2. Every document carries an id and a sha256, so any recorded number can be
   traced back to the exact text that produced it.
3. Held-out instruction data is kept in three clearly separated families -
   exact repeats, paraphrases of trained facts, and facts never asked during
   SFT - so "it answered one trained question" can never be reported as
   "instruction following works".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Document:
    """One paragraph of the training world, with provenance."""

    doc_id: str
    split: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Pretraining corpus. Original teaching text written for this course.
# --------------------------------------------------------------------------

_TRAIN_PARAGRAPHS = [
    "pip is a small robot in a quiet garden. pip has two red stones and three blue stones. there are five stones in all.",
    "the red stones sit in a round box. the blue stones sit in a square box. pip opens the round box and sees two stones.",
    "mira visits the garden each morning. mira gives pip one green leaf. pip puts the green leaf beside the round box.",
    "a little lamp lights the garden at night. when the sun rises, pip turns the lamp off. when the sun sets, pip turns the lamp on.",
    "pip counts the red stones first. one stone and one stone make two stones. two red stones and three blue stones make five stones.",
    "mira likes short and clear answers. when mira asks how many red stones there are, pip says two. when she asks how many blue stones there are, pip says three.",
    "one day pip moves one red stone into the square box. the round box now has one stone. the square box now has four stones.",
    "pip puts the red stone back. the round box has two stones again. the square box has three stones again, and the garden is quiet.",
]

_VALIDATION_PARAGRAPHS = [
    "mira sees a small robot near the garden lamp. there are two stones in the round box and three in the square box.",
    "at night the lamp is on. at sunrise the robot turns it off. mira asks a clear question about the stones.",
]


def documents() -> List[Document]:
    docs = [
        Document(f"train/p{i + 1}", "train", t) for i, t in enumerate(_TRAIN_PARAGRAPHS)
    ]
    docs += [
        Document(f"validation/p{i + 1}", "validation", t)
        for i, t in enumerate(_VALIDATION_PARAGRAPHS)
    ]
    return docs


def split_documents(split: str) -> List[Document]:
    return [d for d in documents() if d.split == split]


def corpus_sha256(split: str = "train") -> str:
    """Hash of a split, used in every run manifest."""
    h = hashlib.sha256()
    for d in split_documents(split):
        h.update(d.doc_id.encode("utf-8"))
        h.update(b"\0")
        h.update(d.text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


# --------------------------------------------------------------------------
# Instruction data, in four clearly separated families.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Instruction:
    prompt: str
    answer: str
    family: str


#: Trained during SFT.
SFT_TRAIN: List[Instruction] = [
    Instruction("how many red stones does pip have?", "two.", "train"),
    Instruction("how many blue stones does pip have?", "three.", "train"),
    Instruction("how many stones are there in all?", "five.", "train"),
    Instruction("who visits the garden?", "mira.", "train"),
    Instruction("what is pip?", "a small robot.", "train"),
    Instruction("where are the red stones?", "in the round box.", "train"),
    Instruction("where are the blue stones?", "in the square box.", "train"),
    Instruction("what color is the leaf?", "green.", "train"),
    Instruction("what does pip turn on at night?", "the lamp.", "train"),
    Instruction("what does mira like?", "short and clear answers.", "train"),
    Instruction("how many stones are one and one?", "two.", "train"),
    Instruction("how many stones are two and three?", "five.", "train"),
]

#: Same wording as training. Measures memorisation, nothing more.
SFT_EVAL_REPEAT: List[Instruction] = [
    Instruction(i.prompt, i.answer, "repeat") for i in SFT_TRAIN[:6]
]

#: Trained FACTS, untrained WORDING. Measures paraphrase robustness.
SFT_EVAL_PARAPHRASE: List[Instruction] = [
    Instruction("how many red stones are there?", "two.", "paraphrase"),
    Instruction("count the blue stones for me.", "three.", "paraphrase"),
    Instruction("how many stones in total?", "five.", "paraphrase"),
    Instruction("who comes to the garden?", "mira.", "paraphrase"),
    Instruction("tell me what pip is.", "a small robot.", "paraphrase"),
    Instruction("which box holds the red stones?", "in the round box.", "paraphrase"),
]

#: Facts present in the paragraphs but never asked during SFT.
SFT_EVAL_HELDOUT: List[Instruction] = [
    Instruction("what does pip turn off when the sun rises?", "the lamp.", "heldout"),
    Instruction("how many stones are in the square box?", "three.", "heldout"),
    Instruction("what does mira give pip?", "one green leaf.", "heldout"),
    Instruction("where does pip put the green leaf?", "beside the round box.", "heldout"),
    Instruction("when does pip turn the lamp on?", "when the sun sets.", "heldout"),
]


def instruction_families() -> Dict[str, List[Instruction]]:
    return {
        "train": SFT_TRAIN,
        "repeat": SFT_EVAL_REPEAT,
        "paraphrase": SFT_EVAL_PARAPHRASE,
        "heldout": SFT_EVAL_HELDOUT,
    }


# --------------------------------------------------------------------------
# Preference data. Chosen/rejected pairs with the reason recorded, so a
# learner can see WHY a pair was ranked and that "chosen" is not a proof of
# factual correctness.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    reason: str
    #: True when the rejected answer is also factually correct and only the
    #: style differs. These pairs teach that preference != correctness.
    rejected_is_factual: bool = False


PREFERENCE_PAIRS: List[PreferencePair] = [
    PreferencePair(
        "how many red stones does pip have?",
        "two.",
        "i will not answer that.",
        "refusal of an ordinary question",
    ),
    PreferencePair(
        "how many blue stones does pip have?",
        "three.",
        "maybe three, maybe four, it is hard to say.",
        "hedging when the answer is known",
    ),
    PreferencePair(
        "how many stones are there in all?",
        "five.",
        "five. five. five. five.",
        "repetition",
    ),
    PreferencePair(
        "who visits the garden?",
        "mira.",
        "a person visits the garden each morning and that person is called mira.",
        "wordiness; mira likes short answers",
        rejected_is_factual=True,
    ),
    PreferencePair(
        "what is pip?",
        "a small robot.",
        "pip is a large robot.",
        "factually wrong",
    ),
    PreferencePair(
        "where are the red stones?",
        "in the round box.",
        "in the square box.",
        "factually wrong",
    ),
]


# --------------------------------------------------------------------------
# Verifiable tasks. Deterministic generators, separate seeds per family, and
# a hidden answer that the verifier sees and the policy prompt never does.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiableTask:
    prompt: str
    answer: int
    family: str
    #: Everything needed to regenerate this item.
    recipe: Dict[str, object] = field(default_factory=dict)


def _lcg(seed: int):
    """Tiny deterministic generator. Written out so a learner can see that
    'random' task data is reproducible from a seed, with no hidden state."""
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0xFFFFFFFF
        yield state >> 16


def _permuted_pairs(seed: int, lo: int = 1, hi: int = 6) -> List[Tuple[int, int]]:
    """Every (a, b) pair in range, in a deterministic shuffled order.

    Enumerating the whole space and then splitting it is what makes the task
    families genuinely disjoint. Drawing each family from its own random seed
    would NOT: with a small value range the same pair reappears across seeds,
    and a "held-out" family quietly contains items the model warmed up on.
    """
    pairs = [(a, b) for a in range(lo, hi + 1) for b in range(lo, hi + 1)]
    rng = _lcg(seed)
    # Fisher-Yates, driven by the visible generator above.
    for i in range(len(pairs) - 1, 0, -1):
        j = next(rng) % (i + 1)
        pairs[i], pairs[j] = pairs[j], pairs[i]
    return pairs


#: Fraction of the two-operand value space reserved for the warm-start family.
_WARM_START_FRACTION = 0.5

#: One seed shuffles the shared value space; the split below is disjoint.
_PAIR_SEED = 7


def _two_operand_split() -> Dict[str, List[Tuple[int, int]]]:
    pairs = _permuted_pairs(_PAIR_SEED)
    cut = int(len(pairs) * _WARM_START_FRACTION)
    return {"add": pairs[:cut], "template": pairs[cut:]}


def generate_tasks(family: str, n: int, seed: int) -> List[VerifiableTask]:
    """Three families with deliberately different difficulty.

    ``add``      - the template and the VALUES a warm start may use.
    ``template`` - the same template, values disjoint from ``add``.
    ``compose``  - two operations, a structurally held-out composition.

    ``seed`` only orders the items within a family; which values belong to
    which family is fixed by :func:`_two_operand_split` so the disjointness
    cannot drift when ``n`` or ``seed`` changes.
    """
    if family == "compose":
        rng = _lcg(seed)
        out: List[VerifiableTask] = []
        seen = set()
        guard = 0
        while len(out) < n and guard < 10000:
            guard += 1
            a = next(rng) % 5 + 1
            b = next(rng) % 5 + 1
            c = next(rng) % 3 + 1
            key = (a, b, c)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                VerifiableTask(
                    prompt=f"pip has {a} red stones and {b} blue stones. "
                    f"mira gives pip {c} more red stones. how many stones in all?",
                    answer=a + b + c,
                    family=family,
                    recipe={"a": a, "b": b, "c": c, "op": "add3"},
                )
            )
        return out

    if family not in ("add", "template"):
        raise ValueError(f"unknown task family: {family!r}")

    pairs = _two_operand_split()[family]
    if n > len(pairs):
        raise ValueError(
            f"family {family!r} holds only {len(pairs)} disjoint items; asked for {n}"
        )
    # Rotate by the seed so the order is seed-dependent but the SET is not.
    offset = next(_lcg(seed)) % len(pairs)
    chosen = [pairs[(offset + i) % len(pairs)] for i in range(n)]
    return [
        VerifiableTask(
            prompt=f"what is {a} + {b}?",
            answer=a + b,
            family=family,
            recipe={"a": a, "b": b, "op": "add2"},
        )
        for a, b in chosen
    ]


#: Distinct seeds, so no item can leak between families.
TASK_SEEDS = {"add": 101, "template": 202, "compose": 303}


def task_families(n: int = 8) -> Dict[str, List[VerifiableTask]]:
    return {f: generate_tasks(f, n, s) for f, s in TASK_SEEDS.items()}


# --------------------------------------------------------------------------
# Chat template. One definition, used by SFT, preference, RL and evaluation,
# so a formatting change can never silently desynchronise two stages.
# --------------------------------------------------------------------------

PROMPT_TEMPLATE = "user: {prompt}\nassistant: "


def format_prompt(prompt: str) -> str:
    return PROMPT_TEMPLATE.format(prompt=prompt)


def all_text() -> str:
    """Every character the tokenizer must be able to represent.

    Deliberately built from the actual data rather than a hand-typed alphabet,
    so adding a document can never leave an out-of-vocabulary character
    silently mapping to <unk>.
    """
    parts: List[str] = [d.text for d in documents()]
    for fam in instruction_families().values():
        for item in fam:
            parts.append(format_prompt(item.prompt))
            parts.append(item.answer)
    for p in PREFERENCE_PAIRS:
        parts.append(format_prompt(p.prompt))
        parts.append(p.chosen)
        parts.append(p.rejected)
    for fam in task_families().values():
        for t in fam:
            parts.append(format_prompt(t.prompt))
    parts.append("answer: 0123456789")
    return "".join(parts)


def dataset_sha256() -> str:
    """One hash over EVERY dataset a run touches.

    Hashing only the pretraining paragraphs cannot distinguish two runs whose
    instruction, preference or task data differ completely - and those runs
    produce different checkpoints, different curves and different evaluations.
    A provenance token that cannot tell them apart is not provenance.
    """
    h = hashlib.sha256()
    for d in documents():
        h.update(f"doc:{d.doc_id}:{d.split}:{d.text}\0".encode("utf-8"))
    for name, items in instruction_families().items():
        for i in items:
            h.update(f"instr:{name}:{i.prompt}:{i.answer}\0".encode("utf-8"))
    for p in PREFERENCE_PAIRS:
        h.update(f"pref:{p.prompt}:{p.chosen}:{p.rejected}\0".encode("utf-8"))
    for name, items in task_families(18).items():
        for t in items:
            h.update(f"task:{name}:{t.prompt}:{t.answer}\0".encode("utf-8"))
    h.update(f"template:{PROMPT_TEMPLATE}\0".encode("utf-8"))
    return h.hexdigest()
