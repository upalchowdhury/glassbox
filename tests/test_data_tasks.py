"""Data provenance and the disjointness of the verifiable task families.

"Held out" has to be a fact about the data, not a label on it. The 'add' family
is what the RL policy warm-starts on and 'template' is what GRPO is scored on,
so a single shared item would turn the headline RL result into memorisation.

Drawing each family from its own random seed would NOT make them disjoint: with
a 1-6 value range the same (a, b) pair reappears across seeds. The families
partition one enumerated value space instead, which is what these tests check at
every size the course uses.
"""

from __future__ import annotations

import unittest

from glassbox import data


class TaskFamiliesAreDisjoint(unittest.TestCase):
    SIZES = (4, 8, 18)

    def test_add_and_template_share_no_item_at_any_size(self) -> None:
        for n in self.SIZES:
            families = data.task_families(n)
            add = {t.prompt for t in families["add"]}
            template = {t.prompt for t in families["template"]}
            with self.subTest(size=n):
                self.assertEqual(len(add), n)
                self.assertEqual(len(template), n)
                overlap = sorted(add & template)
                self.assertEqual(
                    overlap,
                    [],
                    "'add' and 'template' share {} item(s) at n={}: {}".format(
                        len(overlap), n, overlap
                    ),
                )

    def test_the_recipes_are_disjoint_too_not_just_the_prompt_strings(self) -> None:
        for n in self.SIZES:
            families = data.task_families(n)
            add = {(t.recipe["a"], t.recipe["b"]) for t in families["add"]}
            template = {(t.recipe["a"], t.recipe["b"]) for t in families["template"]}
            with self.subTest(size=n):
                self.assertEqual(add & template, set())

    def test_a_larger_size_is_a_superset_of_a_smaller_one(self) -> None:
        """The SET a family draws from must not drift when n changes."""
        for family in ("add", "template"):
            small = {t.prompt for t in data.generate_tasks(family, 4, 101)}
            large = {t.prompt for t in data.generate_tasks(family, 18, 101)}
            with self.subTest(family=family):
                self.assertTrue(small <= large)

    def test_the_value_space_is_fully_partitioned_at_full_size(self) -> None:
        families = data.task_families(18)
        add = {(t.recipe["a"], t.recipe["b"]) for t in families["add"]}
        template = {(t.recipe["a"], t.recipe["b"]) for t in families["template"]}
        whole = {(a, b) for a in range(1, 7) for b in range(1, 7)}
        self.assertEqual(add | template, whole)
        self.assertEqual(len(add) + len(template), len(whole))

    def test_asking_for_more_items_than_exist_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            data.generate_tasks("add", 19, 101)

    def test_an_unknown_family_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            data.generate_tasks("nonsense", 2, 1)

    def test_the_family_split_does_not_depend_on_the_seed(self) -> None:
        a = {t.prompt for t in data.generate_tasks("add", 18, 101)}
        b = {t.prompt for t in data.generate_tasks("add", 18, 999)}
        self.assertEqual(a, b, "changing the seed changed WHICH values are in 'add'")

    def test_the_seed_orders_the_items(self) -> None:
        a = [t.prompt for t in data.generate_tasks("add", 18, 101)]
        b = [t.prompt for t in data.generate_tasks("add", 18, 999)]
        self.assertNotEqual(a, b, "the seed has no effect on item order")
        self.assertEqual(sorted(a), sorted(b))

    def test_compose_is_structurally_different_not_just_different_values(self) -> None:
        compose = data.task_families(8)["compose"]
        for task in compose:
            with self.subTest(prompt=task.prompt):
                self.assertEqual(task.recipe["op"], "add3")
                self.assertNotIn("what is", task.prompt)
                self.assertEqual(
                    task.answer,
                    task.recipe["a"] + task.recipe["b"] + task.recipe["c"],
                )
        two_operand = {
            t.prompt for f in ("add", "template") for t in data.task_families(18)[f]
        }
        self.assertEqual({t.prompt for t in compose} & two_operand, set())

    def test_every_answer_is_correct(self) -> None:
        for name, tasks in data.task_families(18).items():
            for task in tasks:
                with self.subTest(family=name, prompt=task.prompt):
                    if task.recipe["op"] == "add2":
                        self.assertEqual(
                            task.answer, task.recipe["a"] + task.recipe["b"]
                        )
                    else:
                        self.assertEqual(
                            task.answer,
                            task.recipe["a"] + task.recipe["b"] + task.recipe["c"],
                        )

    def test_the_hidden_answer_is_never_in_the_prompt_the_policy_sees(self) -> None:
        """If the answer leaks into the prompt the reward is a lookup, not a check."""
        for name, tasks in data.task_families(18).items():
            for task in tasks:
                with self.subTest(family=name, prompt=task.prompt):
                    self.assertNotIn("answer", task.prompt.lower())
                    if name == "add" or name == "template":
                        # '2 + 3' must not already contain '5' as the result.
                        self.assertNotIn(
                            "= {}".format(task.answer), task.prompt
                        )

    def test_generation_is_reproducible(self) -> None:
        for family, seed in data.TASK_SEEDS.items():
            with self.subTest(family=family):
                a = data.generate_tasks(family, 8, seed)
                b = data.generate_tasks(family, 8, seed)
                self.assertEqual(
                    [(t.prompt, t.answer) for t in a],
                    [(t.prompt, t.answer) for t in b],
                )

    def test_families_use_distinct_seeds(self) -> None:
        seeds = list(data.TASK_SEEDS.values())
        self.assertEqual(len(set(seeds)), len(seeds))


class DocumentProvenance(unittest.TestCase):
    def test_splits_are_documents_and_are_disjoint(self) -> None:
        train = data.split_documents("train")
        validation = data.split_documents("validation")
        self.assertEqual(len(train), 8)
        self.assertEqual(len(validation), 2)
        self.assertEqual(
            {d.text for d in train} & {d.text for d in validation}, set()
        )
        self.assertEqual(len(data.documents()), 10)

    def test_every_document_has_a_unique_id_and_a_hash(self) -> None:
        ids = [d.doc_id for d in data.documents()]
        self.assertEqual(len(set(ids)), len(ids))
        for d in data.documents():
            with self.subTest(doc=d.doc_id):
                self.assertEqual(len(d.sha256), 64)
                self.assertTrue(d.doc_id.startswith(d.split + "/"))

    def test_corpus_hashes_differ_between_splits(self) -> None:
        self.assertNotEqual(
            data.corpus_sha256("train"), data.corpus_sha256("validation")
        )

    def test_the_dataset_hash_covers_more_than_the_paragraphs(self) -> None:
        """A provenance token that cannot tell two runs apart is not provenance."""
        self.assertNotEqual(data.dataset_sha256(), data.corpus_sha256("train"))
        self.assertEqual(len(data.dataset_sha256()), 64)

    def test_the_dataset_hash_is_stable_across_calls(self) -> None:
        self.assertEqual(data.dataset_sha256(), data.dataset_sha256())

    def test_instruction_families_are_separated_as_documented(self) -> None:
        families = data.instruction_families()
        self.assertEqual(
            sorted(families), ["heldout", "paraphrase", "repeat", "train"]
        )
        trained_prompts = {i.prompt for i in families["train"]}
        self.assertTrue(
            {i.prompt for i in families["repeat"]} <= trained_prompts,
            "the 'repeat' family must reuse trained wording exactly",
        )
        self.assertEqual(
            {i.prompt for i in families["paraphrase"]} & trained_prompts,
            set(),
            "a 'paraphrase' prompt is word-for-word a training prompt",
        )
        self.assertEqual(
            {i.prompt for i in families["heldout"]} & trained_prompts, set()
        )

    def test_paraphrase_answers_are_trained_facts(self) -> None:
        trained_answers = {i.answer for i in data.SFT_TRAIN}
        for item in data.SFT_EVAL_PARAPHRASE:
            with self.subTest(prompt=item.prompt):
                self.assertIn(item.answer, trained_answers)

    def test_heldout_facts_appear_in_the_paragraphs(self) -> None:
        corpus = " ".join(d.text for d in data.split_documents("train"))
        for item in data.SFT_EVAL_HELDOUT:
            with self.subTest(prompt=item.prompt):
                stem = item.answer.rstrip(".")
                self.assertIn(
                    stem,
                    corpus,
                    "the held-out answer {!r} is not derivable from the "
                    "training paragraphs".format(item.answer),
                )

    def test_the_chat_template_has_one_definition(self) -> None:
        self.assertIn("{prompt}", data.PROMPT_TEMPLATE)
        self.assertEqual(
            data.format_prompt("x"), data.PROMPT_TEMPLATE.format(prompt="x")
        )
        self.assertTrue(data.format_prompt("x").endswith("assistant: "))

    def test_preference_pairs_are_well_formed(self) -> None:
        self.assertEqual(len(data.PREFERENCE_PAIRS), 6)
        for pair in data.PREFERENCE_PAIRS:
            with self.subTest(prompt=pair.prompt):
                self.assertNotEqual(pair.chosen, pair.rejected)
                self.assertTrue(pair.reason.strip())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
