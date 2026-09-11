"""Decoding controls must actually control something.

The previous engine divided the logits by ``temperature`` and then took
``argmax``. Argmax is invariant under positive scaling, so the control did
nothing - and a test that sampled two strings and checked they differed would
have passed anyway, because the two strings were identical for a different
reason. ``test_argmax_is_invariant_to_temperature`` below states that invariance
explicitly, so the reason the bug hid is itself part of the suite, and every
other temperature test measures the DISTRIBUTION instead.

The old decoder also sliced the prompt to its last 32 tokens. The chat-formatted
prompt ``"user: how many red stones does pip have?\\nassistant: "`` is 53 tokens,
so the model answered from the middle of the question - that single bug was why
the headline SFT answer looked broken.
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from glassbox import data
from glassbox.generate import (
    GREEDY,
    DecodeConfig,
    _filter_logits,
    generate,
    sequence_logprob,
)

from . import fixtures


def _next_logits(model, tok, prompt):
    ids = torch.tensor([tok.encode(prompt, bos=True)])
    with torch.no_grad():
        return model(ids)["logits"][0, -1]


def _entropy(logits: torch.Tensor, temperature: float) -> float:
    p = F.softmax(logits / temperature, dim=-1)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum())


class Temperature(unittest.TestCase):
    TEMPERATURES = (0.25, 0.5, 1.0, 2.0, 4.0)

    def setUp(self) -> None:
        self.tok = fixtures.tokenizer()
        self.model = fixtures.trained_model()
        self.logits = _next_logits(self.model, self.tok, "pip has ")

    def test_entropy_increases_strictly_with_temperature(self) -> None:
        entropies = [_entropy(self.logits, t) for t in self.TEMPERATURES]
        for (t_low, h_low), (t_high, h_high) in zip(
            zip(self.TEMPERATURES, entropies), zip(self.TEMPERATURES[1:], entropies[1:])
        ):
            with self.subTest(low=t_low, high=t_high):
                self.assertLess(
                    h_low,
                    h_high,
                    "entropy did not rise from T={} ({:.4f}) to T={} "
                    "({:.4f})".format(t_low, h_low, t_high, h_high),
                )

    def test_entropy_is_bounded_by_ln_vocab(self) -> None:
        for t in self.TEMPERATURES:
            with self.subTest(temperature=t):
                self.assertLessEqual(
                    _entropy(self.logits, t), math.log(len(self.tok)) + 1e-6
                )

    def test_argmax_is_invariant_to_temperature(self) -> None:
        """This is WHY the old bug hid, stated as a property.

        Any test that only compared argmax outputs - or two greedy strings -
        could not have detected that temperature was being applied before an
        argmax.
        """
        top = int(torch.argmax(self.logits))
        for t in self.TEMPERATURES:
            with self.subTest(temperature=t):
                self.assertEqual(int(torch.argmax(self.logits / t)), top)

    def test_greedy_is_reported_as_greedy(self) -> None:
        self.assertTrue(DecodeConfig(temperature=0.0).is_greedy)
        self.assertTrue(DecodeConfig(temperature=-1.0).is_greedy)
        self.assertFalse(DecodeConfig(temperature=0.7).is_greedy)
        self.assertIn("greedy", GREEDY.describe())
        self.assertIn("temperature 0.7", DecodeConfig(temperature=0.7).describe())

    def test_greedy_decoding_is_deterministic(self) -> None:
        a = generate(self.model, self.tok, "pip has ", DecodeConfig(max_new_tokens=12))
        b = generate(self.model, self.tok, "pip has ", DecodeConfig(max_new_tokens=12))
        self.assertEqual(a.new_ids, b.new_ids)

    def test_a_seeded_sample_is_reproducible_and_an_unequal_seed_is_not_asserted(
        self,
    ) -> None:
        cfg = DecodeConfig(max_new_tokens=12, temperature=1.0, seed=11)
        a = generate(self.model, self.tok, "pip has ", cfg)
        b = generate(self.model, self.tok, "pip has ", cfg)
        self.assertEqual(a.new_ids, b.new_ids)


class TopKAndTopP(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = fixtures.tokenizer()
        self.model = fixtures.trained_model()
        self.logits = _next_logits(self.model, self.tok, "pip has ")
        self.order = [int(i) for i in torch.argsort(self.logits, descending=True)]

    def test_top_k_leaves_exactly_k_candidates(self) -> None:
        for k in (1, 2, 3, 5, 10, 44):
            with self.subTest(k=k):
                filtered = _filter_logits(self.logits, k, 0.0)
                kept = int(torch.isfinite(filtered).sum())
                self.assertEqual(
                    kept, k, "top-{} left {} candidates".format(k, kept)
                )

    def test_top_k_keeps_the_k_highest_scoring_tokens(self) -> None:
        for k in (1, 3, 10):
            with self.subTest(k=k):
                filtered = _filter_logits(self.logits, k, 0.0)
                kept = {
                    int(i) for i in torch.nonzero(torch.isfinite(filtered)).flatten()
                }
                self.assertEqual(kept, set(self.order[:k]))
                for i in kept:
                    self.assertEqual(float(filtered[i]), float(self.logits[i]))

    def test_top_k_at_or_above_vocab_size_is_a_no_op(self) -> None:
        for k in (len(self.tok), len(self.tok) + 5):
            with self.subTest(k=k):
                filtered = _filter_logits(self.logits, k, 0.0)
                self.assertTrue(torch.equal(filtered, self.logits))

    def test_top_k_reported_by_the_decoder_matches_the_filter(self) -> None:
        for k in (1, 3, 10):
            with self.subTest(k=k):
                g = generate(
                    self.model,
                    self.tok,
                    "pip has ",
                    DecodeConfig(max_new_tokens=1, temperature=1.0, top_k=k, seed=2),
                    record_steps=True,
                )
                self.assertEqual(int(g.steps[0]["candidates_kept"]), k)

    def test_top_p_keeps_a_prefix_of_the_sorted_distribution(self) -> None:
        for p in (0.1, 0.3, 0.5, 0.9, 0.99):
            with self.subTest(top_p=p):
                filtered = _filter_logits(self.logits, 0, p)
                kept = [
                    int(i) for i in torch.nonzero(torch.isfinite(filtered)).flatten()
                ]
                self.assertTrue(kept, "top-p left no candidates")
                self.assertEqual(
                    sorted(kept),
                    sorted(self.order[: len(kept)]),
                    "top-p kept a non-prefix set: {}".format(kept),
                )

    def test_top_p_is_never_empty_and_always_keeps_the_mode(self) -> None:
        for p in (1e-9, 0.01, 0.5):
            with self.subTest(top_p=p):
                filtered = _filter_logits(self.logits, 0, p)
                self.assertGreaterEqual(int(torch.isfinite(filtered).sum()), 1)
                self.assertTrue(math.isfinite(float(filtered[self.order[0]])))

    def test_a_smaller_top_p_never_keeps_more_tokens(self) -> None:
        sizes = [
            int(torch.isfinite(_filter_logits(self.logits, 0, p)).sum())
            for p in (0.1, 0.3, 0.5, 0.9, 0.99)
        ]
        self.assertEqual(sizes, sorted(sizes))

    def test_top_p_mass_reaches_the_threshold(self) -> None:
        probs = F.softmax(self.logits, dim=-1)
        for p in (0.3, 0.5, 0.9):
            with self.subTest(top_p=p):
                filtered = _filter_logits(self.logits, 0, p)
                kept = torch.isfinite(filtered)
                self.assertGreaterEqual(float(probs[kept].sum()), p)


class EndOfSequence(unittest.TestCase):
    """EOS must end decoding.

    The forced-token stub pins this to the decode loop so it cannot start
    passing or failing because training changed, and the real-model case
    confirms the same path fires on learned weights.
    """

    def setUp(self) -> None:
        self.tok = fixtures.tokenizer()
        self.cfg = fixtures.model_config()

    def test_eos_stops_decoding_immediately(self) -> None:
        stub = fixtures.ForcedTokenModel(
            len(self.tok), self.cfg.context, self.tok.eos_id
        )
        g = generate(stub, self.tok, "pip has ", DecodeConfig(max_new_tokens=10))
        self.assertEqual(g.new_ids, [self.tok.eos_id])
        self.assertTrue(g.stopped_on_eos)
        self.assertFalse(g.hit_length_limit)
        self.assertEqual(g.text, "", "EOS should not appear in decoded text")

    def test_stop_at_eos_false_keeps_going(self) -> None:
        stub = fixtures.ForcedTokenModel(
            len(self.tok), self.cfg.context, self.tok.eos_id
        )
        g = generate(
            stub,
            self.tok,
            "pip has ",
            DecodeConfig(max_new_tokens=10, stop_at_eos=False),
        )
        self.assertEqual(len(g.new_ids), 10)
        self.assertEqual(set(g.new_ids), {self.tok.eos_id})
        self.assertFalse(g.stopped_on_eos)
        self.assertTrue(g.hit_length_limit)

    def test_length_limit_is_reported_when_eos_never_comes(self) -> None:
        ordinary = self.tok.stoi["a"]
        stub = fixtures.ForcedTokenModel(len(self.tok), self.cfg.context, ordinary)
        g = generate(stub, self.tok, "pip has ", DecodeConfig(max_new_tokens=7))
        self.assertEqual(g.new_ids, [ordinary] * 7)
        self.assertFalse(g.stopped_on_eos)
        self.assertTrue(g.hit_length_limit)
        self.assertEqual(g.text, "aaaaaaa")

    def test_the_real_model_stops_on_a_learned_eos(self) -> None:
        model = fixtures.trained_model()
        paragraph = data.split_documents("train")[0].text
        g = generate(model, self.tok, paragraph, DecodeConfig(max_new_tokens=12))
        self.assertTrue(
            g.stopped_on_eos,
            "the model did not emit EOS at the end of a document it was "
            "trained on; it generated {!r}".format(g.text),
        )
        self.assertEqual(g.new_ids[-1], self.tok.eos_id)
        self.assertNotIn(self.tok.eos_id, g.new_ids[:-1])


class PromptIsNotTruncated(unittest.TestCase):
    """A prompt longer than 32 tokens must be used in full.

    The old decoder sliced ``ids[-32:]`` before the forward pass.
    """

    LONG_PROSE = (
        "pip is a small robot in a quiet garden. pip has two red stones and three "
    )

    def setUp(self) -> None:
        self.tok = fixtures.tokenizer()
        self.model = fixtures.trained_model()
        self.cfg = fixtures.model_config()

    def test_prompt_ids_are_the_whole_encoded_prompt(self) -> None:
        for prompt in (
            self.LONG_PROSE,
            data.format_prompt("how many red stones does pip have?"),
            data.format_prompt("what does pip turn off when the sun rises?"),
        ):
            expected = self.tok.encode(prompt, bos=True)
            with self.subTest(prompt=prompt[:40], length=len(expected)):
                self.assertGreater(
                    len(expected), 32, "this prompt is too short to be a test"
                )
                g = generate(
                    self.model, self.tok, prompt, DecodeConfig(max_new_tokens=1)
                )
                self.assertEqual(len(g.prompt_ids), len(expected))
                self.assertEqual(g.prompt_ids, expected)

    def test_the_prefill_pass_sees_every_prompt_token(self) -> None:
        """Asserted against the stub, so it holds whatever the weights are."""
        stub = fixtures.ForcedTokenModel(
            len(self.tok), self.cfg.context, self.tok.stoi["a"]
        )
        prompt = self.LONG_PROSE
        expected = self.tok.encode(prompt, bos=True)
        generate(stub, self.tok, prompt, DecodeConfig(max_new_tokens=3))
        self.assertEqual(
            stub.calls[0],
            expected,
            "the prefill pass received {} tokens, not the full {}".format(
                len(stub.calls[0]), len(expected)
            ),
        )
        self.assertEqual(len(stub.calls[0]), len(expected))

    def test_the_uncached_path_also_sees_every_prompt_token(self) -> None:
        stub = fixtures.ForcedTokenModel(
            len(self.tok), self.cfg.context, self.tok.stoi["a"]
        )
        prompt = self.LONG_PROSE
        expected = self.tok.encode(prompt, bos=True)
        generate(
            stub,
            self.tok,
            prompt,
            DecodeConfig(max_new_tokens=3, use_cache=False),
        )
        self.assertEqual(stub.calls[0], expected)
        # Without the cache the whole prefix is re-fed every step, and the full
        # prompt is always at the front of it: one prefill plus one recompute per
        # generated token, each one token longer than the last.
        n = len(expected)
        self.assertEqual(
            [len(call) for call in stub.calls], [n, n + 1, n + 2, n + 3]
        )
        for call in stub.calls:
            self.assertEqual(call[:n], expected)

    def test_the_first_token_comes_from_the_full_context(self) -> None:
        prompt = self.LONG_PROSE
        full = self.tok.encode(prompt, bos=True)
        with torch.no_grad():
            from_full = int(
                torch.argmax(self.model(torch.tensor([full]))["logits"][0, -1])
            )
            from_last_32 = int(
                torch.argmax(
                    self.model(torch.tensor([full[-32:]]))["logits"][0, -1]
                )
            )
        g = generate(self.model, self.tok, prompt, DecodeConfig(max_new_tokens=1))
        self.assertEqual(g.new_ids[0], from_full)
        # The discriminator: on this fixture the two disagree, so a decoder that
        # truncated to 32 tokens would fail the assertion above instead of
        # passing by coincidence.
        self.assertNotEqual(
            from_full,
            from_last_32,
            "full-context and 32-token-truncated decoding agree on this "
            "prompt, so this test cannot detect truncation",
        )

    def test_cached_and_uncached_decoding_agree_on_a_long_prompt(self) -> None:
        cached = generate(
            self.model,
            self.tok,
            self.LONG_PROSE,
            DecodeConfig(max_new_tokens=10, use_cache=True),
        )
        uncached = generate(
            self.model,
            self.tok,
            self.LONG_PROSE,
            DecodeConfig(max_new_tokens=10, use_cache=False),
        )
        self.assertEqual(cached.new_ids, uncached.new_ids)


class SequenceLogprob(unittest.TestCase):
    """The scoring helper DPO and GRPO depend on has the same shift to get right."""

    def test_only_the_response_tokens_are_scored(self) -> None:
        tok = fixtures.tokenizer()
        model = fixtures.trained_model()
        prompt = data.format_prompt("how many red stones does pip have?")
        out = sequence_logprob(model, tok, prompt, "two.")
        expected = tok.encode("two.") + [tok.eos_id]
        self.assertEqual(out["num_scored_tokens"], len(expected))
        self.assertEqual([p["id"] for p in out["per_token"]], expected)

    def test_the_first_scored_logprob_comes_from_the_last_prompt_position(self) -> None:
        tok = fixtures.tokenizer()
        model = fixtures.trained_model()
        prompt = data.format_prompt("who visits the garden?")
        response = "mira."
        out = sequence_logprob(model, tok, prompt, response)
        prompt_ids = tok.encode(prompt, bos=True)
        response_ids = tok.encode(response) + [tok.eos_id]
        ids = torch.tensor([prompt_ids + response_ids])
        with torch.no_grad():
            logprobs = F.log_softmax(model(ids)["logits"], dim=-1)
        expected_first = float(
            logprobs[0, len(prompt_ids) - 1, response_ids[0]]
        )
        self.assertAlmostEqual(
            out["per_token"][0]["logprob"], expected_first, places=6
        )

    def test_total_is_the_sum_and_is_negative(self) -> None:
        tok = fixtures.tokenizer()
        model = fixtures.trained_model()
        out = sequence_logprob(
            model, tok, data.format_prompt("what is pip?"), "a small robot."
        )
        self.assertAlmostEqual(
            out["total_logprob"],
            sum(p["logprob"] for p in out["per_token"]),
            places=4,
        )
        self.assertLess(out["total_logprob"], 0.0)
        self.assertAlmostEqual(
            out["mean_logprob"],
            out["total_logprob"] / out["num_scored_tokens"],
            places=6,
        )


class InferenceDoesNotTrain(unittest.TestCase):
    def test_weights_are_unchanged_by_generation(self) -> None:
        tok = fixtures.tokenizer()
        model = fixtures.trained_model()
        before = fixtures.weight_hash(model)
        generate(model, tok, "pip has ", DecodeConfig(max_new_tokens=20))
        generate(
            model,
            tok,
            "mira ",
            DecodeConfig(max_new_tokens=20, temperature=1.0, top_k=5, seed=1),
        )
        self.assertEqual(before, fixtures.weight_hash(model))

    def test_generation_restores_the_training_flag(self) -> None:
        tok = fixtures.tokenizer()
        model = fixtures.trained_model()
        model.train()
        generate(model, tok, "pip has ", DecodeConfig(max_new_tokens=2))
        self.assertTrue(model.training, "generate() left the model in eval mode")
        model.eval()
        generate(model, tok, "pip has ", DecodeConfig(max_new_tokens=2))
        self.assertFalse(model.training)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
