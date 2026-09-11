"""The run artifact: every stage present, JSON-serialisable, and an HTML
injection that cannot break out of its script element.

``</script>`` inside JSON string data terminates the element early. The training
text is learner-supplied, so this is a real injection path, and the escape has to
survive a round trip through ``JSON.parse`` - modelled here by ``json.loads``,
which decodes ``\\u003c`` the same way a browser does.

``ExportEndToEnd`` is the slowest test in the suite (it trains every stage once,
about 13 seconds). ``GLASSBOX_SKIP_SLOW=1`` skips it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from glassbox.export import (
    DATA_END,
    DATA_START,
    RunConfig,
    build_run,
    inject_into_html,
    write_run,
)
from glassbox.model import ModelConfig
from glassbox.preference import DPOConfig, RewardConfig
from glassbox.rlvr import GRPOConfig
from glassbox.train import PretrainConfig, SFTConfig

from . import fixtures

EXPECTED_STAGES = (
    "pretrain",
    "sft",
    "lora_sft",
    "reward_model",
    "dpo",
    "warm_start",
    "grpo",
    "evaluation",
    "tokenizers",
    "verifier_table",
)

EXPECTED_CHECKPOINTS = (
    "initial",
    "base",
    "sft",
    "lora",
    "dpo",
    "warm_start",
    "rl",
)


def tiny_run_config() -> RunConfig:
    """The smallest configuration that still exercises every code path."""
    cfg = RunConfig(
        seed=fixtures.SEED,
        model=ModelConfig(vocab_size=len(fixtures.tokenizer())),
        pretrain=PretrainConfig(steps=3, eval_every=3, warmup_steps=1),
        sft=SFTConfig(steps=3, eval_every=3, warmup_steps=1),
    )
    cfg.reward = RewardConfig(steps=2)
    cfg.dpo = DPOConfig(steps=2)
    cfg.warm_start = SFTConfig(steps=3, eval_every=3, warmup_steps=1)
    cfg.grpo = GRPOConfig(
        iterations=1,
        group_size=2,
        inner_epochs=1,
        max_new_tokens=4,
        temperature=1.0,
        top_k=6,
    )
    return cfg


@unittest.skipIf(fixtures.SKIP_SLOW, fixtures.SKIP_SLOW_REASON)
class ExportEndToEnd(unittest.TestCase):
    """One whole pipeline at tiny step counts."""

    @classmethod
    def setUpClass(cls) -> None:
        # Named 'artifact', not 'run': unittest.TestCase.run IS the method the
        # runner calls, and assigning a dict over it breaks the whole suite.
        cls.artifact = build_run(tiny_run_config(), verbose=False)

    def test_the_top_level_shape_is_what_the_interface_expects(self) -> None:
        self.assertEqual(
            sorted(self.artifact),
            [
                "checkpoint_order",
                "checkpoints",
                "checks",
                "data",
                "manifest",
                "stages",
            ],
        )

    def test_every_stage_key_is_present(self) -> None:
        for stage in EXPECTED_STAGES:
            with self.subTest(stage=stage):
                self.assertIn(stage, self.artifact["stages"])
                self.assertTrue(self.artifact["stages"][stage])
        self.assertEqual(sorted(self.artifact["stages"]), sorted(EXPECTED_STAGES))

    def test_every_checkpoint_is_present_and_ordered(self) -> None:
        self.assertEqual(
            sorted(self.artifact["checkpoints"]), sorted(EXPECTED_CHECKPOINTS)
        )
        self.assertEqual(tuple(self.artifact["checkpoint_order"]), EXPECTED_CHECKPOINTS)

    def test_checkpoint_lineage_is_recorded(self) -> None:
        parents = {
            label: body["parent"]
            for label, body in self.artifact["checkpoints"].items()
        }
        self.assertEqual(
            parents,
            {
                "initial": None,
                "base": "initial",
                "sft": "base",
                "lora": "base",
                "dpo": "sft",
                "warm_start": "sft",
                "rl": "warm_start",
            },
        )

    def test_the_whole_run_is_json_serialisable_and_round_trips(self) -> None:
        text = json.dumps(self.artifact)
        self.assertEqual(
            json.loads(text)["manifest"]["run_id"],
            self.artifact["manifest"]["run_id"],
        )

    def test_the_run_contains_no_value_that_json_parse_would_reject(self) -> None:
        """NaN, Infinity and -Infinity are not JSON.

        Python's json module emits them as bare literals and accepts them back,
        but ``JSON.parse`` in the browser throws - so a single non-finite float
        anywhere in the artifact makes the page fail to load. ``allow_nan=False``
        is the precise test; searching the serialised text for the substring
        "NaN" is not, because a check description legitimately contains the word.
        """
        try:
            text = json.dumps(self.artifact, allow_nan=False)
        except ValueError as exc:  # pragma: no cover - only on a real defect
            self.fail(
                "the run artifact contains a non-finite float, which JSON.parse "
                "cannot read: {}".format(exc)
            )

        def reject(constant):
            raise AssertionError(
                "JSON.parse would reject the literal {!r}".format(constant)
            )

        self.assertEqual(
            json.loads(text, parse_constant=reject)["manifest"]["run_id"],
            self.artifact["manifest"]["run_id"],
        )

    def test_the_manifest_carries_provenance(self) -> None:
        manifest = self.artifact["manifest"]
        for key in (
            "schema_version",
            "seed",
            "torch_version",
            "python_version",
            "dataset_sha256",
            "corpus_sha256",
            "tokenizer_sha256",
            "parameters",
            "run_id",
        ):
            with self.subTest(key=key):
                self.assertIn(key, manifest)
                self.assertIsNotNone(manifest[key])
        self.assertEqual(manifest["parameters"], 3048)
        self.assertEqual(len(manifest["dataset_sha256"]), 64)
        self.assertTrue(manifest["run_id"].startswith("run_"))
        self.assertTrue(manifest["honest_limits"])

    def test_the_checks_block_reports_every_check(self) -> None:
        checks = self.artifact["checks"]
        self.assertEqual(checks["total"], 26)
        failed = [r["name"] for r in checks["results"] if not r["passed"]]
        self.assertEqual(failed, [], "failing checks in the exported run: {}".format(failed))
        self.assertTrue(checks["all_passed"])

    def test_each_checkpoint_records_its_own_gradient_step(self) -> None:
        """The old engine aliased ONE recording into every checkpoint while the
        interface let the learner switch between them.

        'lora' legitimately shares 'base''s displayed matrix - LoRA freezes
        blocks.0.mlp_up, so it cannot have moved - which is why this asserts on
        the checkpoints that DID train that matrix rather than on all of them.
        """
        steps = {
            label: self.artifact["checkpoints"][label]["gradient_step"]
            for label in EXPECTED_CHECKPOINTS
        }
        for label, step in steps.items():
            with self.subTest(checkpoint=label):
                self.assertEqual(step["prompt"], "pip has ")
                self.assertIn("finite_difference", step)
        matrices = {
            label: json.dumps(step["before"])
            for label, step in steps.items()
            if label != "lora"
        }
        self.assertEqual(
            len(set(matrices.values())),
            len(matrices),
            "two checkpoints that each trained blocks.0.mlp_up reported "
            "identical 'before' matrices, so a recording is being shared",
        )
        self.assertEqual(
            matrices["base"],
            json.dumps(steps["lora"]["before"]),
            "the LoRA checkpoint's frozen matrix should still equal the base "
            "checkpoint's",
        )

    def test_the_lora_checkpoint_reports_its_frozen_matrix_honestly(self) -> None:
        lora = self.artifact["checkpoints"]["lora"]["gradient_step"]
        self.assertFalse(lora["matrix_is_trainable"])
        self.assertIsNotNone(lora["frozen_note"])
        self.assertIsNone(lora["gradient"])
        self.assertFalse(lora["display_matrix_changed"])
        self.assertGreater(lora["frozen_parameter_elements"], 0)
        self.assertFalse(lora["all_parameters_updated"])
        for label in ("initial", "base", "sft", "dpo", "warm_start", "rl"):
            with self.subTest(checkpoint=label):
                step = self.artifact["checkpoints"][label]["gradient_step"]
                self.assertTrue(step["matrix_is_trainable"])
                self.assertIsNone(step["frozen_note"])
                self.assertTrue(step["all_parameters_updated"])

    def test_the_evaluation_reports_every_checkpoint_with_a_pinned_protocol(self) -> None:
        reports = self.artifact["stages"]["evaluation"]["reports"]
        self.assertEqual(
            [r["checkpoint"] for r in reports], list(EXPECTED_CHECKPOINTS)
        )
        for report in reports:
            with self.subTest(checkpoint=report["checkpoint"]):
                self.assertTrue(report["protocol"])
                self.assertEqual(
                    sorted(report["instruction_exact_match"]),
                    ["heldout", "paraphrase", "repeat", "train"],
                )
                self.assertEqual(
                    sorted(report["task_pass_1"]), ["add", "compose", "template"]
                )

    def test_the_baselines_are_exported_next_to_the_model_losses(self) -> None:
        baselines = self.artifact["stages"]["evaluation"]["baselines"]
        self.assertIn("bigram_laplace", baselines["validation"])

    def test_write_run_produces_a_named_file_and_a_latest_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            path = write_run(self.artifact, out)
            self.assertTrue(path.exists())
            self.assertEqual(path.name, self.artifact["manifest"]["run_id"] + ".json")
            latest = out / "latest.json"
            self.assertTrue(latest.exists())
            self.assertEqual(
                json.loads(latest.read_text())["manifest"]["run_id"],
                self.artifact["manifest"]["run_id"],
            )


class HtmlInjection(unittest.TestCase):
    """Escaping, checked against a payload that really does contain ``</script>``."""

    HOSTILE = 'a </script><script>alert("x")</script> b'

    def _inject(self, run):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "index.html"
        path.write_text(
            "<html><head><title>t</title></head><body>BEFORE"
            + DATA_START
            + '{"placeholder":true}'
            + DATA_END
            + "AFTER</body></html>"
        )
        inject_into_html(run, path)
        html = path.read_text()
        start = html.index(DATA_START) + len(DATA_START)
        end = html.index(DATA_END, start)
        return html, html[start:end]

    def test_a_hostile_paragraph_cannot_terminate_the_data_block_early(self) -> None:
        run = {"data": {"paragraphs": [self.HOSTILE, "plain"]}, "n": 1}
        html, payload = self._inject(run)
        self.assertNotIn(
            "</script>",
            payload,
            "the data block contains a literal </script>, which terminates the "
            "element early",
        )
        self.assertNotIn("<", payload, "an unescaped '<' remains in the payload")
        # Exactly one </script> in the whole file: the real terminator.
        self.assertEqual(html.count("</script>"), 1)

    def test_the_payload_round_trips_through_a_json_parse_equivalent(self) -> None:
        run = {"data": {"paragraphs": [self.HOSTILE, "plain"]}, "n": 1}
        html, payload = self._inject(run)
        # json.loads decodes < exactly as JSON.parse does.
        self.assertEqual(json.loads(payload), run)
        self.assertEqual(
            json.loads(payload)["data"]["paragraphs"][0],
            self.HOSTILE,
            "the original characters did not survive the escape",
        )

    def test_the_surrounding_html_is_preserved(self) -> None:
        html, _ = self._inject({"n": 1})
        self.assertIn("BEFORE" + DATA_START, html)
        self.assertIn(DATA_END + "AFTER", html)
        self.assertIn("<title>t</title>", html)

    def test_injecting_twice_replaces_rather_than_appends(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "index.html"
        path.write_text(
            "<html><body>" + DATA_START + "{}" + DATA_END + "</body></html>"
        )
        inject_into_html({"first": True}, path)
        inject_into_html({"second": True}, path)
        html = path.read_text()
        self.assertEqual(html.count(DATA_START), 1)
        self.assertEqual(html.count(DATA_END), 1)
        start = html.index(DATA_START) + len(DATA_START)
        end = html.index(DATA_END, start)
        self.assertEqual(json.loads(html[start:end]), {"second": True})

    def test_a_file_with_no_data_block_is_rejected(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "index.html"
        path.write_text("<html><body>nothing here</body></html>")
        with self.assertRaises(ValueError):
            inject_into_html({"n": 1}, path)
        self.assertEqual(path.read_text(), "<html><body>nothing here</body></html>")

    def test_every_less_than_sign_is_escaped_not_only_the_closing_tag(self) -> None:
        run = {"text": "1 < 2 and <b>bold</b>"}
        _, payload = self._inject(run)
        self.assertNotIn("<", payload)
        self.assertEqual(json.loads(payload), run)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
