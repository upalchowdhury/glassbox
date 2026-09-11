"""Regression tests for server.py.

The headline test is :class:`ScriptBreakoutTests`, which reproduces the stored
XSS the previous server had and proves it is closed. ``json.dumps`` does not
escape ``<``, so a learner-supplied paragraph containing
``</script><script>alert(1)</script>`` used to terminate the run-data element
early and run. The fix is to escape ``<`` as ``\\u003c``; these tests assert the
served HTML contains no early ``</script>``, and that the payload still
round-trips byte-for-byte through ``json.loads`` so the escaping changed only
the transport and not the data.

Runnable three ways, all verified:

    python3 tests/test_server.py                  # no pytest needed
    python3 -m unittest discover -s tests -v
    python3 -m pytest tests/test_server.py -q

Use the interpreter that has torch and fastapi:
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3

Set GLASSBOX_SLOW_TESTS=1 to also run the end-to-end test that trains a real
short run through POST /api/run (about 20 seconds).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server  # noqa: E402  (sys.path must be set first)
from glassbox.export import DATA_END, DATA_START  # noqa: E402

#: The payload. This exact string closed the data element in the old server.
PAYLOAD = "</script><script>alert(document.domain)</script>"

PARAGRAPH = f"pip has two red stones. {PAYLOAD} and three blue stones."

APP_SHELL = textwrap.dedent(
    f"""\
    <!doctype html>
    <html><head><title>Glassbox</title></head>
    <body>
    <h1>Glassbox</h1>
    {DATA_START}{{}}{DATA_END}
    <script>
    'use strict';
    const RUN = JSON.parse(document.getElementById('run-data').textContent);
    </script>
    </body></html>
    """
)


def data_block_of(html: str) -> str:
    """The raw text between the data element's opening and closing tags."""
    start = html.index(DATA_START) + len(DATA_START)
    return html[start : html.index(DATA_END, start)]


def fake_run(paragraph: str) -> Dict[str, Any]:
    """A run-artifact-shaped dict carrying the payload where learner text goes."""
    return {
        "manifest": {"run_id": "run_abcdef123456", "seed": 42, "parameters": 3048},
        "data": {"paragraphs": [paragraph], "vocab": ["<pad>", "<bos>", "<eos>", "<unk>"]},
        "checkpoints": {},
        "checkpoint_order": [],
        "stages": {},
        "checks": {"total": 0, "passed": 0, "failed": 0, "all_passed": True},
    }


def loaded(artifact: Dict[str, Any]) -> server.LoadedRun:
    return server.LoadedRun(
        run_id=str(artifact["manifest"]["run_id"]),
        artifact=artifact,
        source="test",
        weights=None,
        loaded_at=0.0,
    )


class ScriptBreakoutTests(unittest.TestCase):
    """The injection path, closed and kept closed."""

    def test_payload_would_have_broken_out_of_raw_json_dumps(self) -> None:
        """Establish that the vulnerability is real, not hypothetical.

        If this test ever fails it means json.dumps started escaping ``<`` and
        the rest of these tests are guarding nothing.
        """
        raw = json.dumps(fake_run(PARAGRAPH))
        self.assertIn("</script>", raw)
        self.assertIn("<script>", raw)

    def test_escape_for_script_removes_every_angle_bracket(self) -> None:
        escaped = server.escape_for_script(fake_run(PARAGRAPH))
        self.assertNotIn("<", escaped)
        self.assertNotIn("</script>", escaped)
        self.assertIn("\\u003c", escaped)

    def test_escaping_preserves_the_data_exactly(self) -> None:
        """The fix must change transport only. JSON.parse undoes \\u003c."""
        artifact = fake_run(PARAGRAPH)
        restored = json.loads(server.escape_for_script(artifact))
        self.assertEqual(restored, artifact)
        self.assertEqual(restored["data"]["paragraphs"][0], PARAGRAPH)

    def test_rendered_page_has_no_early_script_close(self) -> None:
        html, note = server.render_app_html(APP_SHELL, loaded(fake_run(PARAGRAPH)))
        block = data_block_of(html)
        self.assertNotIn("</script>", block)
        self.assertNotIn("<", block)
        self.assertIn("run_abcdef123456", note)
        self.assertEqual(json.loads(block)["data"]["paragraphs"][0], PARAGRAPH)

    def test_rendered_page_has_exactly_one_data_element(self) -> None:
        """An early close would leave a second, injected <script> in the page."""
        html, _ = server.render_app_html(APP_SHELL, loaded(fake_run(PARAGRAPH)))
        self.assertEqual(html.count(DATA_START), 1)
        # The shell's own script plus the data element's close: two closes, and
        # no third one contributed by the payload.
        self.assertEqual(html.count("</script>"), 2)
        self.assertNotIn("alert(document.domain)</script>", html)

    def test_served_html_is_escaped_end_to_end(self) -> None:
        """Through the real HTTP handler, with the app shell on disk."""
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            shell = Path(tmp) / "index.html"
            shell.write_text(APP_SHELL)
            original_html, original_state = server.APP_HTML, server._STATE
            try:
                server.APP_HTML = shell
                server._STATE = loaded(fake_run(PARAGRAPH))
                with TestClient(server.app) as client:
                    response = client.get("/")
                self.assertEqual(response.status_code, 200)
                block = data_block_of(response.text)
                self.assertNotIn("</script>", block)
                self.assertNotIn("<", block)
                self.assertEqual(
                    json.loads(block)["data"]["paragraphs"][0], PARAGRAPH
                )
            finally:
                server.APP_HTML, server._STATE = original_html, original_state

    def test_missing_app_shell_is_a_message_not_a_traceback(self) -> None:
        from fastapi.testclient import TestClient

        original = server.APP_HTML
        try:
            server.APP_HTML = Path(tempfile.gettempdir()) / "glassbox-absent.html"
            with TestClient(server.app) as client:
                response = client.get("/")
            self.assertEqual(response.status_code, 503)
            self.assertIn("not built yet", response.text)
        finally:
            server.APP_HTML = original

    def test_shell_without_a_data_block_is_served_unchanged(self) -> None:
        html, note = server.render_app_html("<h1>mid-edit</h1>", loaded(fake_run("x")))
        self.assertEqual(html, "<h1>mid-edit</h1>")
        self.assertIn("no", note)

    def test_unclosed_data_block_is_served_unchanged(self) -> None:
        shell = f"<h1>hi</h1>{DATA_START}{{}}"
        html, note = server.render_app_html(shell, loaded(fake_run("x")))
        self.assertEqual(html, shell)
        self.assertIn("not closed", note)


class ImportCostTests(unittest.TestCase):
    """The previous module trained inside its own import. This one must not."""

    def test_import_is_cheap_and_does_not_import_torch(self) -> None:
        code = (
            "import time, sys\n"
            "t = time.perf_counter()\n"
            "import server\n"
            "print(time.perf_counter() - t)\n"
            "print('torch' in sys.modules)\n"
            "print('glassbox' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        seconds, torch_loaded, glassbox_loaded = result.stdout.split()
        self.assertLess(float(seconds), 5.0, f"import server took {seconds}s")
        self.assertEqual(torch_loaded, "False", "importing server imported torch")
        self.assertEqual(glassbox_loaded, "False", "importing server imported glassbox")


class ValidationTests(unittest.TestCase):
    """Every bound is enforced as a 422, and no input reaches a traceback."""

    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient

        cls.client = TestClient(server.app)
        if server.ensure_loaded() is None:
            raise unittest.SkipTest("no runs/latest.json to validate against")

    def post(self, path: str, body: Dict[str, Any]) -> Any:
        return self.client.post(path, json=body)

    def test_instruction_without_answer_is_422(self) -> None:
        response = self.post("/api/run", {"instructions": [{"prompt": "q"}]})
        self.assertEqual(response.status_code, 422)
        self.assertIn("answer", json.dumps(response.json()))

    def test_instruction_with_unexpected_key_is_422(self) -> None:
        response = self.post(
            "/api/run", {"instructions": [{"prompt": "q", "response": "a"}]}
        )
        self.assertEqual(response.status_code, 422)

    def test_oversized_corpus_is_422(self) -> None:
        paragraphs = ["y" * 1000] * 10
        response = self.post("/api/run", {"paragraphs": paragraphs})
        self.assertEqual(response.status_code, 422)
        self.assertIn(str(server.MAX_CORPUS_CHARS), json.dumps(response.json()))

    def test_too_many_paragraphs_is_422(self) -> None:
        response = self.post(
            "/api/run", {"paragraphs": ["a valid paragraph."] * (server.MAX_PARAGRAPHS + 1)}
        )
        self.assertEqual(response.status_code, 422)

    def test_paragraph_too_short_to_window_is_422(self) -> None:
        response = self.post("/api/run", {"paragraphs": ["tiny"]})
        self.assertEqual(response.status_code, 422)

    def test_unknown_checkpoint_is_400_and_lists_the_valid_names(self) -> None:
        response = self.post("/api/infer", {"prompt": "hi", "checkpoint": "nope"})
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertEqual(detail["valid_checkpoints"], list(server.CHECKPOINT_NAMES))

    def test_epsilon_zero_is_422_not_a_zero_division(self) -> None:
        response = self.post("/api/gradient-check", {"epsilon": 0.0})
        self.assertEqual(response.status_code, 422)

    def test_negative_index_is_rejected_not_wrapped(self) -> None:
        response = self.post("/api/gradient-check", {"row": 0, "column": -7})
        self.assertEqual(response.status_code, 422)

    def test_out_of_range_row_reports_the_real_shape(self) -> None:
        response = self.post(
            "/api/gradient-check",
            {"checkpoint": "initial", "matrix": "blocks.0.mlp_up.weight", "row": 9999},
        )
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        self.assertEqual(detail["shape"], [16, 8])
        self.assertEqual(detail["valid_row_range"], [0, 15])

    def test_run_id_path_cannot_escape_the_runs_directory(self) -> None:
        response = self.client.get("/api/run/..long")
        self.assertEqual(response.status_code, 422)


class GradientCheckTests(unittest.TestCase):
    """The finite difference must actually be computed, at the requested cell."""

    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient

        cls.client = TestClient(server.app)
        if server.ensure_loaded() is None:
            raise unittest.SkipTest("no runs/latest.json to check against")

    def check(self, **body: Any) -> Dict[str, Any]:
        body.setdefault("checkpoint", "initial")
        response = self.client.post("/api/gradient-check", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_autograd_matches_the_central_difference(self) -> None:
        result = self.check(matrix="blocks.0.mlp_up.weight", row=3, column=5)
        fd = result["finite_difference"]
        self.assertEqual(fd["dtype"], "float64")
        self.assertTrue(fd["passes"], fd)
        self.assertNotEqual(fd["loss_plus"], fd["loss_minus"])

    def test_the_requested_cell_is_the_cell_checked(self) -> None:
        """Not sgd_step_demo's own largest-gradient pick."""
        a = self.check(matrix="blocks.0.mlp_up.weight", row=0, column=0)
        b = self.check(matrix="blocks.0.mlp_up.weight", row=3, column=5)
        self.assertNotEqual(
            a["finite_difference"]["analytic"], b["finite_difference"]["analytic"]
        )

    def test_epsilon_changes_the_estimate(self) -> None:
        coarse = self.check(matrix="blocks.0.mlp_up.weight", row=3, column=5, epsilon=1e-2)
        fine = self.check(matrix="blocks.0.mlp_up.weight", row=3, column=5, epsilon=1e-4)
        self.assertNotEqual(
            coarse["finite_difference"]["estimate"], fine["finite_difference"]["estimate"]
        )
        # The analytic gradient does not depend on epsilon.
        self.assertAlmostEqual(
            coarse["finite_difference"]["analytic"],
            fine["finite_difference"]["analytic"],
            places=12,
        )


class InferTests(unittest.TestCase):
    """``checkpoint`` must select weights, and the decode controls must bite."""

    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient

        cls.client = TestClient(server.app)
        if server.ensure_loaded() is None:
            raise unittest.SkipTest("no runs/latest.json to infer from")

    def infer(self, **body: Any) -> Dict[str, Any]:
        body.setdefault("checkpoint", "initial")
        body.setdefault("max_new_tokens", 10)
        response = self.client.post("/api/infer", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_initial_checkpoint_reproduces_the_artifacts_own_sample(self) -> None:
        """Random init is deterministic, so this must match what the run recorded."""
        run = server.require_loaded()
        sample = run.artifact["checkpoints"]["initial"]["samples"]["pip has "]
        # The run recorded this under DecodeConfig(max_new_tokens=24).
        result = self.infer(prompt="pip has ", max_new_tokens=24)
        self.assertEqual(result["text"], sample["text"])
        self.assertEqual(result["new_ids"], sample["new_ids"])

    def test_seed_makes_sampling_reproducible(self) -> None:
        a = self.infer(prompt="pip has ", temperature=1.2, seed=7)
        b = self.infer(prompt="pip has ", temperature=1.2, seed=7)
        self.assertEqual(a["text"], b["text"])
        self.assertTrue(a["deterministic"])

    def test_temperature_actually_changes_the_output(self) -> None:
        greedy = self.infer(prompt="pip has ")
        sampled = self.infer(prompt="pip has ", temperature=1.2, seed=7)
        self.assertNotEqual(greedy["text"], sampled["text"])
        self.assertIn("greedy", greedy["protocol"])
        self.assertIn("temperature 1.2", sampled["protocol"])

    def test_kv_cache_does_not_change_the_tokens(self) -> None:
        with_cache = self.infer(prompt="pip has ", use_cache=True)
        without = self.infer(prompt="pip has ", use_cache=False)
        self.assertEqual(with_cache["new_ids"], without["new_ids"])

    def test_out_of_vocabulary_characters_are_reported_not_hidden(self) -> None:
        vocab = set(server.require_loaded().artifact["data"]["vocab"])
        prompt = "ZEBRA"
        expected = sorted({c for c in prompt if c not in vocab})
        self.assertTrue(expected, "this run's vocabulary already covers the probe")
        result = self.infer(prompt=prompt)
        self.assertEqual(result["unknown_characters"], expected)
        self.assertIsNotNone(result["unknown_character_note"])


@unittest.skipUnless(
    os.environ.get("GLASSBOX_SLOW_TESTS") == "1",
    "set GLASSBOX_SLOW_TESTS=1 to train a real run (about 20 seconds)",
)
class EndToEndRunTests(unittest.TestCase):
    """Train a real short run whose corpus carries the payload, then serve it."""

    def test_real_run_with_payload_serves_safe_html(self) -> None:
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            shell = Path(tmp) / "index.html"
            shell.write_text(APP_SHELL)
            original_html, original_state = server.APP_HTML, server._STATE
            try:
                server.APP_HTML = shell
                with TestClient(server.app) as client:
                    built = client.post(
                        "/api/run",
                        json={
                            "paragraphs": [
                                PARAGRAPH,
                                "mira visits the garden each morning with a green leaf.",
                            ],
                            "pretrain_steps": 5,
                            "sft_steps": 5,
                            "warm_start_steps": 5,
                            "dpo_steps": 5,
                            "grpo_iterations": 1,
                            "grpo_group_size": 2,
                            "save": False,
                        },
                    )
                    self.assertEqual(built.status_code, 200, built.text)
                    summary = built.json()
                    # The tokenizer must have been rebuilt from the payload text,
                    # which contributes '<', '>', '/', '(', ')'.
                    self.assertGreater(summary["vocabulary_size"], 45)
                    self.assertEqual(summary["corpus"], "submitted")
                    self.assertEqual(
                        summary["checkpoints"], list(server.CHECKPOINT_NAMES)
                    )

                    page = client.get("/")
                    self.assertEqual(page.status_code, 200)
                    block = data_block_of(page.text)
                    self.assertNotIn("</script>", block)
                    self.assertNotIn("<", block)
                    restored = json.loads(block)
                    self.assertEqual(restored["data"]["paragraphs"][0], PARAGRAPH)

                    # Every checkpoint is now decodable from real weights.
                    for label in server.CHECKPOINT_NAMES:
                        reply = client.post(
                            "/api/infer",
                            json={
                                "prompt": "pip has ",
                                "checkpoint": label,
                                "max_new_tokens": 8,
                            },
                        )
                        self.assertEqual(reply.status_code, 200, reply.text)
                        self.assertEqual(reply.json()["checkpoint"], label)
            finally:
                server.APP_HTML, server._STATE = original_html, original_state


if __name__ == "__main__":
    unittest.main(verbosity=2)


class MissingWeightsTests(unittest.TestCase):
    """The 409 path: a real checkpoint whose tensors are not loaded.

    This is the case the old server got wrong in the most damaging way. It
    accepted ``checkpoint`` and ignored it, so a request for ``base`` was served
    from the SFT weights and *labelled* ``base``. The replacement must refuse,
    and it must distinguish three situations that look similar from outside:

    * an unknown name            -> 400, here is the valid list
    * a known name, no tensors   -> 409, here is what IS available and the remedy
    * ``initial``                -> 200, because random init is reconstructible
      from the manifest's config and seed without any stored weights

    A run artifact with no sidecar is the honest way to reach the middle case,
    so this installs one directly rather than deleting a file on disk.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient

        cls.client = TestClient(server.app)

    def setUp(self) -> None:
        real = server.ensure_loaded()
        if real is None:
            self.skipTest("no runs/latest.json to derive a weightless run from")
        # Same artifact, weights deliberately absent.
        self._saved = server._STATE
        server._STATE = server.LoadedRun(
            run_id=real.run_id,
            artifact=real.artifact,
            source="test (weights withheld)",
            weights=None,
            loaded_at=0.0,
        )

    def tearDown(self) -> None:
        server._STATE = self._saved

    def test_a_known_checkpoint_without_tensors_is_a_409_not_a_substitution(self) -> None:
        for label in ("base", "sft", "lora", "dpo", "warm_start", "rl"):
            with self.subTest(checkpoint=label):
                response = self.client.post(
                    "/api/infer", json={"checkpoint": label, "prompt": "pip has "}
                )
                self.assertEqual(response.status_code, 409, response.text)
                detail = response.json()["detail"]
                self.assertIn(label, detail["error"])
                # It must say what IS available rather than silently answering.
                self.assertIn("available_now", detail)
                self.assertIn("initial", detail["available_now"])
                self.assertNotIn(label, detail["available_now"])
                # And it must tell the reader how to fix it.
                remedy = " ".join(str(v) for v in detail.values())
                self.assertTrue(
                    any(word in remedy for word in ("weights", "run", "rebuild")),
                    f"409 for {label} gives no remedy: {detail}",
                )

    def test_no_text_is_returned_for_a_checkpoint_without_tensors(self) -> None:
        """The failure must be total. A 409 carrying a generation would be worse
        than a 500, because a caller might use it."""
        response = self.client.post(
            "/api/infer", json={"checkpoint": "sft", "prompt": "pip has "}
        )
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("text", response.json())

    def test_initial_still_works_because_it_needs_no_stored_weights(self) -> None:
        response = self.client.post(
            "/api/infer",
            json={"checkpoint": "initial", "prompt": "pip has ", "max_new_tokens": 8},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["checkpoint"], "initial")
        self.assertIsInstance(body["text"], str)

    def test_an_unknown_name_is_still_400_not_409(self) -> None:
        """The two failures must stay distinguishable: 400 means 'no such
        checkpoint', 409 means 'that checkpoint exists but is not loaded'."""
        response = self.client.post(
            "/api/infer", json={"checkpoint": "nope", "prompt": "pip has "}
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("valid_checkpoints", response.json()["detail"])
