"""Run the whole suite with nothing but the standard library.

    /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m tests.run_checks

pytest is not required. If it happens to be installed, ``python3 -m pytest
tests`` collects exactly the same ``unittest.TestCase`` classes and reports the
same results; this entry point exists so the suite can never become
un-runnable because a dependency is missing.

Exit status is 0 only when every test passed. Expected failures are reported
separately and do NOT fail the run: each one is a defect recorded in the suite
with its full-strength assertion intact (see ``tests/README.md``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_suite(pattern: str = "test_*.py") -> unittest.TestSuite:
    loader = unittest.TestLoader()
    return loader.discover(
        start_dir=str(REPO_ROOT / "tests"),
        pattern=pattern,
        top_level_dir=str(REPO_ROOT),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the glassbox test suite without pytest."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="repeat for per-test output",
    )
    parser.add_argument(
        "-k",
        "--pattern",
        default="test_*.py",
        help="file pattern to discover (default: test_*.py)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the slow end-to-end export test (sets GLASSBOX_SKIP_SLOW=1)",
    )
    parser.add_argument(
        "--failfast", action="store_true", help="stop at the first failure"
    )
    args = parser.parse_args(argv)

    if args.fast:
        os.environ["GLASSBOX_SKIP_SLOW"] = "1"

    # Importing after the flag is set, so fixtures sees it.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tests import fixtures

    print("glassbox test suite")
    print("  python          {}".format(sys.version.split()[0]))
    try:
        import torch

        print("  torch           {}".format(torch.__version__))
    except ImportError:  # pragma: no cover - torch is required by the package
        print("  torch           NOT INSTALLED (the suite cannot run)")
        return 2
    print("  pretrain steps  {}".format(fixtures.PRETRAIN_STEPS))
    print("  slow tests      {}".format("skipped" if fixtures.SKIP_SLOW else "included"))
    print()

    suite = build_suite(args.pattern)
    started = time.time()
    result = unittest.TextTestRunner(
        verbosity=args.verbose, failfast=args.failfast
    ).run(suite)
    elapsed = time.time() - started

    ran = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    expected_failures = len(result.expectedFailures)
    unexpected_successes = len(result.unexpectedSuccesses)
    # testsRun counts skipped tests too, so a skip must come out of the passed
    # total or the summary claims a test passed that never executed.
    passed = (
        ran
        - failures
        - errors
        - skipped
        - expected_failures
        - unexpected_successes
    )

    print()
    print("-" * 62)
    print("tests collected     {}".format(ran))
    print("passed              {}".format(passed))
    print("failed              {}".format(failures))
    print("errors              {}".format(errors))
    print("skipped             {}".format(skipped))
    print("expected failures   {}  (known defects, see tests/README.md)".format(
        expected_failures
    ))
    print("unexpected passes   {}".format(unexpected_successes))
    print("wall clock          {:.1f}s".format(elapsed))
    for name, _ in result.expectedFailures:
        print("  expected failure: {}".format(name))
    for test, reason in result.skipped:
        print("  skipped: {}  ({})".format(test, reason))
    print("-" * 62)

    ok = not failures and not errors and not unexpected_successes
    print("RESULT: {}".format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
