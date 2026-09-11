# Glassbox
#
# PYTHON defaults to the only interpreter on this machine that has torch.
# Plain `python3` here is 3.12.1 with no torch, so every target would fail at
# the first import. Point it at a virtualenv to use one instead:
#
#     make run PYTHON=.venv/bin/python

PYTHON ?= /Library/Frameworks/Python.framework/Versions/3.9/bin/python3

.DEFAULT_GOAL := help
.PHONY: help install install-server run smoke checks checks-verbose test test-full test-all serve clean

help:
	@echo "Glassbox targets (PYTHON=$(PYTHON))"
	@echo ""
	@echo "  make install         install the engine requirements (torch, numpy)"
	@echo "  make install-server  also install the optional server extras"
	@echo "  make run             build a run artifact (~100-120s); writes runs/ and app/index.html"
	@echo "  make smoke           the same pipeline at tiny step counts, under a minute"
	@echo "  make checks          the 26 numerical checks on a trained model (~20s)"
	@echo "  make test            the test suite, skipping the slow export test"
	@echo "  make test-full       the whole test suite (~25s)"
	@echo "  make test-all        the whole suite with nothing skipped (~35s)"
	@echo "  make serve           serve the app and API on http://127.0.0.1:8077"
	@echo "  make clean           remove caches and generated run artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

# fastapi, uvicorn and pydantic. Only server.py needs these; the app itself
# is a self-contained HTML file and needs no server at all.
install-server:
	$(PYTHON) -m pip install -r requirements-server.txt

# The whole pipeline. Writes runs/<run_id>.json and runs/latest.json, and inlines
# the run data into app/index.html. Exits non-zero if any of the 26 checks fail.
run:
	$(PYTHON) -m glassbox.export

# Proves the wiring end to end in well under a minute. Its numbers are meaningless.
smoke:
	$(PYTHON) -m glassbox.export --pretrain-steps 50 --sft-steps 20 --grpo-iterations 3 --no-html --out /tmp/glassbox-smoke

# The 26 numerical checks alone, without building a full run artifact.
# This pretrains and instruction-tunes briefly first: several checks are
# vacuous on an untrained model (the loss-mask, EOS and DPO-reference checks
# all want a model that has actually been trained), so checking an untrained
# one would report 26/26 while exercising much less.
checks:
	$(PYTHON) -m glassbox.checks

# Same checks, verbose: prints every measured value beside its verdict.
checks-verbose:
	$(PYTHON) -m glassbox.checks --verbose

# Standard library only, so a missing dependency can never make the suite
# un-runnable. --fast skips the slow end-to-end export test.
test:
	$(PYTHON) -m tests.run_checks --fast

test-full:
	$(PYTHON) -m tests.run_checks

# Nothing skipped at all, including the server test that trains a real run to
# prove the script-injection payload is escaped in the served HTML.
test-all:
	GLASSBOX_SLOW_TESTS=1 $(PYTHON) -m tests.run_checks

serve:
	$(PYTHON) server.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -f runs/*.json runs/*.pt
	@echo "removed caches and generated run artifacts; rebuild them with: make run"
