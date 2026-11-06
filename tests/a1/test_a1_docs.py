"""docs/a1.md keeps up with the A1 scripts: every option they accept has a row, and no row is stale."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "a1.md"
SCRIPTS = ("make_demo_data.py", "preflight.py", "make_plans.py", "run_a1.py", "evaluate.py", "make_figures.py")


def _option_flags():
    """``option_flags`` from tests/test_docs.py, which reads a script's options without importing it."""
    spec = importlib.util.spec_from_file_location("_main_test_docs", ROOT / "tests" / "test_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.option_flags


def _section(script: str) -> str:
    """The part of docs/a1.md under the heading of one script."""
    text = PAGE.read_text()
    match = re.search(rf"### `{re.escape(script)}`\n(.*?)(?=\n### |\n## |\Z)", text, flags=re.S)
    assert match, f"docs/a1.md has no '### `{script}`' section under Options."
    return match.group(1)


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_option_of_each_a1_script_has_a_row(script):
    flags = _option_flags()(ROOT / "experiments" / "a1" / script)
    section = _section(script)
    missing = [flag for flag in flags if f"| `{flag}` |" not in section]
    assert flags and not missing, f"docs/a1.md, {script}: no row for {missing}."


@pytest.mark.parametrize("script", SCRIPTS)
def test_no_documented_option_is_gone(script):
    flags = set(_option_flags()(ROOT / "experiments" / "a1" / script))
    documented = set(re.findall(r"^\| `(--[a-z0-9-]+)` \|", _section(script), flags=re.M))
    assert documented <= flags, f"docs/a1.md documents {sorted(documented - flags)}, which {script} no longer accepts."


def test_the_page_is_linked_from_the_index():
    assert re.search(r"^a1$", (ROOT / "docs" / "index.md").read_text(), flags=re.M)


def test_the_demo_task_runs_every_step():
    task = next(line for line in (ROOT / "pyproject.toml").read_text().splitlines() if line.startswith("a1-demo ="))
    for script in ("preflight.py", "make_plans.py", "run_a1.py", "evaluate.py", "make_figures.py"):
        assert f"experiments/a1/{script}" in task
    assert "--dev-mode" in task and 'depends-on = ["demo-data-a1"]' in task
