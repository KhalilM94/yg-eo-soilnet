"""The documentation keeps up with the code: every command-line option has a page entry."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI_DOCS = ROOT / "docs" / "cli"

# script -> the page that documents it.
TOOLS = {
    "main.py": "main.md",
    "tune.py": "tune.md",
    "relog.py": "relog.md",
    "replot.py": "replot.md",
    "export_predictions.py": "export_predictions.md",
}


def option_flags(script_path: Path) -> list[str]:
    """Every long option a script declares, read from its source without importing it."""
    tree = ast.parse(script_path.read_text())
    flags: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "attr", "") != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and str(argument.value).startswith("--"):
                flags.append(argument.value)
    return flags


@pytest.mark.parametrize(("script", "page"), sorted(TOOLS.items()))
def test_every_option_is_documented(script, page):
    """A new option has to be added to its tool's page, or this fails."""
    flags = option_flags(ROOT / script)
    assert flags, f"{script} declares no options; has it stopped using argparse?"

    text = (CLI_DOCS / page).read_text()
    undocumented = sorted(flag for flag in flags if f"`{flag}`" not in text)
    assert not undocumented, (
        f"docs/cli/{page} does not mention {undocumented}. Add a row for each to its options "
        "table, so the page still describes the tool."
    )


@pytest.mark.parametrize("page", sorted(TOOLS.values()))
def test_pages_do_not_document_options_that_are_gone(page):
    """An option removed from a script has to be removed from its page too."""
    script = next(name for name, filename in TOOLS.items() if filename == page)
    flags = set(option_flags(ROOT / script))

    text = (CLI_DOCS / page).read_text()
    # Only the options tables are checked, where a flag is written as `--name` in a cell.
    documented = {line.split("`")[1] for line in text.splitlines() if line.startswith("| `--")}
    stale = sorted(documented - flags)
    assert not stale, f"docs/cli/{page} documents {stale}, which {script} no longer accepts."


def test_every_tool_has_a_page():
    """A new script needs a page, and a row in the tools table."""
    index = (CLI_DOCS / "index.md").read_text()
    for script, page in TOOLS.items():
        assert (CLI_DOCS / page).exists(), f"docs/cli/{page} is missing."
        assert page in index, f"docs/cli/index.md does not link to {page}."
