# Contributing

How to run the checks, build this documentation, and write docstrings and pages the way the rest
of the project does.

## Running the checks

All commands run from the repository root, in the `dev` environment:

| Command | What it does |
|---|---|
| `pixi run -e dev test` | Runs every test (about 2.5 minutes). |
| `pixi run -e dev test-fast` | Skips the 20 slowest tests (about 2 minutes). |
| `pixi run -e dev doctest` | Runs the `>>>` examples written inside the docstrings. |
| `pixi run -e dev lint` | Checks the code style, and that every public module, class and function has a docstring. |
| `pixi run -e dev docs` | Builds this documentation into `docs/_build/html/`. Any warning fails the build. |

Open `docs/_build/html/index.html` in a browser to read the built documentation.

## Who we write for

The readers are environmental scientists and intermediate Python programmers. They know soil
science, remote sensing and Python; they do not necessarily know machine-learning vocabulary. Every
page and docstring is written so that such a reader can use the code without looking anything up.

## Writing rules

1. **Say what it does first, in one plain sentence.** The first line of a docstring is a complete
   sentence saying what the thing does or is. A second short paragraph may say when you would use
   it.
2. **Prefer everyday words to jargon.** "trained 10 times from different random starts, then
   averaged" rather than "a 10-member ensemble". When a technical word is the right one, link it to
   the {doc}`glossary` - ``:term:`ensemble` `` in a docstring, ``{term}`ensemble` `` in a page - or
   explain it in the same sentence.
3. **Give units.** Every value that has units says them: g/kg, %, metres, decimal years.
4. **Show an example.** Every public function and class has an `Examples` section. For functions
   that just compute something, write runnable `>>>` examples with their real output; they are
   checked by `pixi run -e dev doctest`. For classes that need data or a trained model, show a short
   usage snippet based on the demo dataset and mark it `# doctest: +SKIP`.
5. **Describe the code as it is, not its history.** No bug stories, run ids or dates - git keeps the
   history. Keep a reason only if it stops a reader from making a mistake, and write it as a rule:
   "Scaling is learned from the training points only, so test scores stay honest."
6. **Keep comments short and plain.** A comment explains a line whose purpose is not obvious.

## Docstring layout

Docstrings follow the [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html) that
numpy, pandas and scipy use. Functions and classes people call directly have `Parameters`,
`Returns` (or `Yields`), `Raises` where relevant, and `Examples`. Private helpers (names starting
with `_`) get one to three lines.

**Before** - accurate, but hard to use:

```python
def regression_metrics(y_true, y_pred, *, split="test", suffix=""):
    """The unified metric set for one (observed, predicted) pair, in original target units.

    Returns an empty dict when there are fewer than two finite pairs, rather than emitting NaN
    metrics that would then poison the leaderboard. Metrics whose denominator is degenerate
    (zero target variance for R2, zero RMSE for RPD/RPIQ) are individually omitted for the same
    reason - the same policy `_log_epoch_metrics` already applies on the Lightning side.
    """
```

**After**:

```python
def regression_metrics(y_true, y_pred, *, split="test", suffix=""):
    """Score predictions against lab measurements for one target.

    Parameters
    ----------
    y_true : array-like
        Measured values for one target, in its own units.
    y_pred : array-like
        Predicted values, in the same order and units.
    split : str, default "test"
        Added to every score name: ``"test"`` gives ``rmse_test``, ``mae_test``, ...

    Returns
    -------
    dict of str to float
        One entry per score. Empty if fewer than two points can be scored.

    Examples
    --------
    >>> scores = regression_metrics([10, 20, 30, 40], [12, 18, 33, 39])
    >>> round(scores["rmse_test"], 3)
    2.121
    """
```

## Documentation pages

Pages live in `docs/` and are written in Markdown ([MyST](https://myst-parser.readthedocs.io/)).
Every page shows at least one concrete example - a command, a YAML snippet or an excerpt of an
output file. The reference pages under `docs/reference/` are built automatically from the
docstrings; you do not edit them to change what they say, you edit the docstrings.

When you add a command-line option, add it to the tool's page under `docs/cli/`: the test
`tests/test_docs.py` fails if an option is missing from its page.
