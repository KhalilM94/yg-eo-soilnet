"""Draw every A1 figure, and write what each was drawn from and a draft caption for each.

:func:`render_all` saves, in the run folder's ``figures/`` unless told otherwise:

- ``<name>.png`` and ``<name>.pdf`` for each figure of :data:`FIGURE_NAMES` it draws;
- ``data/<table>.csv``: every table a figure plotted;
- ``data/table_*.tex``: the tables meant for the paper (RPIQ by class, the basins), also as LaTeX;
- ``captions.md``: one draft caption per figure, with the numbers read from those tables.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Sequence

import matplotlib.pyplot as plt

from yg_eo_soilnet.a1.figures import (
    basins,
    class_bias,
    correlations,
    design,
    maps,
    obs_pred,
    shap,
    spatial,
    stage1,
    transfer,
)
from yg_eo_soilnet.a1.figures.core import (
    FigureInputs,
    figure_caption,
    figure_data,
    is_message_figure,
    save_figure,
    write_table,
)
from yg_eo_soilnet.a1.plans import RANDOM

logger = logging.getLogger(__name__)

#: The main-text figures, in order.
MAIN_FIGURES = (
    "fig01_design",
    "fig02_map",
    "fig03_correlations",
    "fig04_stage1",
    "fig05_spatial",
    "fig06_obs_pred",
    "fig07_class_bias",
    "fig08_transfer",
    "fig09_mixed",
    "fig10_basins",
    "fig11_shap_groups",
)
#: The supplementary figures, in order.
SUPPLEMENTARY_FIGURES = (
    "figS1_obs_pred_blocks",
    "figS2_obs_pred_models",
    "figS3_class_scores",
    "figS4_shap_features",
    "figS5_dissimilarity",
)
#: Every figure :func:`render_all` draws, by the file name each is saved under.
FIGURE_NAMES = MAIN_FIGURES + SUPPLEMENTARY_FIGURES
#: The figures' tables that are also written as LaTeX, for the paper.
LATEX_TABLES = ("table_rpiq_by_class", "table_basins")
#: The captions file.
CAPTIONS_FILE = "captions.md"


def _builders(
    inputs: FigureInputs, *, relief: bool, cache_dir: Optional[str], n_boot: int, shap_plan: str
) -> dict[str, Callable]:
    return {
        "fig01_design": lambda: design.figure_design(inputs),
        "fig02_map": lambda: maps.figure_map(inputs, relief=relief, cache_dir=cache_dir),
        "fig03_correlations": lambda: correlations.figure_correlations(inputs, n=n_boot),
        "fig04_stage1": lambda: stage1.figure_stage1(inputs, plan_id=RANDOM),
        "fig05_spatial": lambda: spatial.figure_spatial(inputs),
        "fig06_obs_pred": lambda: obs_pred.figure_obs_pred(inputs),
        "fig07_class_bias": lambda: class_bias.figure_class_bias(inputs),
        "fig08_transfer": lambda: transfer.figure_transfer(inputs),
        "fig09_mixed": lambda: transfer.figure_mixed(inputs),
        "fig10_basins": lambda: basins.figure_basins(inputs),
        "fig11_shap_groups": lambda: shap.figure_shap_groups(inputs, plan_id=shap_plan),
        "figS1_obs_pred_blocks": lambda: obs_pred.figure_obs_pred_blocks(inputs),
        "figS2_obs_pred_models": lambda: obs_pred.figure_obs_pred_models(inputs),
        "figS3_class_scores": lambda: class_bias.figure_class_scores(inputs),
        "figS4_shap_features": lambda: shap.figure_shap_features(inputs, plan_id=shap_plan),
        "figS5_dissimilarity": lambda: basins.figure_dissimilarity(inputs),
    }


def select_figures(only: Optional[Sequence[str]] = None, skip: Optional[Sequence[str]] = None) -> list[str]:
    """The figures to draw, in :data:`FIGURE_NAMES` order: ``only`` those (all by default), minus ``skip``.

    Raises
    ------
    ValueError
        A name that is not in :data:`FIGURE_NAMES`.

    Examples
    --------
    >>> select_figures(only=["fig06_obs_pred", "fig05_spatial"])
    ['fig05_spatial', 'fig06_obs_pred']
    >>> len(select_figures(skip=["fig02_map"])) == len(FIGURE_NAMES) - 1
    True
    """
    unknown = sorted(set(only or ()).union(skip or ()) - set(FIGURE_NAMES))
    if unknown:
        raise ValueError(f"Unknown figure names {unknown}; the figures are {list(FIGURE_NAMES)}.")
    chosen = [name for name in FIGURE_NAMES if not only or name in set(only)]
    return [name for name in chosen if name not in set(skip or ())]


def render_all(
    inputs: FigureInputs,
    folder=None,
    *,
    relief: bool = True,
    cache_dir: Optional[str] = None,
    n_boot: Optional[int] = None,
    shap_plan: str = RANDOM,
    close: bool = False,
    only: Optional[Sequence[str]] = None,
    skip: Optional[Sequence[str]] = None,
) -> dict:
    """Draw the figures, save each as PNG and PDF, and write their data and draft captions.

    Parameters
    ----------
    inputs : FigureInputs
    folder : str or pathlib.Path, optional
        Where to save; the run folder's ``figures/`` unless given.
    relief : bool, default True
        Shaded relief under the map (downloaded once, then cached); False draws flat ground.
    cache_dir : str, optional
        Where the map's tiles and borders are cached.
    n_boot : int, optional
        Bootstrap draws for the intervals the figures compute themselves; ``inputs.n_boot`` unless given.
    shap_plan : str, default "random"
        The plan whose global model figures 11 and S4 explain.
    close : bool, default False
        Close each figure once saved, to free memory when they are not going to be shown.
    only, skip : list of str, optional
        Draw only these figures / not these (names of :data:`FIGURE_NAMES`).

    Returns
    -------
    dict
        File name to figure, in :data:`FIGURE_NAMES` order.
    """
    names = select_figures(only, skip)
    folder = Path(folder) if folder else inputs.run_dir / "figures"
    if n_boot is not None:
        inputs.n_boot = n_boot
    builders = _builders(inputs, relief=relief, cache_dir=cache_dir, n_boot=inputs.n_boot, shap_plan=shap_plan)
    figures, captions = {}, {}
    for name in names:
        figure = builders[name]()
        save_figure(figure, name, folder)
        for table_name, table in figure_data(figure).items():
            if table_name in LATEX_TABLES:
                write_table(table, folder / "data" / table_name)
            else:
                (folder / "data").mkdir(parents=True, exist_ok=True)
                table.to_csv(folder / "data" / f"{table_name}.csv", index=False)
        captions[name] = figure_caption(figure)
        if is_message_figure(figure):
            logger.warning(f"{name}: {figure._a1_message}")
        else:
            logger.info(f"{name}: drawn")
        if close:
            plt.close(figure)
        figures[name] = figure
    write_captions(captions, folder / CAPTIONS_FILE)
    return figures


def write_captions(captions: dict, path) -> Path:
    """Write ``captions.md``: one section per figure, its draft caption under it.

    Captions already in the file for figures not drawn this time are kept, so ``--only`` does not
    lose the others.
    """
    path = Path(path)
    kept = read_captions(path) if path.exists() else {}
    kept.update(captions)
    ordered = [name for name in FIGURE_NAMES if name in kept] + [name for name in kept if name not in FIGURE_NAMES]
    lines = ["# Draft figure captions", "", "Written by make_figures.py from the run's tables; edit before use.", ""]
    for name in ordered:
        lines += [f"## {name}", "", kept[name].strip(), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def read_captions(path) -> dict:
    """The captions of a ``captions.md`` (:func:`write_captions`), by figure name."""
    captions, name, body = {}, None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if name:
                captions[name] = "\n".join(body).strip()
            name, body = line[3:].strip(), []
        elif name:
            body.append(line)
    if name:
        captions[name] = "\n".join(body).strip()
    return captions
