"""The Paper A1 figures, drawn from what a run saved.

One module per figure, each with a ``figure_<name>(inputs)`` that reads what it needs from a
:class:`FigureInputs` and a ``draw_<name>(data)`` that draws it from plain tables:

======================  ==================  ======================================================
File name               Module              What it shows
======================  ==================  ======================================================
fig01_design            :mod:`.design`      the experiment as boxes and arrows
fig02_map               :mod:`.maps`        the points by carbonate class; points per basin
fig03_correlations      :mod:`.correlations`  SOM's correlations within each class; spectra
fig04_stage1            :mod:`.stage1`      stage 1's CaCO3 and CEC, and the class confusion
fig05_spatial           :mod:`.spatial`     scores as test points move away from training
fig06_obs_pred          :mod:`.obs_pred`    measured against predicted SOM, global model
fig07_class_bias        :mod:`.class_bias`  bias within each carbonate class
fig08_transfer          :mod:`.transfer`    transferred against local, per basin and group
fig09_mixed             :mod:`.transfer`    what moving local points into training wins back
fig10_basins            :mod:`.basins`      how each held-out basin differs (a table)
fig11_shap_groups       :mod:`.shap`        what the global model relies on, per group
figS1-figS5             (same modules)      every block size; every model; class scores; SHAP
                                            per input; transfer against dissimilarity
======================  ==================  ======================================================

:func:`render_all` (:mod:`.render`) draws them, saves each as PNG and PDF, and writes the tables
each was drawn from (``figures/data/``) and a draft caption for each (``figures/captions.md``).
Plain names come from ``configs/a1/labels.yml`` (:func:`pretty_name`); sizes, colours, markers and
axis wording from :mod:`.style`.
"""

from yg_eo_soilnet.a1.figures.core import (
    BAREST_BAND,
    S2_WAVELENGTHS,
    FigureInputs,
    _input_column,
    _spearman_columns,
    _tree_shap,
    attach,
    bare_soil_spectra,
    class_confusion,
    class_counts,
    dissimilarity_frame,
    figure_caption,
    figure_data,
    is_message_figure,
    save_figure,
    shap_by_class,
    spearman_by_class,
    stage1_points,
    write_table,
)
from yg_eo_soilnet.a1.figures.labels import pretty_name, pretty_split
from yg_eo_soilnet.a1.figures.render import (
    FIGURE_NAMES,
    MAIN_FIGURES,
    SUPPLEMENTARY_FIGURES,
    render_all,
    select_figures,
)

__all__ = [
    "BAREST_BAND",
    "FIGURE_NAMES",
    "MAIN_FIGURES",
    "S2_WAVELENGTHS",
    "SUPPLEMENTARY_FIGURES",
    "FigureInputs",
    "_input_column",
    "_spearman_columns",
    "_tree_shap",
    "attach",
    "bare_soil_spectra",
    "class_confusion",
    "class_counts",
    "dissimilarity_frame",
    "figure_caption",
    "figure_data",
    "is_message_figure",
    "pretty_name",
    "pretty_split",
    "render_all",
    "save_figure",
    "select_figures",
    "shap_by_class",
    "spearman_by_class",
    "stage1_points",
    "write_table",
]
