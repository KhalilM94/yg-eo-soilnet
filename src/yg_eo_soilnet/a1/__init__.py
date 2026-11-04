"""The experiments of Paper A1: soil organic matter from static covariates and time-series percentiles.

Four questions, each answered by a script under ``experiments/a1/``:

(a) Does adding predicted carbonate (CaCO3) and cation exchange capacity (CEC) as inputs help
    predict soil organic matter (SOM)? The two are predicted by TabICL, without ever seeing the lab
    values of the points they are used to score.
(b) How does the error depend on how calcareous the soil is? Scores are broken down by FAO
    carbonate class, and one model per class is compared with one model for all.
(c) How well does a model trained in one region predict another?
(d) Does that loss grow with how different the two regions are?

Everything here is separate from ``main.py``: it reads the same data files through
:class:`~yg_eo_soilnet.data_manager.DataManager` and reuses the project's split, scores and figure
style, but it changes none of them. Its settings live in ``configs/a1/``.
"""
