"""Explaining the deep-learning model.

It reads dated readings rather than a table of numbers, so there is no plain list of inputs to hide
one at a time. Instead the contributions are traced back through the model, from the point where
the readings have been laid out on the :term:`calendar grid`: the model says which tensors an
explanation should vary (``explanation_parts``) and how to predict from exactly those
(``forward_from_parts``), and the contributions are added up per input.

Each band of each :term:`data source` therefore gets its own contribution, as does each covariate,
each category and the location, so a figure can say the July Sentinel-2 red band mattered rather
than only that "the time series" did.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yg_eo_soilnet.explain.result import STANDARDIZED_LOG1P, ShapResult

# Torch is imported lazily throughout: this module is imported by explain/__init__, which the logger
# reaches only when EXPLAIN_ENABLED is true, but keeping the heavy imports inside functions means an
# import of the package itself stays cheap.


def _collect_batches(datamodule, limit: int) -> list[dict]:
    """Read batches until enough points have been gathered, or the data runs out."""
    loader_factory = getattr(datamodule, "test_dataloader", None)
    if loader_factory is None:
        raise TypeError("The datamodule has no test_dataloader(); nothing to explain.")

    try:
        datamodule.setup("test")
    except Exception:
        # Already set up, or this datamodule does not need a test stage. Either way the loader
        # below is the real check.
        pass

    batches: list[dict] = []
    gathered = 0
    for batch in loader_factory():
        batches.append(batch)
        gathered += int(np.asarray(batch["x_static"]).shape[0])
        if gathered >= limit:
            break
    return batches


def _concatenate_parts(per_batch: list[list], torch) -> list:
    """Join each input across batches, checking the shapes agree.

    They can disagree only when the grid was left to size itself per batch, in which case the run is
    told to fix ``grid_years`` rather than being given a quietly wrong explanation.
    """
    if len(per_batch) == 1:
        return per_batch[0]

    reference = per_batch[0]
    for parts in per_batch[1:]:
        if any(a.shape[1:] != b.shape[1:] for a, b in zip(reference, parts)):
            return reference

    return [torch.cat([parts[index] for parts in per_batch], dim=0) for index in range(len(reference))]


def _group_values(part_values: list[np.ndarray], groups: list[dict]) -> np.ndarray:
    """Add up the raw contributions into one number per input.

    Every cell of a band's grid has its own raw contribution; the band's contribution is their sum, and
    adding them is valid because SHAP contributions add.
    """
    columns = []
    for group in groups:
        array = part_values[group["part"]]
        selected = array[:, group["columns"], ...]
        summed = selected.reshape(selected.shape[0], -1).sum(axis=1)
        columns.append(summed)
    return np.stack(columns, axis=1) if columns else np.empty((0, 0))


def _destandardize(values: np.ndarray, mean: list[float], scale: list[float]) -> np.ndarray:
    """Put the covariates back in their own units, so the figures' colours mean something."""
    if len(mean) != values.shape[-1] or len(scale) != values.shape[-1]:
        return values
    return values * np.asarray(scale, dtype=np.float64) + np.asarray(mean, dtype=np.float64)


def _colour_values(model, parts, groups: list[dict], categorical_codes, state: dict, torch) -> np.ndarray:
    """One value per point per input, for the figures' colour axis.

    An input spanning many columns has no single value of its own, so each kind gets whatever means
    something for it: a band gets its average reading, a category its code.
    """
    n_samples = int(parts[0].shape[0])
    colours = np.full((n_samples, len(groups)), np.nan, dtype=np.float64)

    static_values = _destandardize(
        parts[0].detach().cpu().numpy().astype(np.float64),
        state.get("static_mean") or [],
        state.get("static_scale") or [],
    )

    sequence_mean = state.get("sequence_mean") or {}
    sequence_scale = state.get("sequence_scale") or {}

    label_mean = np.asarray(state.get("label_mean") or [], dtype=np.float64)
    label_scale = np.asarray(state.get("label_scale") or [], dtype=np.float64)
    # For the residual base block, which is published in the target's space rather than the lab
    # standardizer's and so inverts with these instead.
    target_mean = np.asarray(state.get("target_mean") or [], dtype=np.float64)
    target_scale = np.asarray(state.get("target_scale") or [], dtype=np.float64)
    coord_min = np.asarray(state.get("coord_min") or [], dtype=np.float64)
    coord_max = np.asarray(state.get("coord_max") or [], dtype=np.float64)
    auxiliary_index = getattr(model, "auxiliary_index", None)
    auxiliary_index = None if auxiliary_index is None else auxiliary_index.detach().cpu().numpy()

    categorical_names = (
        list(getattr(getattr(model, "static_encoder", None), "embeddings", None).feature_names)
        if (
            getattr(model, "has_static_features", False)
            and getattr(getattr(model, "static_encoder", None), "embeddings", None) is not None
        )
        else []
    )

    for column, group in enumerate(groups):
        kind = group["kind"]

        if kind in ("static", "context"):
            colours[:, column] = static_values[:, group["columns"][0]]

        elif kind == "spatial":
            # Back to degrees, undoing the train-bbox mapping the datamodule applied. Colouring by
            # the normalized value would work but reads as a meaningless -1..1; in degrees the
            # beeswarm's colour axis is literally "how far north", which is the thing worth seeing.
            normalized = parts[group["part"]].detach().cpu().numpy().astype(np.float64)
            position = group["columns"][0]
            value = normalized[:, position]
            if position < coord_min.size and position < coord_max.size:
                span = coord_max[position] - coord_min[position]
                value = (value + 1.0) / 2.0 * span + coord_min[position]
            colours[:, column] = value

        elif kind == "categorical" and categorical_codes is not None:
            if group["name"] in categorical_names:
                colours[:, column] = categorical_codes[:, categorical_names.index(group["name"])]

        elif kind == "temporal":
            modality = group["modality"]
            layout = model.rasterizers[modality].channel_layout()
            if group["columns"][0] not in layout["values"]:
                continue  # the month positional row: a coordinate, left as NaN
            band = layout["values"].index(group["columns"][0])
            grid = parts[group["part"]].detach().cpu().numpy().astype(np.float64)
            observed = grid[:, layout["cell_observed"][0]] > 0.5
            readings = grid[:, group["columns"][0]]
            counts = observed.reshape(observed.shape[0], -1).sum(axis=1)
            totals = (readings * observed).reshape(readings.shape[0], -1).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                means = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
            mean_values = sequence_mean.get(modality) or []
            scale_values = sequence_scale.get(modality) or []
            if band < len(mean_values) and band < len(scale_values):
                means = means * float(scale_values[band]) + float(mean_values[band])
            colours[:, column] = means

        elif kind == "auxiliary":
            selected = parts[group["part"]].detach().cpu().numpy().astype(np.float64)
            position = group["columns"][0]
            value = selected[:, position]
            if auxiliary_index is not None and position < len(auxiliary_index):
                roster_index = int(auxiliary_index[position])
                if roster_index < label_mean.size and roster_index < label_scale.size:
                    value = value * label_scale[roster_index] + label_mean[roster_index]
            colours[:, column] = value

        elif kind == "residual_base":
            # Already in the TARGET's space, not the lab standardizer's - that is the whole point of
            # the block - so it inverts with the target statistics, not label_mean/label_scale. The
            # result is the base prediction in original units, which is the number a reader
            # comparing the correction against what it corrected actually wants.
            block = parts[group["part"]].detach().cpu().numpy().astype(np.float64)
            position = group["columns"][0]
            value = block[:, position]
            if position < target_mean.size and position < target_scale.size:
                value = value * target_scale[position] + target_mean[position]
            if bool(getattr(model, "targets_are_log1p", False)):
                value = np.expm1(value / 10.0)
            colours[:, column] = value

    return colours


def lightning_shap_results(*, config, model, bundle, target: str) -> list[ShapResult]:
    """Explain a trained deep-learning model, for every target it predicts.

    Parameters
    ----------
    model : SoilCNNLightningModule
        The trained model.
    datamodule : SoilSequenceDataModule
        Its data, used for the points to explain and their background.
    target : str
        The :term:`target group`, used to name outputs when the model carries no target names.
    config : Config, optional
        Read for ``explain.max_samples`` and the background size.

    Returns
    -------
    list of ShapResult
        One per target.
    """
    import shap
    import torch
    from torch import nn

    if not hasattr(model, "explanation_parts") or not hasattr(model, "forward_from_parts"):
        raise TypeError(
            f"{type(model).__name__} has no attribution seam: SoilCNNLightningModule implements "
            "explanation_parts/forward_from_parts, and a model explained here must too."
        )

    max_samples = int(getattr(config, "EXPLAIN_MAX_SAMPLES", 500))
    background_samples = int(getattr(config, "EXPLAIN_BACKGROUND_SAMPLES", 100))

    was_training = model.training
    model.eval()
    try:
        batches = _collect_batches(bundle.datamodule, max_samples + background_samples)
        if not batches:
            return []

        per_batch_parts = []
        groups: list[dict] = []
        categorical_chunks = []
        with torch.no_grad():
            for batch in batches:
                parts, groups = model.explanation_parts(batch)
                per_batch_parts.append([part.detach() for part in parts])
                codes = batch.get("x_categorical")
                if codes is not None:
                    categorical_chunks.append(codes.detach().cpu().numpy())

        parts = _concatenate_parts(per_batch_parts, torch)
        categorical_codes = (
            np.concatenate(categorical_chunks, axis=0)[: int(parts[0].shape[0])] if categorical_chunks else None
        )

        total = int(parts[0].shape[0])
        if total < 2 or not groups:
            return []

        # Background first, explained rows after it, so the two never overlap: a point explained
        # against itself contributes a zero-length interval to the expected-gradients integral.
        split = min(max(1, background_samples), total - 1)
        background = [part[:split] for part in parts]
        explain = [part[split : split + max_samples] for part in parts]
        if int(explain[0].shape[0]) == 0:
            return []
        # The colour values describe the EXPLAINED rows, so the codes follow the same slice. Using
        # the full array here silently misaligns every categorical colour by `split` rows.
        if categorical_codes is not None:
            categorical_codes = categorical_codes[split : split + max_samples]

        class _PartsModule(nn.Module):
            """Presents the model's explanation entry point as an ordinary network."""

            def __init__(self, wrapped):
                """Hold the model this wraps."""
                super().__init__()
                self.wrapped = wrapped

            def forward(self, *inputs):
                """Predict from the varied inputs alone."""
                return self.wrapped.forward_from_parts(list(inputs))

        explainer = shap.GradientExplainer(_PartsModule(model), background)
        raw_values = explainer.shap_values(explain)
        if not isinstance(raw_values, list):
            raw_values = [raw_values]

        state = model.get_preprocessing_state() if hasattr(model, "get_preprocessing_state") else {}
        colours = _colour_values(model, explain, groups, categorical_codes, state, torch)

        feature_names = [str(group["name"]) for group in groups]
        blocks = [group["modality"] if group["kind"] == "temporal" else group["kind"] for group in groups]

        target_names = list(getattr(model, "target_names", None) or []) or [target]
        n_outputs = _output_count(raw_values, explain)

        results = []
        for output_index in range(n_outputs):
            per_part = [_slice_output(array, part, output_index) for array, part in zip(raw_values, explain)]
            values = _group_values(per_part, groups)
            results.append(
                ShapResult(
                    values=values,
                    data=colours,
                    feature_names=feature_names,
                    blocks=blocks,
                    target_name=str(target_names[output_index] if output_index < len(target_names) else output_index),
                    # Attribution is taken on forward(), which is what the loss sees, so these are
                    # contributions in STANDARDIZED LOG1P space - not in the target's own units.
                    # Only predict_step inverts the transform, and it is downstream of the seam.
                    output_space=STANDARDIZED_LOG1P,
                )
            )
        return results
    finally:
        if was_training:
            model.train()


def _output_count(raw_values: list, parts: list) -> int:
    """How many targets the contributions cover.

    Worked out by comparing against the inputs' shape rather than by counting dimensions: the library
    adds a trailing axis only for a model with several outputs.
    """
    reference = np.asarray(raw_values[0])
    part_shape = tuple(parts[0].shape)
    if reference.shape == part_shape:
        return 1
    if reference.ndim == len(part_shape) + 1:
        return int(reference.shape[-1])
    return 1


def _slice_output(array: Any, part, output_index: int) -> np.ndarray:
    """One target's contributions."""
    values = np.asarray(array, dtype=np.float64)
    if values.shape == tuple(part.shape):
        return values
    if values.ndim == len(tuple(part.shape)) + 1:
        return values[..., output_index]
    return values
