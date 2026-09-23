"""Tiny sequence bundles, CNNs and builder configs.

Every model here is untrained: freshly initialised under a fixed seed, which is all the serving,
packaging and checkpoint tests need - they compare two code paths on the same weights.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule

S2 = {"s2": ["S2_B02", "S2_B08"]}


def sequence_bundle(
    *,
    n_points: int = 16,
    static: Sequence[str] = ("clay_pct", "ph"),
    modalities: Mapping[str, Sequence[str]] = S2,
    target: str = "organic_matter_pct",
    categorical: bool = True,
    lab_roster: Sequence[str] | None = None,
    coord_names: Sequence[str] | None = None,
    observations: int | tuple[int, int] = (3, 7),
    seed: int = 0,
) -> SoilSequenceBundle:
    """A bundle as the builder would produce it, with every optional part switchable.

    ``observations`` is either a fixed count per point or a ``(low, high)`` range, high exclusive,
    for ragged sequences. Coordinates are realistic Morocco degrees and deliberately not round, so
    a lossy float32 round trip shows.
    """
    generator = np.random.default_rng(seed)

    def count() -> int:
        return observations if isinstance(observations, int) else int(generator.integers(*observations))

    fields: dict = dict(
        point_ids=[f"p{index}" for index in range(n_points)],
        static_features=generator.normal(20, 5, (n_points, len(static))).astype(np.float32),
        static_feature_names=list(static),
    )
    if categorical:
        fields["static_categoricals"] = np.asarray(
            [[generator.choice(["sandy", "loam"])] for _ in range(n_points)], dtype=object
        )
        fields["categorical_feature_names"] = ["texture"]
    else:
        fields["static_categoricals"] = np.empty((n_points, 0), dtype=object)
        fields["categorical_feature_names"] = []
    if coord_names:
        fields["coords"] = np.column_stack(
            [generator.uniform(28.60413, 35.65917, n_points), generator.uniform(-10.00382, -1.93641, n_points)]
        )
        fields["coord_names"] = list(coord_names)
    fields["targets"] = generator.normal(3, 1, (n_points, 1)).astype(np.float32)
    fields["target_names"] = [target]
    if lab_roster:
        fields["label_features"] = generator.normal(5, 1, (n_points, len(lab_roster))).astype(np.float32)
        fields["label_feature_names"] = list(lab_roster)

    sequences = {
        name: [generator.normal(0.2, 0.05, (count(), len(columns))).astype(np.float32) for _ in range(n_points)]
        for name, columns in modalities.items()
    }
    times = {
        name: [2020.0 + np.sort(generator.random(values.shape[0])) * 2.0 for values in sequences[name]]
        for name in modalities
    }
    return SoilSequenceBundle(
        **fields,
        sequences=sequences,
        sequence_times=times,
        modality_columns={name: list(columns) for name, columns in modalities.items()},
        temporal_enabled=True,
    )


def tiny_cnn(
    bundle: SoilSequenceBundle,
    *,
    auxiliary: Sequence[str] | None = None,
    seed: int = 0,
    **overrides,
) -> tuple[SoilCNNLightningModule, SoilSequenceDataModule]:
    """A small, untrained soil_cnn fitted to ``bundle``'s shapes, with its preprocessing attached."""
    torch.manual_seed(seed)
    datamodule = SoilSequenceDataModule(sequence_bundle=bundle, batch_size=8, val_size=0.25, test_size=0.25, seed=42)
    datamodule.setup("fit")

    kwargs = dict(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        target_names=datamodule.target_names,
        categorical_cardinalities=datamodule.categorical_cardinalities,
        categorical_vocabularies=datamodule.categorical_vocabularies,
        categorical_feature_names=datamodule.categorical_feature_names,
        modality_dims=datamodule.modality_dims,
        temporal_enabled=True,
        grid_years=datamodule.grid_years,
        coord_dim=datamodule.coord_dim,
        auxiliary_label_columns=list(auxiliary) if auxiliary else None,
        auxiliary_available_names=datamodule.label_feature_names if auxiliary else None,
        static_hidden_dims=[6],
        head_hidden_dims=[6],
        cnn_hidden_dims=[4],
        modality_embed_dim=4,
        dropout=0.0,
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
    )
    kwargs.update(overrides)
    model = SoilCNNLightningModule(**kwargs)
    model.attach_preprocessing_state(datamodule.preprocessing_state())
    model.eval()
    return model, datamodule


def sequence_builder_config(
    tmp_path: Path,
    static_path: Path,
    timeseries_path: Path,
    *,
    targets_path: Path | None = None,
    **overrides,
) -> SimpleNamespace:
    """The config SoilSequenceBuilder reads, for CSVs a test wrote itself.

    Only the keys every builder test shares; a test adds the rest (LABEL_COLUMNS, CARRY_LABEL_COLUMNS,
    the coordinate and context switches, ...) as overrides, so nothing it did not ask for changes.
    """
    targets_path = targets_path or static_path
    config = SimpleNamespace(
        DATA_FOLDER=str(tmp_path),
        DATA_FILE="static.csv",
        STATIC_CSV_PATH=str(static_path),
        TIMESERIES_CSV_PATH=str(timeseries_path),
        POINT_ID_COLUMN="point_id",
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
        TIME_COLUMN="obs_date",
        TEMPORAL_FEATURES_ENABLED=True,
        TEMPORAL_FEATURES={"enabled": True, "time_column": "obs_date"},
        MODALITY_PREFIX_MAP={"s2": "S2_"},
        S1_COLUMNS=[],
        S2_COLUMNS=[],
        MODIS_COLUMNS=[],
        TARGET_COLUMNS=["target_a"],
        LABEL_COLUMNS=["target_a"],
        PREDICTOR_COLUMNS=[],
        IGNORED_COLUMNS=["point_id", "lat", "lon"],
        ELIMINATED_FEATURES=["point_id", "lat", "lon"],
        CATEGORICAL_FEATURES=[],
        EXCLUDE_CATEGORICAL=False,
        EXISTING_HS_FEATURES={"enabled": False},
        RANDOM_SEED=42,
        TEST_SIZE=0.25,
        DATA_INDEX_MANIFEST_PATH=None,
        STATIC_SOURCE=None,
        TARGETS_SOURCE=None,
        TIMESERIES_SOURCE=None,
        STATIC_FEATURES_FOLDER=None,
        TARGETS_FOLDER=None,
        TIMESERIES_FOLDER=None,
        TARGETS_FILE=Path(targets_path).name,
        TARGETS_CSV_PATH=str(targets_path),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def correlated_bundle(correlation: float = 0.8, n_points: int = 200, seed: int = 0) -> SoilSequenceBundle:
    """Two static-only targets with a known correlation, for the covariance and structural losses."""
    generator = np.random.default_rng(seed)
    first = generator.standard_normal(n_points)
    second = correlation * first + np.sqrt(1.0 - correlation**2) * generator.standard_normal(n_points)
    return SoilSequenceBundle(
        point_ids=list(range(n_points)),
        static_features=generator.standard_normal((n_points, 2)).astype(np.float32),
        static_feature_names=["f1", "f2"],
        targets=np.column_stack([first, second]).astype(np.float32),
        target_names=["target_a", "target_b"],
    )


def eval_frame(n_rows: int = 60, with_uncertainty: bool = True) -> pd.DataFrame:
    """A single-target eval frame, with the ensemble's sigma and a 2-sigma interval by default."""
    rng = np.random.default_rng(0)
    observed = rng.normal(loc=20.0, scale=5.0, size=n_rows)
    predicted = observed + rng.normal(scale=2.0, size=n_rows)
    frame = pd.DataFrame({"target": observed, "prediction": predicted})
    if with_uncertainty:
        sigma = np.abs(rng.normal(loc=2.0, scale=0.5, size=n_rows))
        frame["prediction_std"] = sigma
        frame["prediction_lower"] = predicted - 2.0 * sigma
        frame["prediction_upper"] = predicted + 2.0 * sigma
    return frame
