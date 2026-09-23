"""The calendar-grid CNN's shared fixtures: a hand-built batch, a builder-built bundle, and the
residual helpers that read a zeroed head back to its base offset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder, to_decimal_year

from tests.support.builders import sequence_builder_config

LABEL_NAMES = ["lab_a", "lab_b", "lab_c"]


def cnn_batch(batch_size=4, length=24, channels=3, seed=0, start="2019-01-01", months_step=1, year_offset=0, labels=3):
    generator = torch.Generator().manual_seed(seed)
    dates = pd.date_range(start, periods=length, freq=f"{months_step}MS")
    if year_offset:
        dates = dates + pd.DateOffset(years=year_offset)
    times = torch.as_tensor(to_decimal_year(pd.Series(dates))).unsqueeze(0).repeat(batch_size, 1)
    return {
        "x_static": torch.randn(batch_size, 5, generator=generator),
        "y": torch.randn(batch_size, 1, generator=generator),
        "sequences": {"m": torch.randn(batch_size, length, channels, generator=generator)},
        "sequence_mask": {"m": torch.ones(batch_size, length, dtype=torch.bool)},
        "sequence_time": {"m": times},
        "sequence_validity": {"m": torch.ones(batch_size, length, channels, dtype=torch.bool)},
        # Appended last so the generator draws above keep their values and every existing
        # expectation in this file still holds.
        "x_labels": torch.randn(batch_size, labels, generator=generator),
        "x_label_validity": torch.ones(batch_size, labels, dtype=torch.bool),
    }


# Roster stats for LABEL_NAMES. Deliberately not 0/1: an identity standardizer would hide a missing
# de-standardization hop, which is exactly the bug this file exists to catch.
LABEL_MEAN = [2.0, 30.0, 7.0]
LABEL_SCALE = [0.5, 4.0, 2.0]


def zero_head(module) -> None:
    """Silence the head so forward() returns the offset alone.

    Zeroing the weights is not enough - every Linear has a bias, and the last one is what the head
    would otherwise contribute.
    """
    for parameter in module.output_head.parameters():
        nn.init.zeros_(parameter)


def expected_base(batch, *, index=1, target_mean=None, target_scale=None, log1p=False):
    """The base column mapped by hand, the long way round, as the module should compute it."""
    base = batch["x_labels"][:, index].double() * LABEL_SCALE[index] + LABEL_MEAN[index]
    if log1p:
        base = 10.0 * torch.log1p(base.clamp_min(0.0))
    if target_mean is not None:
        base = (base - target_mean) / target_scale
    return base


def write_sequence_csvs(tmp_path: Path, dates_by_point, split: bool = False):
    """Write the fixture as one joint file, or as separate static and targets files.

    `split` is not decoration: the targets join used to carry only the ACTIVE targets, so the two
    layouts disagreed about which lab columns a model could select from the very same declaration.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    point_ids = sorted(dates_by_point)
    static_df = pd.DataFrame(
        {
            "point_id": point_ids,
            "lat": [0.1 * index for index in range(len(point_ids))],
            "lon": [0.1 * index for index in range(len(point_ids))],
            "target_a": [1.0 + index for index in range(len(point_ids))],
            "static_1": [10.0 + index for index in range(len(point_ids))],
            # Measured lab values: named in LABEL_COLUMNS so they are never features, but carried
            # on the bundle so a model may opt into them. lab_sparse is deliberately incomplete.
            "lab_dense": [100.0 + 10.0 * index for index in range(len(point_ids))],
            "lab_sparse": [np.nan if index % 2 else 5.0 + index for index in range(len(point_ids))],
        }
    )
    targets_df = static_df
    if split:
        # The lab values live with the targets, which is where a real targets file keeps them.
        lab_columns = ["point_id", "target_a", "lab_dense", "lab_sparse"]
        targets_df = static_df[lab_columns]
        static_df = static_df.drop(columns=[column for column in lab_columns if column != "point_id"])

    rows = []
    for point_id, dates in dates_by_point.items():
        for index, date in enumerate(dates):
            rows.append(
                {
                    "point_id": point_id,
                    "obs_date": date,
                    "S2_b2": 1.0 + index,
                    "S2_b3": np.nan if (point_id == 2 and index == 0) else 2.0 + index,
                }
            )
    static_path, timeseries_path = tmp_path / "static.csv", tmp_path / "ts.csv"
    targets_path = tmp_path / "targets.csv" if split else static_path
    static_df.to_csv(static_path, index=False)
    if split:
        targets_df.to_csv(targets_path, index=False)
    pd.DataFrame(rows).to_csv(timeseries_path, index=False)
    return static_path, timeseries_path, targets_path


def built_sequence_bundle(tmp_path: Path, logger, dates_by_point, carry_labels: bool = True, split: bool = False):
    static_path, timeseries_path, targets_path = write_sequence_csvs(tmp_path, dates_by_point, split=split)
    config = sequence_builder_config(
        tmp_path,
        static_path,
        timeseries_path,
        targets_path=targets_path,
        LABEL_COLUMNS=["target_a", "lab_dense", "lab_sparse"],
        CARRY_LABEL_COLUMNS=carry_labels,
    )
    return SoilSequenceBuilder(config, logger, DataManager(config, logger)).build()


def detach_logging(module):
    """Silence LightningModule.log, which warns when there is no Trainer attached.

    For tests that exercise the step logic directly rather than through a Trainer: the suite runs
    with filterwarnings = ["error"], so the warning would fail the test for the wrong reason.
    """
    module.log = lambda *args, **kwargs: None
    return module
