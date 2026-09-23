"""Prepare the data once, before the study starts.

Preparing it per trial would cost more than the training it feeds.
"""

from __future__ import annotations

from typing import Any, Mapping

from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory

PAYLOAD_KEYS = {"sequence": "sequence_bundle"}


def build_lightning_input(
    entry: str,
    spec: Mapping[str, Any],
    config: Any,
    split_data: Mapping[str, Any],
    *,
    logger: Any = None,
    data_manager: Any = None,
) -> dict[str, Any]:
    """The shared split and the prepared data every trial trains on."""
    data = dict(split_data)
    input_kind = LightningConfigFactory._input_kind(spec)
    payload_key = PAYLOAD_KEYS.get(input_kind)
    if payload_key is None:
        raise ValueError(
            f"Registry entry {entry!r} declares input_kind {input_kind!r}; expected one of "
            f"{', '.join(sorted(PAYLOAD_KEYS))}."
        )
    if data.get(payload_key) is not None:
        return data

    factory = LightningConfigFactory({entry: dict(spec)}, config, logger=logger, data_manager=data_manager)
    data[payload_key] = factory._build_sequence_bundle(spec)
    return data
