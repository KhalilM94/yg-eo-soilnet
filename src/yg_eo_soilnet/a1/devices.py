"""The processor each A1 stage fits on, and the checks that it is the one asked for.

XGBoost asked for ``device: cuda`` on a machine where it finds no usable graphics card does not
stop: it warns ("Device is changed from GPU to CPU") and carries on on the processor, so a run
believed to be on the graphics card would silently be on the processor, hours slower. Before
fitting anything, ``run_a1.py`` therefore fits a tiny model on the card (:func:`check_cuda`) and
stops with a clear message if XGBoost was built without CUDA or fell back.

The device of each stage is recorded in the run folder's manifest, in the MLflow sub-run of each
plan and in every predictions file (``device_stage1``, ``device_stage2``), so two runs fitted on
different processors are never compared unknowingly (:func:`check_same_devices`).
"""

from __future__ import annotations

import warnings
from typing import Mapping, Optional

import numpy as np

#: The device recorded for a stage of a run made before devices were recorded.
UNKNOWN_DEVICE = "unknown"


class DeviceUnavailable(RuntimeError):
    """A stage asks for the graphics card, and XGBoost cannot use it."""


class DeviceMismatch(ValueError):
    """Runs to be compared were fitted on different processors."""


def is_cuda(device: Optional[str]) -> bool:
    """Whether a ``device`` setting names a graphics card (``cuda``, ``cuda:0``, ``gpu``)."""
    return str(device or "").lower().split(":")[0] in ("cuda", "gpu")


def check_cuda(device: str = "cuda") -> None:
    """Fit a tiny XGBoost model on ``device``; raise if it cannot run there.

    Raises
    ------
    DeviceUnavailable
        If XGBoost was built without CUDA, or fitted on the processor instead (its "Device is
        changed from GPU to CPU" fallback), or failed on the card.
    """
    import xgboost

    if not xgboost.build_info().get("USE_CUDA"):
        raise DeviceUnavailable(
            f"A stage asks for device {device!r}, but this XGBoost ({xgboost.__version__}) was built without CUDA. "
            "Use the environment with the CUDA build (pixi run -e dev), or run with --device cpu."
        )
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(64, 3)), rng.normal(size=64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = xgboost.XGBRegressor(n_estimators=2, max_depth=2, tree_method="hist", device=device)
            model.fit(X, y)
        except Exception as error:  # an XGBoostError, or a CUDA error under it
            raise DeviceUnavailable(
                f"XGBoost could not fit on {device!r}: {error}. Run with --device cpu, or check the graphics card "
                "(nvidia-smi)."
            ) from error
    fallback = [str(item.message) for item in caught if _is_fallback(str(item.message))]
    if fallback:
        raise DeviceUnavailable(
            f"XGBoost was asked for {device!r} but fell back to the processor ({fallback[0].strip()}). No usable "
            "graphics card was found; run with --device cpu, or check the card (nvidia-smi)."
        )


def _is_fallback(message: str) -> bool:
    """Whether an XGBoost warning says it left the graphics card for the processor."""
    lowered = message.lower()
    return "device is changed from gpu to cpu" in lowered or ("gpu" in lowered and "not available" in lowered)


def check_same_devices(devices: Mapping[str, Mapping[str, str]], *, allow_mixed: bool = False) -> None:
    """Refuse to compare runs fitted on different processors, unless ``allow_mixed``.

    Parameters
    ----------
    devices : mapping
        Per run (a name), its ``{"stage1": device, "stage2": device}``, as the run's manifest
        records them.
    allow_mixed : bool, default False
        Compare them anyway (``compare.py --allow-mixed-devices``): the trees can differ in their
        last digits between the processor and the graphics card.

    Raises
    ------
    DeviceMismatch
        If two runs differ in a stage's device and ``allow_mixed`` is not set.
    """
    if allow_mixed or len(devices) < 2:
        return
    for stage in ("stage1", "stage2"):
        seen = {name: str((recorded or {}).get(stage, UNKNOWN_DEVICE)) for name, recorded in devices.items()}
        if len(set(seen.values())) > 1:
            listed = ", ".join(f"{name}: {device}" for name, device in seen.items())
            raise DeviceMismatch(
                f"The runs were fitted on different processors in {stage} ({listed}). Their scores can differ for "
                "that reason alone; pass --allow-mixed-devices to compare them anyway."
            )
