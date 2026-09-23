"""`soil_cnn`: the project's deep-learning model, and every switch that shapes it."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from yg_eo_soilnet.models.lightningmodules._regression_base import (
    SoilRegressionLightningBase,
    as_float_list,
    as_float_matrix,
    batch_get,
)
from yg_eo_soilnet.models.lightningmodules.mlp import build_mlp_stack
from yg_eo_soilnet.models.lightningmodules.spatial_encoders import HarmonicPositionEncoder
from yg_eo_soilnet.models.lightningmodules.tabular_encoders import TabularStaticEncoder
from yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders import (
    AnnualGrid2DEncoder,
    AttentionFusion,
    CalendarGridRasterizer,
    ConcatGatedFusion,
    DilatedTempCNNEncoder,
)

logger = logging.getLogger(__name__)

#: The ways the branches can be combined; see :term:`fusion`.
FUSION_TYPES = ("gated", "attention")
#: With attention fusion, whether the covariates arrive as one token or one per column.
STATIC_TOKEN_MODES = ("summary", "per_feature")


def _as_width_list(value: Any) -> list[int]:
    """Read a width setting as a list: ``64`` and ``[64]`` both mean one 64-wide layer."""
    if isinstance(value, (list, tuple)):
        return [int(width) for width in value]
    return [int(value)]


class SoilCNNLightningModule(SoilRegressionLightningBase):
    """The project's deep-learning model: covariates, time series and optionally location.

    One branch summarizes the covariates, one summarizes each :term:`data source`\'s time series by
    laying it out as a :term:`calendar grid` and scanning it, and one optionally reads the point's
    location. A :term:`fusion` step combines the summaries and a small stack of layers predicts
    every target::

        covariates  -------------------> covariate branch ----.
        (numbers and categories)                               \\
        time series -> calendar grid -> one CNN per source ------>  fusion  --.
        (years x months, per source)                           /   gated or  |
        coordinates -------------------> location branch ----'    attention  v
                                                                          prediction head --> targets
        measured lab values (optional) --------------------------------------'
        an earlier model's prediction (optional) ----------------------------'

    Any number of data sources of any width works; it follows from ``modality_dims``.

    **Switches.** Three settings change what is built, and every combination works:

    ``fusion``
        ``gated`` scales each value of each branch by a learned weight; ``attention`` lets the
        branches re-read one another. See
        :class:`~yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders.ConcatGatedFusion` and
        :class:`~yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders.AttentionFusion`.
    ``auxiliary_enabled``
        Feeds measured lab values named in ``auxiliary_label_columns`` straight to the prediction
        head - an :term:`auxiliary lab input`. Only honest when those values will really be measured
        for the new points too; naming a target raises rather than being ignored.
    ``residual_enabled``
        Makes the model predict a *correction* to an existing prediction instead of the value
        itself - a :term:`residual base`. ``residual_base_columns`` names, per target, the column
        holding that prediction, usually one exported by an earlier run and joined back onto the
        data. The base is converted into the units the model trains in and added to what the head
        predicts, and it is also fed to the head as an input, so the correction is not made blind.

        **This is only sound if those base predictions were made without seeing this run's test
        points.** Nothing here can check that; a warning naming each base column is logged when the
        model is built.

    Parameters
    ----------
    static_dim : int
        How many numeric covariates come in, the measured-or-filled flags included. Category columns
        arrive separately.
    target_dim : int
        How many targets this model predicts.
    target_names : sequence of str, optional
        Their names, used to name the per-target scores.
    fitted_target_names : sequence of str, optional
        Every target the *run* fits, which with one model per target is more than this model
        predicts. Used to refuse an auxiliary or base column that is a target of the run.
    categorical_cardinalities : sequence of int, optional
        How many codes each category column has.
    categorical_vocabularies : sequence of sequence of str, optional
        Their labels in code order, saved so the model can read new data on its own.
    categorical_feature_names : sequence of str, optional
        The category column names.
    embedding_dims : optional
        Embedding widths; see
        :func:`~yg_eo_soilnet.models.lightningmodules.tabular_encoders.resolve_embedding_dims`.
    embedding_dropout : float, default 0.0
        Dropout on the joined category embeddings.
    embedding_max_dim : int, default 50
        Cap on an automatically chosen embedding width.
    continuous_norm : {"none", "batch", "layer"}, default "none"
        Extra normalization of the numeric covariates, which are already standardized.
    modality_dims : mapping of str to int, optional
        How many channels each data source has: ``{"s2": 5, "clim": 2}``.
    temporal_enabled : bool, default True
        Read the time series at all. With no data sources the model uses the covariates only.
    grid_years : int, optional
        How many years the calendar grid spans. Left out, each batch uses its own longest history.
    temporal_encoder : {"dilated_tempcnn", "annual_grid2d"}, default "dilated_tempcnn"
        Which encoder reads the grid.
    cnn_hidden_dims : int, sequence or mapping, default (32,)
        Layer widths of that encoder. A mapping gives different widths per data source, and must
        then name them all.
    modality_embed_dim : int or mapping, default 32
        How wide each data source's summary is.
    cnn_norm : {"batch", "group", "none"}, default "batch"
        Normalization inside the encoder. ``"group"`` suits grids with many empty cells.
    pool : {"masked_avg", "avg"}, default "masked_avg"
        Whether the encoder's final average covers only the months that hold a reading.
    month_positional : bool, default True
        Tell the encoder which column is which month.
    use_validity_channels : bool, default True
        Carry the measured-or-filled flags for the readings.
    auxiliary_label_columns : sequence of str, optional
        Measured lab columns to feed to the prediction head.
    auxiliary_available_names : sequence of str, optional
        Every lab column the data carries, which the names above are resolved against.
    auxiliary_validity_channels : bool, default True
        Also tell the model which of those values were measured rather than filled in.
    auxiliary_hidden_dims : sequence of int, optional
        Layer widths summarizing them. Empty passes them straight through.
    auxiliary_dropout : float, default 0.0
        Dropout in that block.
    coord_dim : int, default 0
        How many coordinate columns the data carries: 2 with ``USE_HARMONIC_COORDS``, otherwise 0,
        which leaves the location branch out entirely.
    harmonic_num_frequencies : int, default 6
        How finely the location branch resolves position; see
        :class:`~yg_eo_soilnet.models.lightningmodules.spatial_encoders.HarmonicPositionEncoder`.
    harmonic_include_input : bool, default True
        Also pass the coordinates through unchanged.
    harmonic_hidden_dims : sequence of int, optional
        Layer widths summarizing the location branch.
    harmonic_dropout : float, default 0.0
        Dropout in that branch.
    static_hidden_dims : sequence of int, default (64,)
        Layer widths of the covariate branch.
    head_hidden_dims : sequence of int, default (128, 64)
        Layer widths of the prediction head.
    head_norm_final : bool, default False
        Normalize the head's last layer. Off by default: it would throw away the size of the values
        the prediction is read from.
    dropout : float, default 0.1
        Dropout through the model.
    loss_name : str, default "mse"
        What to minimize; see
        :func:`~yg_eo_soilnet.models.lightningmodules.losses.build_loss_fn`.
    huber_delta : float, default 1.0
        Where ``huber`` and ``smooth_l1`` switch from squared to absolute error.
    loss_base : str, default "mse"
        The per-target loss a structural loss adds its penalty to.
    loss_lambda : float, default 0.1
        How heavily that penalty counts.
    loss_shrinkage : float, default 0.05
        For ``mahalanobis``.
    loss_min_batch : int, default 16
        For ``correlation_penalty``.
    cosine_space : {"original", "standardized"}, default "original"
        For ``cosine``.
    target_covariance : array-like, optional
        How the training targets vary together; supplied by the datamodule.
    learning_rate : float, default 0.001
        How large a step training takes.
    optimizer_name : {"adamw", "adam"}, default "adamw"
        Which optimizer.
    weight_decay : float, default 0.0001
        How strongly large weights are penalized.
    scheduler_type : str, default "plateau"
        ``"plateau"`` lowers the learning rate when ``val_loss`` stops improving.
    scheduler_factor : float, default 0.5
        What the learning rate is multiplied by then.
    scheduler_patience : int, default 5
        Epochs without improvement to wait first.
    scheduler_min_lr : float, default 1e-06
        The lowest the learning rate may go.
    scheduler_monitor : str, default "val_loss"
        Which score the schedule watches.
    target_mean, target_scale : array-like, optional
        The target standardization statistics, from the datamodule.
    target_transform : {None, "log1p"}, optional
        Whether the targets were log-transformed.
    predict_variance : bool, default False
        Predict a spread alongside each value; see :term:`variance head`.
    beta_nll : float, default 0.5
        How that head balances fitting the values against fitting their spread.
    auxiliary_enabled : bool, default True
        Read ``auxiliary_label_columns`` at all. False ignores the list without clearing it, so
        switching back needs no second edit.
    fusion : {"gated", "attention"}, default "gated"
        How the branch summaries are combined.
    attention_static_tokens : {"summary", "per_feature"}, default "summary"
        With attention fusion: whether the covariates arrive as one token or one per column. With
        one per column, ``static_hidden_dims`` has no effect.
    attention_d_model : int, default 64
        Width every token is projected to.
    attention_nhead : int, default 4
        How many attention heads; must divide ``attention_d_model``.
    attention_num_layers : int, default 2
        How many transformer layers.
    attention_ff_multiplier : int, default 2
        Width of each layer's internal step, as a multiple of ``attention_d_model``.
    attention_dropout : float, default 0.1
        Dropout inside the transformer.
    attention_readout : {"cls", "mean", "flatten"}, default "cls"
        What the fusion passes on.
    residual_enabled : bool, default False
        Predict a correction to an existing prediction instead of the value.
    residual_base_columns : mapping of str to str, optional
        Per target, the lab column holding that existing prediction. Required when
        ``residual_enabled``.
    residual_base_hidden_dims : sequence of int, optional
        Layer widths of the block reading the base. Empty passes it straight through.
    residual_base_dropout : float, default 0.0
        Dropout in that block.
    residual_base_validity_channels : bool, default True
        Also tell the model where the base was actually available.
    residual_base_max_missing : float, default 0.05
        Refuse to train if a base column is missing on more than this share of points: a missing
        base is filled in, and the model would then be correcting a constant.
    auxiliary_label_mean, auxiliary_label_scale : array-like, optional
        The lab-value standardization statistics, from the datamodule. Needed to convert a base back
        into the target's units.

    Raises
    ------
    ValueError
        If a setting holds an unknown value, an auxiliary or base column names a target of the run,
        a named column is not carried with the data, or ``residual_enabled`` is set without a base
        for every target.

    Examples
    --------
    >>> model = SoilCNNLightningModule(                       # doctest: +SKIP
    ...     static_dim=12, target_dim=3, modality_dims={"s2": 5, "clim": 2},
    ...     target_names=["organic_matter_g_kg", "clay_pct", "ph_water"])
    >>> model.output_head[-1].out_features                    # doctest: +SKIP
    3
    """

    # Also set on the class, not only on each model: a model saved to MLflow restores its own
    # settings but not its class's, so an older saved model reads these defaults from the class it
    # was saved as - which is what the two older names below exist to record.
    fusion_type = "gated"
    attention_static_tokens = "summary"
    auxiliary_enabled = True
    residual_enabled = False

    def __init__(
        self,
        static_dim: int,
        target_dim: int,
        target_names: Optional[Sequence[str]] = None,
        fitted_target_names: Optional[Sequence[str]] = None,
        categorical_cardinalities: Optional[Sequence[int]] = None,
        categorical_vocabularies: Optional[Sequence[Sequence[str]]] = None,
        categorical_feature_names: Optional[Sequence[str]] = None,
        embedding_dims: Any = None,
        embedding_dropout: float = 0.0,
        embedding_max_dim: int = 50,
        continuous_norm: str = "none",
        modality_dims: Optional[Mapping[str, int]] = None,
        temporal_enabled: bool = True,
        grid_years: Optional[int] = None,
        temporal_encoder: str = "dilated_tempcnn",
        cnn_hidden_dims: Any = (32,),
        modality_embed_dim: Any = 32,
        cnn_norm: str = "batch",
        pool: str = "masked_avg",
        month_positional: bool = True,
        use_validity_channels: bool = True,
        auxiliary_label_columns: Optional[Sequence[str]] = None,
        auxiliary_available_names: Optional[Sequence[str]] = None,
        auxiliary_validity_channels: bool = True,
        auxiliary_hidden_dims: Optional[Sequence[int]] = None,
        auxiliary_dropout: float = 0.0,
        # The location branch; left out entirely at coord_dim=0, which is what the datamodule
        # reports unless USE_HARMONIC_COORDS is on.
        coord_dim: int = 0,
        harmonic_num_frequencies: int = 6,
        harmonic_include_input: bool = True,
        harmonic_hidden_dims: Optional[Sequence[int]] = None,
        harmonic_dropout: float = 0.0,
        static_hidden_dims: Sequence[int] = (64,),
        head_hidden_dims: Sequence[int] = (128, 64),
        head_norm_final: bool = False,
        dropout: float = 0.1,
        loss_name: str = "mse",
        huber_delta: float = 1.0,
        # Read only by the losses that work across targets; the datamodule supplies the covariance.
        loss_base: str = "mse",
        loss_lambda: float = 0.1,
        loss_shrinkage: float = 0.05,
        loss_min_batch: int = 16,
        cosine_space: str = "original",
        target_covariance: Optional[Any] = None,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adamw",
        weight_decay: float = 1e-4,
        scheduler_type: str = "plateau",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        scheduler_min_lr: float = 1e-6,
        scheduler_monitor: str = "val_loss",
        target_mean: Optional[Any] = None,
        target_scale: Optional[Any] = None,
        target_transform: Optional[str] = None,
        # Predict a spread beside each value; set from uncertainty.heteroscedastic.
        predict_variance: bool = False,
        beta_nll: float = 0.5,
        *,
        # False ignores the column list without clearing it, so switching back needs no second
        # edit. True by default, which is how models saved before the switch behaved.
        auxiliary_enabled: bool = True,
        # The attention_* settings are read only with fusion: attention.
        fusion: str = "gated",
        attention_static_tokens: str = "summary",
        attention_d_model: int = 64,
        attention_nhead: int = 4,
        attention_num_layers: int = 2,
        attention_ff_multiplier: int = 2,
        attention_dropout: float = 0.1,
        attention_readout: str = "cls",
        # The residual_* settings are read only with residual_enabled.
        residual_enabled: bool = False,
        residual_base_columns: Optional[Mapping[str, str]] = None,
        residual_base_hidden_dims: Optional[Sequence[int]] = None,
        residual_base_dropout: float = 0.0,
        residual_base_validity_channels: bool = True,
        residual_base_max_missing: float = 0.05,
        # The lab-value statistics, which are what convert a base back into the target's units.
        auxiliary_label_mean: Optional[Any] = None,
        auxiliary_label_scale: Optional[Any] = None,
    ):
        super().__init__()
        # Converted to plain values before the settings are recorded: a checkpoint holding arrays
        # cannot be read back safely.
        target_mean = as_float_list(target_mean)
        target_scale = as_float_list(target_scale)
        target_covariance = as_float_matrix(target_covariance)
        head_hidden_dims = [int(width) for width in head_hidden_dims]
        static_hidden_dims = [int(width) for width in static_hidden_dims]
        # The category numbering lives with the model, so a saved model can read new data.
        categorical_cardinalities = [int(value) for value in (categorical_cardinalities or [])]
        categorical_vocabularies = [
            [str(category) for category in vocabulary] for vocabulary in (categorical_vocabularies or [])
        ]
        categorical_feature_names = [str(name) for name in (categorical_feature_names or [])]
        # Both the chosen columns and the full list they were chosen from are recorded, so the
        # model knows which lab columns it expects rather than trusting their position.
        auxiliary_label_columns = [str(name) for name in (auxiliary_label_columns or [])]
        auxiliary_available_names = [str(name) for name in (auxiliary_available_names or [])]
        auxiliary_hidden_dims = [int(width) for width in (auxiliary_hidden_dims or [])]
        # Plain lists of whole numbers, whether they came from a default or from YAML.
        harmonic_hidden_dims = [int(width) for width in (harmonic_hidden_dims or [])]
        target_names = [str(name) for name in (target_names or [])]
        # Every target the run fits, which with one model per target is more than this model
        # predicts.
        fitted_target_names = [str(name) for name in (fitted_target_names or target_names)]
        auxiliary_enabled = bool(auxiliary_enabled)
        fusion = str(fusion).lower()
        if fusion not in FUSION_TYPES:
            raise ValueError(f"fusion must be one of {list(FUSION_TYPES)}, got {fusion!r}")
        attention_static_tokens = str(attention_static_tokens).lower()
        if fusion == "attention" and attention_static_tokens not in STATIC_TOKEN_MODES:
            raise ValueError(
                f"attention_static_tokens must be one of {list(STATIC_TOKEN_MODES)}, got {attention_static_tokens!r}"
            )
        attention_d_model = int(attention_d_model)
        attention_nhead = int(attention_nhead)
        attention_num_layers = int(attention_num_layers)
        attention_ff_multiplier = int(attention_ff_multiplier)
        attention_dropout = float(attention_dropout)
        attention_readout = str(attention_readout).lower()
        residual_enabled = bool(residual_enabled)
        residual_base_columns = {
            str(target): str(column) for target, column in dict(residual_base_columns or {}).items()
        }
        residual_base_hidden_dims = [int(width) for width in (residual_base_hidden_dims or [])]
        # A list either way, so the width check below reports a clear error rather than failing on
        # a missing value.
        auxiliary_label_mean = as_float_list(auxiliary_label_mean) or []
        auxiliary_label_scale = as_float_list(auxiliary_label_scale) or []
        self.save_hyperparameters()

        self._init_regression_targets(
            target_dim=target_dim,
            target_names=target_names,
            target_mean=target_mean,
            target_scale=target_scale,
            target_transform=target_transform,
            predict_variance=predict_variance,
            beta_nll=beta_nll,
            loss_name=loss_name,
            huber_delta=huber_delta,
            loss_base=loss_base,
            loss_lambda=loss_lambda,
            loss_shrinkage=loss_shrinkage,
            loss_min_batch=loss_min_batch,
            cosine_space=cosine_space,
            target_covariance=target_covariance,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            scheduler_min_lr=scheduler_min_lr,
            scheduler_monitor=scheduler_monitor,
        )

        # Numeric covariates only: the categories arrive as codes and contribute their embeddings.
        self.static_dim = int(static_dim)
        self.static_hidden_dims = list(static_hidden_dims)
        # The covariate branch's output width is its last layer.
        self.static_hidden_dim = self.static_hidden_dims[-1]
        # A saved model may carry the labels without their count; one follows from the other.
        if not categorical_cardinalities and categorical_vocabularies:
            categorical_cardinalities = [len(vocabulary) + 1 for vocabulary in categorical_vocabularies]
        self.categorical_cardinalities = list(categorical_cardinalities)
        self.categorical_vocabularies = list(categorical_vocabularies)
        self.categorical_feature_names = list(categorical_feature_names) or [
            f"categorical_{index}" for index in range(len(self.categorical_cardinalities))
        ]
        self.temporal_encoder_name = str(temporal_encoder).lower()
        if self.temporal_encoder_name not in {"dilated_tempcnn", "annual_grid2d"}:
            raise ValueError(f"temporal_encoder must be 'dilated_tempcnn' or 'annual_grid2d', got {temporal_encoder!r}")
        # None lets each batch use its own span. Harmless, because empty years contribute nothing,
        # but a fixed span keeps every batch's grid the same shape.
        self.grid_years = None if grid_years in (None, 0) else max(1, int(grid_years))

        self.modality_dims = {
            str(name).lower(): int(dim)
            for name, dim in dict(modality_dims or {}).items()
            if dim is not None and int(dim) > 0
        }
        self.temporal_enabled = bool(temporal_enabled) and bool(self.modality_dims)
        self._cnn_hidden_dims = cnn_hidden_dims
        self._modality_embed_dim = modality_embed_dim

        # Before the covariate branch: one token per column changes how it is built.
        self.fusion_type = fusion
        self.attention_static_tokens = attention_static_tokens
        self.static_encoder = self._build_static_encoder(
            dropout,
            embedding_dims=embedding_dims,
            embedding_dropout=embedding_dropout,
            embedding_max_dim=embedding_max_dim,
            continuous_norm=continuous_norm,
        )

        self.rasterizers = nn.ModuleDict()
        self.temporal_encoders = nn.ModuleDict()
        if self.temporal_enabled:
            for modality_name, modality_dim in self.modality_dims.items():
                rasterizer = CalendarGridRasterizer(
                    num_channels=modality_dim,
                    grid_years=self.grid_years,
                    use_validity_channels=use_validity_channels,
                    month_positional=month_positional,
                )
                self.rasterizers[modality_name] = rasterizer
                encoder_cls = (
                    DilatedTempCNNEncoder if self.temporal_encoder_name == "dilated_tempcnn" else AnnualGrid2DEncoder
                )
                self.temporal_encoders[modality_name] = encoder_cls(
                    num_channels=rasterizer.output_channels,
                    output_dim=int(
                        self._per_modality_value(self._modality_embed_dim, modality_name, "modality_embed_dim")
                    ),
                    hidden_dims=_as_width_list(
                        self._per_modality_value(self._cnn_hidden_dims, modality_name, "cnn_hidden_dims")
                    ),
                    dropout=dropout,
                    norm=cnn_norm,
                    pool=pool,
                )

        self.target_names = list(target_names)
        self.fitted_target_names = list(fitted_target_names)
        self.auxiliary_enabled = auxiliary_enabled
        self.auxiliary_validity_channels = bool(auxiliary_validity_channels)
        self.auxiliary_encoder = self._build_auxiliary_encoder(
            auxiliary_label_columns if auxiliary_enabled else [],
            auxiliary_available_names,
            auxiliary_hidden_dims,
            auxiliary_dropout,
        )

        self.coordinate_encoder = self._build_coordinate_encoder(
            coord_dim,
            harmonic_num_frequencies,
            harmonic_include_input,
            harmonic_hidden_dims,
            harmonic_dropout,
        )

        # After the auxiliary branch, whose column list the base is checked against.
        self.residual_enabled = residual_enabled
        self.residual_base_columns = residual_base_columns
        self.residual_base_validity_channels = bool(residual_base_validity_channels)
        self.residual_base_max_missing = float(residual_base_max_missing)
        self.base_encoder = self._build_residual_base_encoder(
            auxiliary_label_mean,
            auxiliary_label_scale,
            residual_base_hidden_dims,
            float(residual_base_dropout),
        )

        self.fusion = self._build_fusion(
            d_model=attention_d_model,
            nhead=attention_nhead,
            num_layers=attention_num_layers,
            ff_multiplier=attention_ff_multiplier,
            dropout=attention_dropout,
            readout=attention_readout,
        )
        # Built last, from the widths every branch above settled on.
        self.output_head = build_mlp_stack(
            self.fusion.output_dim + self.auxiliary_output_dim + self.residual_base_output_dim,
            head_hidden_dims,
            # Twice the number of targets when the model also predicts a spread.
            self.head_output_dim,
            dropout=dropout,
            activation="gelu",
            norm_final=bool(head_norm_final),
        )

    # --- construction helpers ----------------------------------------------

    def _per_modality_value(self, setting: Any, modality_name: str, label: str) -> Any:
        """Read a setting that is either one value for every data source, or one per source.

        Raises
        ------
        ValueError
            If it is given per source and one is missing - silently defaulting would undo the
            per-source sizing without a word.
        """
        if isinstance(setting, Mapping):
            value = setting.get(modality_name, setting.get(str(modality_name).lower()))
            if value is None:
                raise ValueError(
                    f"Modality {modality_name!r} has no entry in {label} "
                    f"(configured: {sorted(setting)}). Add one for it, or use a single value "
                    "to apply the same setting to every modality."
                )
            return value
        return setting

    @property
    def _static_tokens_per_feature(self) -> bool:
        """Whether attention fusion reads one token per covariate column."""
        return self.fusion_type == "attention" and self.attention_static_tokens == "per_feature"

    def _static_token_dims(self) -> list[int]:
        """How the covariate summary is cut into tokens for attention fusion."""
        if self.attention_static_tokens == "summary":
            return [self.static_hidden_dim]
        if not self.has_static_features:
            return []
        return [1] * self.static_dim + list(self.static_encoder.embedding_dims)

    def _build_fusion(self, **attention: Any) -> nn.Module:
        """Build the :term:`fusion`, gated or attention. The two are called the same way."""
        # The data sources in the order the branches are joined.
        temporal_dims = [encoder.output_dim for encoder in self.temporal_encoders.values()]
        if self.fusion_type == "attention":
            return AttentionFusion(self._static_token_dims(), temporal_dims, self.coordinate_output_dim, **attention)
        # The covariate branch keeps its width even with no covariates, so the fused vector always
        # has the same shape.
        return ConcatGatedFusion(self.static_hidden_dim, sum(temporal_dims), self.coordinate_output_dim)

    def _build_auxiliary_encoder(
        self,
        selected: list[str],
        available: list[str],
        hidden_dims: list[int],
        dropout: float,
    ) -> nn.Module:
        """Build the branch reading the measured lab values, after checking the named columns.

        Columns are chosen by name, never by position: the order of the lab columns follows
        ``LABEL_COLUMNS``, so a position would point at a different measurement the moment that list
        is reordered.

        Raises
        ------
        ValueError
            If a named column is a target of the run, is not carried with the data, or is listed
            twice; or if the target names needed for that check were not supplied.
        """
        self.auxiliary_label_columns = list(selected)
        self.auxiliary_available_names = list(available)
        self.auxiliary_output_dim = 0
        # Present even when empty, so a model saved without auxiliary columns still loads into one
        # that declares them.
        self.register_buffer("auxiliary_index", torch.zeros(0, dtype=torch.long), persistent=True)
        if not selected:
            return nn.Identity()

        if not self.fitted_target_names:
            # Without the target names the check below cannot run, and skipping it is how a model
            # ends up reading its own answer.
            raise ValueError(
                "auxiliary_label_columns requires target_names so a selected column can be checked "
                "against what is being fitted; pass target_names explicitly."
            )

        # Checked against every target the run fits, not only this model's: with one model per
        # target, another target as an input leaks the answer through whatever the two share.
        leaking = [name for name in selected if name in set(self.fitted_target_names)]
        if leaking:
            raise ValueError(
                f"auxiliary_label_columns may not name a column being fitted: {sorted(leaking)} "
                f"also appear(s) in the run's targets {sorted(self.fitted_target_names)}. The model "
                "would read a target as an input."
            )

        if not available:
            # A different message from the unknown-column case below: here the columns may be
            # perfectly well declared and simply not carried.
            raise ValueError(
                f"auxiliary_label_columns names {sorted(selected)} but no lab columns are being "
                "carried with the data. Set CARRY_LABEL_COLUMNS: true in data_spec.yml to make the "
                "LABEL_COLUMNS entries available as auxiliary inputs."
            )

        unknown = [name for name in selected if name not in set(available)]
        if unknown:
            raise ValueError(
                f"auxiliary_label_columns names column(s) the data does not carry: {sorted(unknown)}. "
                f"Available: {sorted(available)}. A lab column must be listed in LABEL_COLUMNS and "
                "present in the static or targets source to be selectable."
            )

        duplicates = sorted({name for name in selected if selected.count(name) > 1})
        if duplicates:
            raise ValueError(f"auxiliary_label_columns lists duplicate column(s): {duplicates}")

        self.auxiliary_index = torch.as_tensor([available.index(name) for name in selected], dtype=torch.long)
        # The flags double the width: one per column, so the model can discount a filled-in value
        # instead of reading it as a measurement.
        input_dim = len(selected) * (2 if self.auxiliary_validity_channels else 1)
        # No widths makes this a pass-through, which is the raw mode; one code path either way.
        encoder = build_mlp_stack(
            input_dim,
            hidden_dims,
            None,
            dropout=dropout,
            activation="gelu",
            use_layer_norm=True,
            # This block feeds the head's inputs, not a prediction, so both are on.
            norm_final=True,
            dropout_final=True,
        )
        self.auxiliary_output_dim = hidden_dims[-1] if hidden_dims else input_dim
        return encoder

    def _build_residual_base_encoder(
        self,
        label_mean: list[float],
        label_scale: list[float],
        hidden_dims: list[int],
        dropout: float,
    ) -> nn.Module:
        """Build the block reading the :term:`residual base`, after checking the named columns.

        One base column per target, chosen by name against the same list of lab columns the
        auxiliary branch uses.

        Raises
        ------
        ValueError
            If no base is given while ``residual_enabled`` is set, a target has none, a base names a
            target of the run or a column the data does not carry, a column is also an auxiliary
            input, or the lab statistics needed to convert the base are missing.
        """
        self.residual_base_output_dim = 0
        if not self.residual_enabled:
            # Nothing registered at all, so a model saved without a base still loads exactly.
            return nn.Identity()

        self.register_buffer("residual_base_index", torch.zeros(0, dtype=torch.long), persistent=True)
        self.register_buffer("residual_base_label_mean", torch.zeros(0), persistent=True)
        self.register_buffer("residual_base_label_scale", torch.ones(0), persistent=True)

        selected = self.residual_base_columns
        if not selected:
            # With the switch on and no base there is nothing to correct. Carrying on would train a
            # different model from the one configured, with nothing to say so.
            raise ValueError(
                "residual_base_columns is required when residual_enabled is true: the head anchors on "
                "a base prediction per target. Set residual_enabled: false for a model with no base."
            )

        missing = [name for name in self.target_names if name not in selected]
        if missing:
            raise ValueError(
                f"residual_base_columns has no entry for target(s) {sorted(missing)}; it must name a "
                f"base column for every target this model fits ({sorted(self.target_names)}). A "
                "partially anchored head would learn residuals for some outputs and absolute values "
                "for others, against a single loss."
            )

        extra = [name for name in selected if name not in set(self.target_names)]
        if extra:
            raise ValueError(
                f"residual_base_columns names target(s) this model does not fit: {sorted(extra)}. "
                f"This model's targets are {sorted(self.target_names)}."
            )

        # The base must be a prediction of the target, not the measured target.
        leaking = [column for column in selected.values() if column in set(self.fitted_target_names)]
        if leaking:
            raise ValueError(
                f"residual_base_columns may not name a column being fitted: {sorted(leaking)} "
                f"also appear(s) in the run's targets {sorted(self.fitted_target_names)}. The base "
                "must be a PREDICTION of the target, not the target."
            )

        shared = sorted(set(selected.values()) & set(self.auxiliary_label_columns))
        if shared:
            raise ValueError(
                f"Column(s) {shared} appear in both residual_base_columns and "
                "auxiliary_label_columns. A base column already reaches the head through its own "
                "block; listing it twice feeds it twice and lets the two roles be configured apart."
            )

        available = list(self.auxiliary_available_names)
        if not available:
            raise ValueError(
                f"residual_base_columns names {sorted(set(selected.values()))} but no lab columns "
                "are being carried with the data. Set CARRY_LABEL_COLUMNS: true in data_spec.yml, "
                "and list the base column(s) in LABEL_COLUMNS."
            )

        unknown = [column for column in selected.values() if column not in set(available)]
        if unknown:
            raise ValueError(
                f"residual_base_columns names column(s) the data does not carry: {sorted(unknown)}. "
                f"Available: {sorted(available)}. A base column must be listed in LABEL_COLUMNS and "
                "present in the static or targets source."
            )

        if len(label_mean) != len(available) or len(label_scale) != len(available):
            # Without these the base stays in standardized lab units, and adding it would shift the
            # prediction by a meaningless amount.
            raise ValueError(
                "residual_base_columns requires auxiliary_label_mean and auxiliary_label_scale at "
                f"the roster's width ({len(available)}); got {len(label_mean)} and "
                f"{len(label_scale)}. They are what maps the base out of the lab standardizer."
            )

        # In the order of the targets, so each base lines up with the value it corrects.
        indices = [available.index(selected[name]) for name in self.target_names]
        self.residual_base_index = torch.as_tensor(indices, dtype=torch.long)
        self.residual_base_label_mean = torch.as_tensor([label_mean[index] for index in indices], dtype=torch.float32)
        self.residual_base_label_scale = torch.as_tensor([label_scale[index] for index in indices], dtype=torch.float32)

        logger.warning(
            "%s anchors on %s. These must be OUT-OF-FOLD predictions for the split this run uses: "
            "nothing in the pipeline can verify that, and in-fold predictions will inflate every "
            "reported metric.",
            type(self).__name__,
            {name: selected[name] for name in self.target_names},
        )

        input_dim = self.target_dim * (2 if self.residual_base_validity_channels else 1)
        # No widths makes this a pass-through, as in the auxiliary and location blocks.
        encoder = build_mlp_stack(
            input_dim,
            hidden_dims,
            None,
            dropout=dropout,
            activation="gelu",
            use_layer_norm=True,
            # This block feeds the head's inputs, not a prediction, so both are on.
            norm_final=True,
            dropout_final=True,
        )
        self.residual_base_output_dim = hidden_dims[-1] if hidden_dims else input_dim
        return encoder

    def _build_coordinate_encoder(
        self,
        coord_dim: Any,
        num_frequencies: int,
        include_input: bool,
        hidden_dims: list[int],
        dropout: float,
    ) -> nn.Module:
        """Build the location branch, or nothing at all when the data carries no coordinates.

        With no coordinates the model has exactly the weights it would have had without the option,
        so checkpoints load either way.
        """
        self.coord_dim = int(coord_dim or 0)
        self.coordinate_output_dim = 0
        if self.coord_dim <= 0:
            return nn.Identity()

        encoder = HarmonicPositionEncoder(
            num_coordinates=self.coord_dim,
            num_frequencies=num_frequencies,
            include_input=include_input,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.coordinate_output_dim = encoder.output_dim
        return encoder

    @property
    def has_coordinates(self) -> bool:
        """Whether this model reads the point's location."""
        return self.coord_dim > 0

    @property
    def has_auxiliary_labels(self) -> bool:
        """Whether this model reads measured lab values as inputs."""
        return bool(self.auxiliary_label_columns)

    @property
    def serving_label_columns(self) -> list[str]:
        """The lab columns new points must supply for this model to predict properly.

        The auxiliary inputs, plus the base columns when the model predicts a correction. A column
        missing from a request is filled in and flagged rather than refused - but a filled-in base
        leaves the model correcting the same constant for every point.
        """
        auxiliary = list(self.auxiliary_label_columns)
        if not self.residual_enabled:
            return auxiliary
        base = [self.residual_base_columns[name] for name in self.target_names]
        return auxiliary + [column for column in base if column not in set(auxiliary)]

    @property
    def has_static_features(self) -> bool:
        """Whether this model reads covariates at all, numeric or category."""
        return self.static_dim > 0 or bool(self.categorical_cardinalities)

    def _build_static_encoder(
        self,
        dropout: float,
        *,
        embedding_dims: Any,
        embedding_dropout: float,
        embedding_max_dim: int,
        continuous_norm: str,
    ) -> nn.Module:
        """Build the covariate branch, or nothing when there are no covariates."""
        if not self.has_static_features:
            return nn.Identity()
        # No final projection: the branch keeps its own width, because the fusion joins the
        # branches rather than mixing them at a shared width.
        return TabularStaticEncoder(
            num_continuous=self.static_dim,
            hidden_dims=self.static_hidden_dims,
            cardinalities=self.categorical_cardinalities,
            embedding_dims=embedding_dims,
            embedding_dropout=embedding_dropout,
            embedding_max_dim=embedding_max_dim,
            feature_names=self.categorical_feature_names,
            dropout=dropout,
            activation="gelu",
            use_layer_norm=True,
            continuous_norm=continuous_norm,
            # One token per column is cut from the unsummarized covariates, so the branch stops
            # before its layers.
            mlp=not self._static_tokens_per_feature,
        )

    # --- coverage guard -----------------------------------------------------

    def on_fit_start(self) -> None:
        """Refuse to train when a :term:`residual base` column is missing on too many points.

        A missing base is filled in with the training median, so the model would be correcting that
        constant rather than a prediction for the point - and nothing else would look wrong.

        Raises
        ------
        ValueError
            If a base column is missing on more than ``residual_base_max_missing`` of the points.
        """
        super().on_fit_start()
        if not self.residual_enabled:
            return
        datamodule = getattr(self.trainer, "datamodule", None) if self.trainer is not None else None
        bundle = getattr(datamodule, "sequence_bundle", None)
        if bundle is None or not hasattr(bundle, "label_missing_fraction"):
            return

        offenders = {}
        for name in self.target_names:
            column = self.residual_base_columns[name]
            fraction = float(bundle.label_missing_fraction(column))
            if fraction > self.residual_base_max_missing:
                offenders[column] = round(fraction, 4)
        if offenders:
            raise ValueError(
                f"Base column(s) {offenders} exceed residual_base_max_missing="
                f"{self.residual_base_max_missing}. A missing base is median-filled, so the head "
                "would be correcting the training median rather than a prediction for that point. "
                "Regenerate the predictions over the full population, or raise the threshold "
                "deliberately."
            )

    # --- forward -----------------------------------------------------------

    def _encode_static(self, x_static: torch.Tensor, x_categorical: Optional[torch.Tensor]) -> torch.Tensor:
        """Summarize the covariates, or return zeros when the model has none."""
        if not self.has_static_features:
            return torch.zeros((x_static.size(0), self.static_hidden_dim), device=x_static.device, dtype=x_static.dtype)
        return self.static_encoder(x_static, x_categorical)

    def _select_coordinates(self, batch: Any, device, dtype) -> Optional[torch.Tensor]:
        """The coordinates as the location branch receives them.

        Kept separate from encoding them so the SHAP figures can credit latitude and longitude
        themselves, rather than the waves they are turned into.
        """
        if not self.has_coordinates:
            return None

        coords = batch_get(batch, "x_coords")
        if coords is None:
            raise KeyError(
                "Batch is missing 'x_coords'; this model was built with coord_dim="
                f"{self.coord_dim} and needs the normalized coordinates the sequence datamodule "
                "collates under USE_HARMONIC_COORDS."
            )
        coords = coords.to(device=device, dtype=dtype)
        if coords.size(-1) != self.coord_dim:
            # At the wrong width the branch would read longitude out of the latitude column and
            # still return something well shaped.
            raise ValueError(
                f"Batch carries {coords.size(-1)} coordinate column(s) but this model was built for {self.coord_dim}."
            )
        return coords

    def _encode_coordinates(self, batch: Any, device, dtype) -> Optional[torch.Tensor]:
        """Run the location branch, or return nothing when the model has none."""
        coords = self._select_coordinates(batch, device=device, dtype=dtype)
        if coords is None:
            return None
        return self.coordinate_encoder(coords)

    def _select_auxiliary(self, batch: Any, device, dtype) -> Optional[torch.Tensor]:
        """The measured lab values as the auxiliary branch receives them, flags after values.

        Kept separate from encoding them so the SHAP figures can credit each column itself.
        """
        if not self.has_auxiliary_labels:
            return None

        values = batch_get(batch, "x_labels")
        if values is None:
            raise KeyError(
                "Batch is missing 'x_labels'; auxiliary_label_columns needs the measured lab values "
                f"({self.auxiliary_label_columns}) that the sequence datamodule collates."
            )
        values = values.to(device=device, dtype=dtype)
        if values.size(-1) != len(self.auxiliary_available_names):
            # The positions were worked out against the lab columns this model was built with. A
            # different width means different columns, and it would read the wrong measurement.
            raise ValueError(
                f"Batch carries {values.size(-1)} lab column(s) but this model resolved its "
                f"auxiliary columns against {len(self.auxiliary_available_names)}; the data no "
                "longer matches the checkpoint's label roster."
            )

        selected = values.index_select(-1, self.auxiliary_index)
        if self.auxiliary_validity_channels:
            validity = batch_get(batch, "x_label_validity")
            if validity is None:
                raise KeyError(
                    "Batch is missing 'x_label_validity'; set auxiliary_validity_channels=False to "
                    "run without the measured-vs-filled flags."
                )
            validity = validity.to(device=device, dtype=dtype).index_select(-1, self.auxiliary_index)
            selected = torch.cat([selected, validity], dim=-1)

        return selected

    def _encode_auxiliary(self, batch: Any, device, dtype) -> Optional[torch.Tensor]:
        """Run the auxiliary branch, or return nothing when the model has none."""
        selected = self._select_auxiliary(batch, device=device, dtype=dtype)
        if selected is None:
            return None
        return self.auxiliary_encoder(selected)

    def _residual_base(self, batch: Any, *, device, dtype) -> torch.Tensor:
        """The :term:`residual base`, converted into the units the model trains in, with its flags.

        One tensor, because it serves twice: it is added to what the head predicts, and it is also
        an input to the head.
        """
        values = batch_get(batch, "x_labels")
        if values is None:
            raise KeyError(
                "Batch is missing 'x_labels'; residual_base_columns needs the base prediction "
                f"column(s) {sorted(set(self.residual_base_columns.values()))} that the sequence "
                "datamodule collates."
            )
        values = values.to(device=device, dtype=dtype)
        if values.size(-1) != len(self.auxiliary_available_names):
            raise ValueError(
                f"Batch carries {values.size(-1)} lab column(s) but this model resolved its base "
                f"columns against {len(self.auxiliary_available_names)}; the data no longer matches "
                "the checkpoint's label roster."
            )

        base = values.index_select(-1, self.residual_base_index)
        base = base * self.residual_base_label_scale + self.residual_base_label_mean
        if bool(self.targets_are_log1p):
            # The same transform the targets went through, negatives clipped the same way.
            base = 10.0 * torch.log1p(base.clamp_min(0.0))
        if bool(self.targets_are_standardized):
            base = (base - self.target_mean) / self.target_scale

        if not self.residual_base_validity_channels:
            return base

        validity = batch_get(batch, "x_label_validity")
        if validity is None:
            raise KeyError(
                "Batch is missing 'x_label_validity'; set residual_base_validity_channels=False to "
                "run without the measured-vs-filled flags."
            )
        validity = validity.to(device=device, dtype=dtype).index_select(-1, self.residual_base_index)
        return torch.cat([base, validity], dim=-1)

    def _head_from_base(self, fused: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
        """Predict a correction from the fused summary and the base, then add the base back."""
        fused = torch.cat([fused, self.base_encoder(block)], dim=-1)
        mean, log_variance = self._split_head_output(self.output_head(fused))
        mean = mean + block[..., : self.target_dim]
        if log_variance is None:
            return mean
        # Added to the predicted values only: adding it to the predicted spreads as well would
        # turn a base of 40 into an absurd uncertainty.
        return torch.cat([mean, log_variance], dim=-1)

    def _rasterize(self, batch: Any, device, dtype) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Lay each :term:`data source`\'s readings onto its :term:`calendar grid`.

        Also where the SHAP explanations start: everything after this point can be traced back to
        the grid, so each band can be credited individually.
        """
        sequences = batch_get(batch, "sequences", {}) or {}
        masks = batch_get(batch, "sequence_mask", {}) or {}
        times = batch_get(batch, "sequence_time", {}) or {}
        validities = batch_get(batch, "sequence_validity", {}) or {}

        grids: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for modality_name in self.temporal_encoders:
            values = sequences.get(modality_name)
            mask = masks.get(modality_name)
            modality_times = times.get(modality_name)
            if values is None or mask is None or modality_times is None:
                raise KeyError(
                    f"Batch is missing sequence data for modality '{modality_name}'; expected it in "
                    "'sequences', 'sequence_mask' and 'sequence_time'"
                )
            validity = validities.get(modality_name)
            grids[modality_name] = self.rasterizers[modality_name](
                values.to(device=device, dtype=dtype),
                mask.to(device=device),
                modality_times.to(device=device),
                None if validity is None else validity.to(device=device),
            )
        return grids

    def _encode_temporal_from_grids(
        self, grids: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        """Summarize every data source's grid and join the summaries."""
        if not self.temporal_encoders:
            return None

        embeddings = []
        for modality_name, encoder in self.temporal_encoders.items():
            grid, cell_mask = grids[modality_name]
            embedding = encoder(grid, cell_mask)
            # A point with no readings at all contributes nothing, rather than a constant the
            # fusion would have to learn to ignore.
            embeddings.append(embedding * cell_mask.flatten(1).any(dim=1, keepdim=True).to(dtype=embedding.dtype))

        return torch.cat(embeddings, dim=-1)

    def _encode_temporal(self, batch: Any, device, dtype) -> Optional[torch.Tensor]:
        """Lay out and summarize the time series, or return nothing when it is not used."""
        if not self.temporal_encoders:
            return None
        return self._encode_temporal_from_grids(self._rasterize(batch, device=device, dtype=dtype))

    def _fuse(self, batch: Any, *, device, dtype) -> torch.Tensor:
        """Run every branch and combine them into the vector the prediction head reads."""
        x_static = batch_get(batch, "x_static")
        if x_static is None:
            raise KeyError("Batch is missing 'x_static'")

        x_static = x_static.to(device=device, dtype=dtype)
        x_categorical = batch_get(batch, "x_categorical")
        if x_categorical is not None:
            x_categorical = x_categorical.to(device=device)

        static_features = self._encode_static(x_static, x_categorical)
        temporal_features = self._encode_temporal(batch, device=device, dtype=dtype)
        # Location goes through the fusion rather than past it, so it can change how much of the
        # other branches gets through.
        coordinate_features = self._encode_coordinates(batch, device=device, dtype=dtype)
        fused = self.fusion(static_features, temporal_features, coordinate_features)

        # Added after the fusion, so a measured lab value reaches the head at full strength rather
        # than competing with the branches that had to infer it.
        auxiliary_features = self._encode_auxiliary(batch, device=device, dtype=dtype)
        if auxiliary_features is not None:
            fused = torch.cat([fused, auxiliary_features], dim=-1)
        return fused

    def forward(self, batch: Any) -> torch.Tensor:
        """Predict from one batch, in the units the model trains in.

        Parameters
        ----------
        batch : mapping
            What the datamodule collates: ``x_static``, ``x_categorical``, ``sequences`` and their
            masks, dates and flags, and ``x_coords`` / ``x_labels`` where the model uses them.

        Returns
        -------
        torch.Tensor
            One value per target, or twice that with a :term:`variance head`.

        Raises
        ------
        KeyError
            If the batch is missing something the model was built to read.
        """
        reference = next(self.parameters())
        device, dtype = reference.device, reference.dtype
        fused = self._fuse(batch, device=device, dtype=dtype)
        if not self.residual_enabled:
            return self.output_head(fused)
        return self._head_from_base(fused, self._residual_base(batch, device=device, dtype=dtype))

    # --- what the SHAP explanations work on ---------------------------------
    # explanation_parts() splits a batch into the tensors the explanation varies, and
    # forward_from_parts() predicts from exactly those. The two must agree exactly, or the figures
    # would describe a model other than the one being scored. Neither is used while training.

    def explanation_parts(self, batch: Any) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        """Split a batch into the inputs an explanation varies, and say what each one is.

        Returns
        -------
        parts : list of torch.Tensor
            The tensors to vary: the covariates, the category embeddings, each data source's grid,
            the coordinates, the auxiliary lab values and the residual base, where present.
        groups : list of dict
            One entry per input, naming it and the columns it occupies. Contributions are added up
            within a group, which is valid because SHAP contributions add.
        """
        reference = next(self.parameters())
        device, dtype = reference.device, reference.dtype

        x_static = batch_get(batch, "x_static")
        if x_static is None:
            raise KeyError("Batch is missing 'x_static'")
        x_static = x_static.to(device=device, dtype=dtype)

        # The spatial-context covariates are ordinary covariates; only their label differs, so the
        # figures can report them as a block.
        context_names = set(getattr(self, "context_feature_names", None) or [])
        parts: list[torch.Tensor] = [x_static]
        groups: list[dict[str, Any]] = [
            {
                "part": 0,
                "kind": "context" if name in context_names else "static",
                "name": name,
                "columns": [index],
            }
            for index, name in enumerate(self._static_feature_names(x_static.size(-1)))
        ]

        # Categories are varied through their embeddings: a code is looked up, not computed, so
        # nothing can be traced through it. One group per column.
        if self.has_static_features and self.static_encoder.embeddings.num_features:
            x_categorical = batch_get(batch, "x_categorical")
            if x_categorical is None:
                raise KeyError("Batch is missing 'x_categorical'")
            embedded = self.static_encoder.embeddings(x_categorical.to(device=device)).to(dtype=dtype)
            part_index = len(parts)
            parts.append(embedded)
            cursor = 0
            for name, width in zip(
                self.static_encoder.embeddings.feature_names,
                self.static_encoder.embeddings.embedding_dims,
            ):
                groups.append(
                    {
                        "part": part_index,
                        "kind": "categorical",
                        "name": name,
                        "columns": list(range(cursor, cursor + int(width))),
                    }
                )
                cursor += int(width)

        # One grid per data source. Each band's readings and its flags are one group, since they
        # describe the same band; the month channels are a group of their own.
        grids = self._rasterize(batch, device=device, dtype=dtype) if self.temporal_encoders else {}
        for modality_name in self.temporal_encoders:
            grid, _cell_mask = grids[modality_name]
            part_index = len(parts)
            parts.append(grid)
            layout = self.rasterizers[modality_name].channel_layout()
            column_names = self._modality_column_names(modality_name, len(layout["values"]))
            for band_index, band_name in enumerate(column_names):
                columns = [layout["values"][band_index]]
                if layout["validity"]:
                    columns.append(layout["validity"][band_index])
                groups.append(
                    {
                        "part": part_index,
                        "kind": "temporal",
                        "modality": modality_name,
                        "name": band_name,
                        "columns": columns,
                    }
                )
            if layout["month_positional"]:
                groups.append(
                    {
                        "part": part_index,
                        "kind": "temporal",
                        "modality": modality_name,
                        "name": f"{modality_name}_month_positional",
                        "columns": list(layout["month_positional"]),
                    }
                )

        # The two coordinates, not the waves they become: the figures then have a row for latitude
        # and one for longitude instead of dozens that mean nothing on their own.
        coords = self._select_coordinates(batch, device=device, dtype=dtype)
        if coords is not None:
            part_index = len(parts)
            parts.append(coords)
            for index, name in enumerate(self._coordinate_names(coords.size(-1))):
                groups.append(
                    {
                        "part": part_index,
                        "kind": "spatial",
                        "name": name,
                        "columns": [index],
                    }
                )

        # The auxiliary lab values, followed by their flags.
        selected = self._select_auxiliary(batch, device=device, dtype=dtype)
        if selected is not None:
            part_index = len(parts)
            parts.append(selected)
            count = len(self.auxiliary_label_columns)
            for index, name in enumerate(self.auxiliary_label_columns):
                columns = [index]
                if self.auxiliary_validity_channels:
                    columns.append(count + index)
                groups.append(
                    {
                        "part": part_index,
                        "kind": "auxiliary",
                        "name": name,
                        "columns": columns,
                    }
                )

        # The residual base last, as one input: it is both an input and the value added back.
        if self.residual_enabled:
            block = self._residual_base(batch, device=device, dtype=dtype)
            part_index = len(parts)
            parts.append(block)
            for index, name in enumerate(self.target_names):
                columns = [index]
                if self.residual_base_validity_channels:
                    columns.append(self.target_dim + index)
                groups.append(
                    {
                        "part": part_index,
                        "kind": "residual_base",
                        "name": self.residual_base_columns[name],
                        "target": name,
                        "columns": columns,
                    }
                )

        return parts, groups

    def forward_from_parts(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        """Predict from the parts :meth:`explanation_parts` returned, and from nothing else.

        The explanation runs the model on blends of a point and other points, so everything the
        prediction needs has to come from the varied tensors themselves - which is why the grid
        carries its own record of which cells hold a reading.

        Returns
        -------
        torch.Tensor
            The predicted values only, never the predicted spread: the figures would otherwise show
            twice as many targets as the model has.
        """
        fused, cursor = self._fuse_from_parts(parts)
        if self.residual_enabled:
            raw = self._head_from_base(fused, parts[cursor])
        else:
            raw = self.output_head(fused)
        return self._split_head_output(raw)[0]

    def _fuse_from_parts(self, parts: Sequence[torch.Tensor]) -> tuple[torch.Tensor, int]:
        """:meth:`_fuse` over the explanation parts; also returns where the base part starts."""
        cursor = 0
        x_static = parts[cursor]
        cursor += 1

        embedded = None
        if self.has_static_features and self.static_encoder.embeddings.num_features:
            embedded = parts[cursor]
            cursor += 1

        if self.has_static_features:
            static_features = self.static_encoder.forward_with_embedding(x_static, embedded)
        else:
            static_features = torch.zeros(
                (x_static.size(0), self.static_hidden_dim), device=x_static.device, dtype=x_static.dtype
            )

        temporal_features = None
        if self.temporal_encoders:
            grids = {}
            for modality_name in self.temporal_encoders:
                grid = parts[cursor]
                cursor += 1
                observed_index = self.rasterizers[modality_name].channel_layout()["cell_observed"][0]
                grids[modality_name] = (grid, grid[:, observed_index] > 0.5)
            temporal_features = self._encode_temporal_from_grids(grids)

        coordinate_features = None
        if self.has_coordinates:
            coordinate_features = self.coordinate_encoder(parts[cursor])
            cursor += 1

        fused = self.fusion(static_features, temporal_features, coordinate_features)

        if self.has_auxiliary_labels:
            fused = torch.cat([fused, self.auxiliary_encoder(parts[cursor])], dim=-1)
            cursor += 1

        return fused, cursor

    def _static_feature_names(self, width: int) -> list[str]:
        """The covariate names, or numbered placeholders when the model carries none."""
        names = list(getattr(self, "static_feature_names", None) or [])
        if len(names) == width:
            return names
        return [f"static_{index}" for index in range(width)]

    def _coordinate_names(self, width: int) -> list[str]:
        """The coordinate names, or numbered placeholders when the model carries none."""
        names = list(getattr(self, "coord_names", None) or [])
        if len(names) == width:
            return names
        return [f"coord_{index}" for index in range(width)]

    def _modality_column_names(self, modality_name: str, width: int) -> list[str]:
        """One data source's band names, or numbered placeholders when the model carries none."""
        stored = (getattr(self, "modality_column_names", None) or {}).get(modality_name)
        names = list(stored or [])
        if len(names) == width:
            return names
        return [f"{modality_name}_{index}" for index in range(width)]
