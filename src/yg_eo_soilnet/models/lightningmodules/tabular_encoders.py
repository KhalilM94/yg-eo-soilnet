"""The branch that reads a point's covariates, numeric and categorical alike.

Each category label becomes an :term:`embedding` - a learned vector, looked up by the label's code -
which is joined to the standardized numeric covariates and summarized by a few layers. The codes
come from :mod:`yg_eo_soilnet.datamodules.categorical`, which numbers the labels from the training
points; code 0 means "unknown or missing".
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from yg_eo_soilnet.models.lightningmodules.mlp import build_mlp_stack


#: Widest embedding the heuristic will produce, regardless of cardinality.
DEFAULT_EMBEDDING_MAX_DIM = 50


def embedding_dim_for(cardinality: int, max_dim: int = DEFAULT_EMBEDDING_MAX_DIM) -> int:
    """Choose how wide the embedding of a category column should be.

    Half its number of codes, capped at ``max_dim``: a column with few labels gets nearly a column
    per label, while one with many is forced to share structure between them, which is the point of
    an embedding. The cap stops a column with hundreds of labels dominating everything else.

    Parameters
    ----------
    cardinality : int
        How many codes the column has: its labels plus the reserved one.
    max_dim : int, default 50
        The widest embedding this will return.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If ``cardinality`` is not positive.

    Examples
    --------
    >>> embedding_dim_for(5), embedding_dim_for(400)
    (3, 50)
    """
    cardinality = int(cardinality)
    if cardinality <= 0:
        raise ValueError(f"cardinality must be positive, got {cardinality}")
    return max(1, min(int(max_dim), (cardinality + 1) // 2))


def resolve_embedding_dims(
    cardinalities: Sequence[int],
    embedding_dims: Any = None,
    *,
    max_dim: int = DEFAULT_EMBEDDING_MAX_DIM,
    feature_names: Optional[Sequence[str]] = None,
) -> list[int]:
    """Work out each category column's embedding width from the configured setting.

    Parameters
    ----------
    cardinalities : sequence of int
        How many codes each column has.
    embedding_dims : None, "auto", int, sequence or mapping, optional
        ``None`` or ``"auto"`` chooses each width with :func:`embedding_dim_for`; one number applies
        the same width everywhere; a list gives one width per column, in order; a mapping gives one
        per column name, and must name them all.
    max_dim : int, default 50
        The cap used by ``"auto"``.
    feature_names : sequence of str, optional
        The column names, for the mapping form and for the error messages.

    Returns
    -------
    list of int

    Raises
    ------
    ValueError
        If a per-column setting does not cover every column.

    Examples
    --------
    >>> resolve_embedding_dims([5, 400])
    [3, 50]
    >>> resolve_embedding_dims([5, 400], 8)
    [8, 8]
    """
    cardinalities = [int(value) for value in cardinalities]
    names = list(feature_names) if feature_names is not None else [str(i) for i in range(len(cardinalities))]

    if embedding_dims is None or (isinstance(embedding_dims, str) and embedding_dims.lower() == "auto"):
        return [embedding_dim_for(cardinality, max_dim) for cardinality in cardinalities]

    if isinstance(embedding_dims, Mapping):
        resolved = []
        for name, cardinality in zip(names, cardinalities):
            value = embedding_dims.get(name, embedding_dims.get(str(name).lower()))
            if value is None:
                raise ValueError(
                    f"Categorical feature {name!r} has no width in embedding_dims "
                    f"(configured: {sorted(embedding_dims)}). Add an entry for it, or use a scalar "
                    "to apply one width to every feature."
                )
            resolved.append(int(value))
        return resolved

    if isinstance(embedding_dims, (int, float)) and not isinstance(embedding_dims, bool):
        return [int(embedding_dims)] * len(cardinalities)

    resolved = [int(value) for value in embedding_dims]
    if len(resolved) != len(cardinalities):
        raise ValueError(
            f"embedding_dims has {len(resolved)} entry/ies but there are {len(cardinalities)} "
            f"categorical feature(s) ({', '.join(names) or 'none'})"
        )
    return resolved


class EntityEmbeddingBlock(nn.Module):
    """One :term:`embedding` table per category column, joined into a single vector.

    A table each, not one shared table: the columns have unrelated labels, and sharing would make a
    soil texture and a landform class compete for the same rows.

    Parameters
    ----------
    cardinalities : sequence of int
        How many codes each column has.
    embedding_dims : optional
        The widths; see :func:`resolve_embedding_dims`.
    dropout : float, default 0.0
        Dropout applied to the joined vector.
    max_dim : int, default 50
        The cap used when the widths are chosen automatically.
    feature_names : sequence of str, optional
        The column names, used in the error messages.

    Raises
    ------
    ValueError
        If a column has no codes, or there are not as many names as columns.
    """

    def __init__(
        self,
        cardinalities: Sequence[int],
        embedding_dims: Any = None,
        *,
        dropout: float = 0.0,
        max_dim: int = DEFAULT_EMBEDDING_MAX_DIM,
        feature_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.cardinalities = [int(value) for value in cardinalities]
        if any(cardinality <= 0 for cardinality in self.cardinalities):
            raise ValueError(f"Every cardinality must be positive, got {self.cardinalities}")

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"categorical_{index}" for index in range(len(self.cardinalities))]
        )
        if len(self.feature_names) != len(self.cardinalities):
            raise ValueError(
                f"Got {len(self.feature_names)} feature name(s) but {len(self.cardinalities)} cardinality/ies"
            )

        self.embedding_dims = resolve_embedding_dims(
            self.cardinalities, embedding_dims, max_dim=max_dim, feature_names=self.feature_names
        )
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, dim) for cardinality, dim in zip(self.cardinalities, self.embedding_dims)
        )
        # On the joined vector, not inside one table: dropping parts of the whole categorical
        # representation regularizes, while dropping inside one lookup only adds noise to it.
        self.dropout = nn.Dropout(float(dropout))
        self.output_dim = int(sum(self.embedding_dims))

    @property
    def num_features(self) -> int:
        """How many category columns this block embeds."""
        return len(self.embeddings)

    def forward(self, x_categorical: torch.Tensor) -> torch.Tensor:
        """Look up each column's code and join the vectors.

        Parameters
        ----------
        x_categorical : torch.Tensor of shape (batch, n_columns)
            The codes.

        Returns
        -------
        torch.Tensor of shape (batch, sum of the widths)

        Raises
        ------
        ValueError
            If the shape is not the one this block was built for.
        """
        if self.num_features == 0:
            return x_categorical.new_zeros((x_categorical.size(0), 0), dtype=torch.float32)

        if x_categorical.dim() != 2:
            raise ValueError(f"x_categorical must be 2-D (batch, features), got {x_categorical.dim()}-D")
        if x_categorical.size(1) != self.num_features:
            raise ValueError(
                f"x_categorical has {x_categorical.size(1)} column(s) but this block embeds "
                f"{self.num_features} feature(s) ({', '.join(self.feature_names)})"
            )

        # The codes are in range by construction, so they are not checked again here: doing so
        # would cost time on every batch.
        indices = x_categorical.long()
        embedded = [embedding(indices[:, column]) for column, embedding in enumerate(self.embeddings)]
        return self.dropout(torch.cat(embedded, dim=-1))


def _build_continuous_norm(kind: str, num_features: int) -> nn.Module:
    """Build the normalization applied to the numeric covariates: none, batch or layer."""
    key = str(kind).lower()
    if key == "none":
        return nn.Identity()
    if key == "batch":
        return nn.BatchNorm1d(num_features)
    if key == "layer":
        return nn.LayerNorm(num_features)
    raise ValueError(f"continuous_norm must be 'none', 'batch' or 'layer', got {kind!r}")


class TabularStaticEncoder(nn.Module):
    """Summarize a point's covariates: category embeddings joined to the numeric columns.

    Parameters
    ----------
    num_continuous : int
        How many numeric covariates come in, the measured-or-filled flags included.
    hidden_dims : sequence of int
        Widths of the summarizing layers: ``[64]`` is one layer, ``[128, 64]`` two.
    output_dim : int, optional
        Width of a final layer. Left out, the summary is ``hidden_dims[-1]`` wide.
    cardinalities : sequence of int, optional
        How many codes each category column has.
    embedding_dims : optional
        The embedding widths; see :func:`resolve_embedding_dims`.
    embedding_dropout : float, default 0.0
        Dropout on the joined embeddings.
    embedding_max_dim : int, default 50
        Cap on an automatically chosen embedding width.
    feature_names : sequence of str, optional
        The category column names.
    dropout : float, default 0.1
        Dropout in the summarizing layers.
    activation : {"relu", "gelu"}, default "relu"
        The activation function.
    use_layer_norm : bool, default True
        Normalize inside each layer.
    continuous_norm : {"none", "batch", "layer"}, default "none"
        Extra normalization of the numeric columns. They are already standardized by the datamodule,
        so ``"none"`` is the default.
    mlp : bool, default True
        With ``False`` the covariates are passed on unsummarized, one value per column, which is
        what attention :term:`fusion` reads.

    Raises
    ------
    ValueError
        If there are no covariates at all, if ``hidden_dims`` is empty while ``mlp`` is true, or if
        ``output_dim`` is given with ``mlp=False``.
    """

    def __init__(
        self,
        num_continuous: int,
        hidden_dims: Sequence[int],
        *,
        output_dim: Optional[int] = None,
        cardinalities: Sequence[int] = (),
        embedding_dims: Any = None,
        embedding_dropout: float = 0.0,
        embedding_max_dim: int = DEFAULT_EMBEDDING_MAX_DIM,
        feature_names: Optional[Sequence[str]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = True,
        continuous_norm: str = "none",
        mlp: bool = True,
    ):
        super().__init__()
        self.num_continuous = int(num_continuous)
        if self.num_continuous < 0:
            raise ValueError(f"num_continuous must be non-negative, got {num_continuous}")

        self.hidden_dims = [int(width) for width in hidden_dims]
        # mlp=False stops at the joined covariates, which attention fusion splits one per column.
        # The widths are then unused, so an empty list is allowed.
        self.mlp = bool(mlp)
        if self.mlp and not self.hidden_dims:
            raise ValueError(
                "TabularStaticEncoder needs at least one hidden width. A zero-layer static branch "
                "would feed the raw concatenation straight into the fusion, which is a different "
                "model rather than a smaller one."
            )
        self.embeddings = EntityEmbeddingBlock(
            cardinalities,
            embedding_dims,
            dropout=embedding_dropout,
            max_dim=embedding_max_dim,
            feature_names=feature_names,
        )

        self.input_dim = self.num_continuous + self.embeddings.output_dim
        if self.input_dim <= 0:
            raise ValueError(
                "TabularStaticEncoder needs at least one feature: got no continuous columns and no "
                "categorical ones. Callers with no static features should skip this encoder."
            )

        # The numeric columns only. They are already standardized from the training points, so
        # none is the default.
        self.continuous_norm = (
            _build_continuous_norm(continuous_norm, self.num_continuous) if self.num_continuous > 0 else nn.Identity()
        )

        if self.mlp:
            self.encoder = build_mlp_stack(
                self.input_dim,
                self.hidden_dims,
                output_dim,
                dropout=float(dropout),
                use_layer_norm=use_layer_norm,
                activation=activation,
                # This branch feeds the fusion, not a prediction, so every layer is a full one.
                norm_final=True,
                dropout_final=True,
            )
            self.output_dim = self.hidden_dims[-1] if output_dim is None else int(output_dim)
        else:
            if output_dim is not None:
                raise ValueError("output_dim projects the MLP's output, and mlp=False builds no MLP.")
            # A pass-through adds nothing to the saved weights.
            self.encoder = nn.Identity()
            self.output_dim = self.input_dim

    @property
    def embedding_dims(self) -> list[int]:
        """The width of each category column's embedding."""
        return list(self.embeddings.embedding_dims)

    def forward(self, x_static: torch.Tensor, x_categorical: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Summarize one batch of covariates.

        Parameters
        ----------
        x_static : torch.Tensor of shape (batch, num_continuous)
            The standardized numeric covariates.
        x_categorical : torch.Tensor of shape (batch, n_categories), optional
            The category codes; required when the encoder has category columns.

        Returns
        -------
        torch.Tensor of shape (batch, output_dim)

        Raises
        ------
        KeyError
            If the batch carries no category codes but the encoder needs them.
        """
        if self.embeddings.num_features and x_categorical is None:
            raise KeyError(
                f"This encoder embeds {self.embeddings.num_features} categorical feature(s) "
                f"({', '.join(self.embeddings.feature_names)}) but the batch carried no "
                "'x_categorical'"
            )

        embedded = self.embeddings(x_categorical) if self.embeddings.num_features else None
        return self.forward_with_embedding(x_static, embedded)

    def forward_with_embedding(self, x_static: torch.Tensor, embedded: Optional[torch.Tensor]) -> torch.Tensor:
        """Like :meth:`forward`, but given the category vectors instead of their codes.

        Used by the SHAP explanations: a code is looked up, not computed, so contributions cannot be
        traced through it. They are traced through the vectors instead and added up per column.
        """
        parts: list[torch.Tensor] = []
        if self.num_continuous > 0:
            parts.append(self.continuous_norm(x_static))
        if embedded is not None:
            parts.append(embedded.to(dtype=x_static.dtype))

        features = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return self.encoder(features)
