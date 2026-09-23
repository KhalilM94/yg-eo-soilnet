"""Lay each point's readings out as a years-by-months table, and scan it like an image.

The readings arrive at irregular dates, while a convolution needs a regular grid. The rasterizer
here bridges the two: each reading goes into the cell for its year and calendar month, and cells
with no reading are left empty and marked as such - nothing is invented. The encoders then scan that
:term:`calendar grid`, so a step along the columns is a step through the seasons.

Two things follow from how the grid is built. No weight depends on its size, so the same model runs
on three years of readings or twelve; and the rows are counted back from each point's own latest
reading, so shifting every date by a decade changes nothing the model sees.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn

#: Columns in a calendar grid.
MONTHS_PER_YEAR = 12

# The day of the year each month starts on, outside a leap year.
_MONTH_START_DAY = (1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)
_DAYS_PER_YEAR = 365.25


def decimal_year_to_month_index(times: torch.Tensor) -> torch.Tensor:
    """Recover the calendar month of a :term:`decimal year`, 0 for January to 11 for December.

    Not simply the twelfth of the year the date falls in: months are not all the same length, and
    that would put 1 November in October. The day of the year is worked out instead and the month
    looked up, with the leap day accounted for.

    Parameters
    ----------
    times : torch.Tensor
        Dates as decimal years.

    Returns
    -------
    torch.Tensor of int
        The month of each, 0 to 11.

    Examples
    --------
    >>> import pandas as pd, torch
    >>> from yg_eo_soilnet.datamodules.sequence.sequence_builder import to_decimal_year
    >>> dates = to_decimal_year(pd.Series(["2021-01-15", "2021-07-02", "2021-11-01", "2021-12-31"]))
    >>> decimal_year_to_month_index(torch.tensor(dates))
    tensor([ 0,  6, 10, 11])
    """
    times = times.to(dtype=torch.float64)
    year = torch.floor(times)
    day_of_year = (times - year) * _DAYS_PER_YEAR + 1.0

    starts = torch.tensor(_MONTH_START_DAY, dtype=torch.float64, device=times.device)
    is_leap = (((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)).to(dtype=torch.float64)
    leap_shift = torch.zeros(MONTHS_PER_YEAR, dtype=torch.float64, device=times.device)
    leap_shift[2:] = 1.0  # March onward starts a day later once February has 29 days
    starts = starts + is_leap.unsqueeze(-1) * leap_shift

    # The small tolerance matters: dividing by 365.25 and multiplying back leaves 1 February at
    # 31.999999..., which would fall in January. Real day numbers are whole, so the slack is safe.
    month = ((day_of_year.unsqueeze(-1) + 1e-6) >= starts).sum(dim=-1) - 1
    return month.clamp(0, MONTHS_PER_YEAR - 1).to(dtype=torch.long)


class CalendarGridRasterizer(nn.Module):
    """Put dated readings into a :term:`calendar grid`: one row per year, one column per month.

    The rows are counted back from each point's **own** latest reading, so the grid is that point's
    recent history rather than a fixed stretch of the calendar. The columns are calendar months, so
    a step along them is a step through the seasons.

    The grid carries, in this order: the readings themselves, one channel each; the
    :term:`validity flags <validity flag>`, saying which were measured rather than filled in; one
    channel saying whether the cell holds a reading at all; and, optionally, two channels naming the
    month, without which the model could learn the shape of a seasonal change but never which column
    is January.

    Parameters
    ----------
    num_channels : int
        How many readings each cell holds - the :term:`data source`\'s columns.
    grid_years : int, optional
        How many years the grid spans. Left out, each batch uses its own longest history.
    use_validity_channels : bool, default True
        Carry the measured-or-filled flags.
    month_positional : bool, default True
        Carry the two channels naming the month.

    Examples
    --------
    >>> import torch
    >>> rasterizer = CalendarGridRasterizer(num_channels=2, grid_years=3)
    >>> rasterizer.output_channels     # 2 readings + 2 flags + 1 observed + 2 month
    7
    >>> readings = torch.ones(1, 2, 2)                 # one point, two dates, two channels
    >>> dates = torch.tensor([[2021.0, 2021.5]])
    >>> seen = torch.ones(1, 2, dtype=torch.bool)
    >>> grid, cell_mask = rasterizer(readings, seen, dates)
    >>> grid.shape, int(cell_mask.sum())               # two cells filled, the rest empty
    (torch.Size([1, 7, 3, 12]), 2)
    """

    def __init__(
        self,
        num_channels: int,
        grid_years: Optional[int] = None,
        *,
        use_validity_channels: bool = True,
        month_positional: bool = True,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.grid_years = None if grid_years in (None, 0) else max(1, int(grid_years))
        self.use_validity_channels = bool(use_validity_channels)
        self.month_positional = bool(month_positional)

        if self.month_positional:
            months = torch.arange(MONTHS_PER_YEAR, dtype=torch.float32)
            angle = 2.0 * math.pi * months / MONTHS_PER_YEAR
            self.register_buffer("month_sin", torch.sin(angle), persistent=False)
            self.register_buffer("month_cos", torch.cos(angle), persistent=False)

    @property
    def output_channels(self) -> int:
        """How many channels the grid has, flags and month channels included."""
        channels = self.num_channels + 1
        if self.use_validity_channels:
            channels += self.num_channels
        if self.month_positional:
            channels += 2
        return channels

    def channel_layout(self) -> dict[str, list[int]]:
        """Which channel carries what, in the order :meth:`forward` writes them.

        Read by the SHAP figures, which would otherwise credit one band's importance to another.

        Returns
        -------
        dict of str to list of int
            Channel numbers under ``"values"``, ``"validity"``, ``"cell_observed"`` and
            ``"month_positional"``.

        Examples
        --------
        >>> CalendarGridRasterizer(num_channels=2, grid_years=3).channel_layout()
        {'values': [0, 1], 'validity': [2, 3], 'cell_observed': [4], 'month_positional': [5, 6]}
        """
        layout: dict[str, list[int]] = {"values": list(range(self.num_channels))}
        cursor = self.num_channels

        if self.use_validity_channels:
            layout["validity"] = list(range(cursor, cursor + self.num_channels))
            cursor += self.num_channels
        else:
            layout["validity"] = []

        layout["cell_observed"] = [cursor]
        cursor += 1

        layout["month_positional"] = list(range(cursor, cursor + 2)) if self.month_positional else []
        return layout

    def resolve_grid_years(self, times: torch.Tensor, mask: torch.Tensor) -> int:
        """How many years the grid spans: the configured number, or this batch's longest history."""
        if self.grid_years is not None:
            return self.grid_years
        if not bool(mask.any()):
            return 1
        years = torch.floor(times.to(dtype=torch.float64))
        largest = years.masked_fill(~mask, float("-inf")).max(dim=1).values
        smallest = years.masked_fill(~mask, float("inf")).min(dim=1).values
        spans = (largest - smallest + 1.0)[mask.any(dim=1)]
        return max(1, int(spans.max().item()))

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        validity: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lay a batch of readings out on the grid.

        Parameters
        ----------
        values : torch.Tensor of shape (batch, n_readings, n_channels)
            The standardized readings, padded to the batch's longest series.
        mask : torch.Tensor of shape (batch, n_readings)
            True where a reading is real rather than padding.
        times : torch.Tensor of shape (batch, n_readings)
            Their dates, as decimal years.
        validity : torch.Tensor, optional
            True where a reading was measured rather than filled in.

        Returns
        -------
        grid : torch.Tensor of shape (batch, output_channels, years, 12)
        cell_mask : torch.Tensor of shape (batch, years, 12)
            True for the cells that hold a reading.

        Raises
        ------
        ValueError
            If the readings have a different number of channels than this rasterizer was built for.
        """
        batch_size, length, channels = values.shape
        if channels != self.num_channels:
            raise ValueError(
                f"CalendarGridRasterizer was built for {self.num_channels} channel(s) but received {channels}"
            )
        mask = mask.to(dtype=torch.bool)
        device = values.device
        grid_years = self.resolve_grid_years(times, mask)
        cells = grid_years * MONTHS_PER_YEAR

        if validity is None:
            validity = torch.ones_like(values, dtype=torch.bool)
        validity = validity.to(dtype=torch.bool) & mask.unsqueeze(-1)

        years = torch.floor(times.to(dtype=torch.float64))
        latest = years.masked_fill(~mask, float("-inf")).max(dim=1, keepdim=True).values
        latest = torch.where(mask.any(dim=1, keepdim=True), latest, torch.zeros_like(latest))

        row = (years - latest).to(dtype=torch.long) + (grid_years - 1)
        column = decimal_year_to_month_index(times)
        flat = row * MONTHS_PER_YEAR + column

        # Readings older than the grid, and padding, go to a spare bin that is cut off below.
        # Squeezing them into a real cell would invent readings.
        keep = mask & (row >= 0) & (row < grid_years)
        flat = torch.where(keep, flat, torch.full_like(flat, cells))

        index = flat.unsqueeze(1).expand(-1, channels, -1)
        source = (values * keep.unsqueeze(-1)).transpose(1, 2)

        totals = torch.zeros(batch_size, channels, cells + 1, device=device, dtype=values.dtype)
        totals.scatter_add_(2, index, source)

        counts = torch.zeros(batch_size, 1, cells + 1, device=device, dtype=values.dtype)
        counts.scatter_add_(2, flat.unsqueeze(1), keep.unsqueeze(1).to(dtype=values.dtype))

        valid_totals = torch.zeros(batch_size, channels, cells + 1, device=device, dtype=values.dtype)
        valid_totals.scatter_add_(2, index, (validity & keep.unsqueeze(-1)).to(dtype=values.dtype).transpose(1, 2))

        totals, counts, valid_totals = totals[..., :cells], counts[..., :cells], valid_totals[..., :cells]

        # Two readings in one month average. Monthly data never hits this, but daily data would.
        occupancy = counts.clamp_min(1.0)
        readings = totals / occupancy
        cell_mask = counts.squeeze(1) > 0

        layers = [readings]
        if self.use_validity_channels:
            layers.append(valid_totals / occupancy)
        layers.append(cell_mask.unsqueeze(1).to(dtype=values.dtype))

        grid = torch.cat(layers, dim=1).view(batch_size, -1, grid_years, MONTHS_PER_YEAR)

        if self.month_positional:
            positional = torch.stack([self.month_sin, self.month_cos]).to(dtype=grid.dtype, device=device)
            positional = positional.view(1, 2, 1, MONTHS_PER_YEAR).expand(batch_size, -1, grid_years, -1)
            grid = torch.cat([grid, positional], dim=1)

        return grid, cell_mask.view(batch_size, grid_years, MONTHS_PER_YEAR)


def masked_global_pool(features: torch.Tensor, cell_mask: torch.Tensor, *, masked: bool = True) -> torch.Tensor:
    """Average each channel over the grid, giving one number per channel.

    With ``masked``, only the cells holding a reading are averaged. Averaging over the whole grid
    instead would scale a point's summary by how full its grid is, pulling every prediction for a
    sparsely observed point towards the average - exactly the points that are hardest already.

    Parameters
    ----------
    features : torch.Tensor of shape (batch, channels, ...)
        What the convolutions produced.
    cell_mask : torch.Tensor
        True for the cells holding a reading.
    masked : bool, default True
        Average over those cells only.

    Returns
    -------
    torch.Tensor of shape (batch, channels)
    """
    spatial_dims = tuple(range(2, features.ndim))
    if not masked:
        return features.mean(dim=spatial_dims)

    weights = cell_mask.to(dtype=features.dtype).unsqueeze(1)
    occupied = weights.sum(dim=spatial_dims)
    pooled = (features * weights).sum(dim=spatial_dims) / occupied.clamp_min(1.0)
    return pooled * (occupied > 0).to(dtype=features.dtype)


def _build_norm(kind: str, num_features: int, dims: int) -> nn.Module:
    """Build the normalization between convolutions: none, group or batch."""
    kind = str(kind).lower()
    if kind == "none":
        return nn.Identity()
    if kind == "group":
        # Group normalization works one point at a time, so the empty cells of other points in the
        # batch cannot affect it.
        return nn.GroupNorm(num_groups=math.gcd(8, num_features) or 1, num_channels=num_features)
    if kind == "batch":
        return nn.BatchNorm1d(num_features) if dims == 1 else nn.BatchNorm2d(num_features)
    raise ValueError(f"norm must be 'batch', 'group' or 'none', got {kind!r}")


class _MaskedConvStack(nn.Module):
    """Convolutions with the empty cells cleared again after each one.

    Without this a gap does not stay neutral: every convolution adds a constant, so an empty cell
    starts emitting a value its neighbours read as a reading on the next layer. Clearing them keeps
    a missing month missing, and keeps the answer the same whether or not the grid has spare years.
    """

    def __init__(self, stages: list[nn.Module], mask_between: bool = True):
        """Hold the convolution stages; ``mask_between`` clears the empty cells between them."""
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.mask_between = bool(mask_between)

    def forward(self, features: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        """Run every stage, clearing the empty cells before and after each."""
        weights = cell_mask.unsqueeze(1).to(dtype=features.dtype)
        if self.mask_between:
            features = features * weights
        for stage in self.stages:
            features = stage(features)
            if self.mask_between:
                features = features * weights
        return features


def _validate_hidden_dims(hidden_dims: Sequence[int], owner: str) -> list[int]:
    """Check the layer widths: one per block, all positive, at least one.

    The list is the stack: ``[64, 128]`` is two blocks, the second wider than the first.
    """
    widths = [int(width) for width in hidden_dims]
    if not widths:
        raise ValueError(f"{owner} needs at least one hidden width; got an empty hidden_dims.")
    if any(width <= 0 for width in widths):
        raise ValueError(f"{owner} hidden_dims must all be positive, got {widths}.")
    return widths


def _conv_stages(conv_factory, hidden_dim: int, norm: str, dropout: float, dims: int) -> list[nn.Module]:
    """Build one block: two convolutions, each with normalization and an activation."""
    first, second = conv_factory()
    return [
        nn.Sequential(first, _build_norm(norm, hidden_dim, dims=dims), nn.GELU(), nn.Dropout(dropout)),
        nn.Sequential(second, _build_norm(norm, hidden_dim, dims=dims), nn.GELU()),
    ]


class DilatedTempCNNEncoder(nn.Module):
    """Read the grid as one long line of months, with a stride that links year to year.

    ``dilated_tempcnn``, the default. Each block has two convolutions: the first looks at
    neighbouring months, the second at the same month one year before and one year after, so a
    single weight can compare this July with last July.

    Parameters
    ----------
    num_channels : int
        Channels coming in, from :class:`CalendarGridRasterizer`.
    output_dim : int
        Width of the summary this branch produces.
    hidden_dims : sequence of int, default (32,)
        One width per block.
    dropout : float, default 0.1
        Dropout inside each block.
    norm : {"batch", "group", "none"}, default "batch"
        Normalization between convolutions. ``"group"`` suits grids with many empty cells.
    pool : {"masked_avg", "avg"}, default "masked_avg"
        Whether the final average covers only the cells holding a reading.
    mask_between_blocks : bool, default True
        Clear the empty cells between blocks.

    Raises
    ------
    ValueError
        If ``pool`` is unknown or ``hidden_dims`` is empty.
    """

    def __init__(
        self,
        num_channels: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
        norm: str = "batch",
        pool: str = "masked_avg",
        mask_between_blocks: bool = True,
    ):
        super().__init__()
        self.pool = str(pool).lower()
        if self.pool not in {"masked_avg", "avg"}:
            raise ValueError(f"pool must be 'masked_avg' or 'avg', got {pool!r}")

        hidden_dims = _validate_hidden_dims(hidden_dims, type(self).__name__)
        stages: list[nn.Module] = []
        in_channels = int(num_channels)
        for width in hidden_dims:
            stages += _conv_stages(
                lambda c=in_channels, w=width: (
                    # Neighbouring months: the trend within a season.
                    nn.Conv1d(c, w, kernel_size=3, padding=1, dilation=1),
                    # The same month a year earlier and a year later.
                    nn.Conv1d(
                        w,
                        w,
                        kernel_size=3,
                        padding=MONTHS_PER_YEAR,
                        dilation=MONTHS_PER_YEAR,
                    ),
                ),
                width,
                norm,
                dropout,
                dims=1,
            )
            in_channels = width
        self.hidden_dims = hidden_dims
        self.blocks = _MaskedConvStack(stages, mask_between=mask_between_blocks)
        self.projection = nn.Linear(hidden_dims[-1], int(output_dim))
        self.output_dim = int(output_dim)

    def forward(self, grid: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        """Summarize one :term:`calendar grid` per point into ``output_dim`` values."""
        batch_size = grid.size(0)
        sequence = grid.reshape(batch_size, grid.size(1), -1)
        flat_mask = cell_mask.reshape(batch_size, -1)
        features = self.blocks(sequence, flat_mask)
        pooled = masked_global_pool(features, flat_mask, masked=self.pool == "masked_avg")
        return self.projection(pooled)


class AnnualGrid2DEncoder(nn.Module):
    """Read the grid as a two-dimensional image: months across, years down.

    ``annual_grid2d``. Each block has two convolutions: one moves along the months within a year, so
    it can only learn the shape of a season; the other moves down the years at a fixed month, so it
    can only learn how that month changes from year to year.

    Parameters
    ----------
    num_channels : int
        Channels coming in, from :class:`CalendarGridRasterizer`.
    output_dim : int
        Width of the summary this branch produces.
    hidden_dims : sequence of int, default (32,)
        One width per block.
    dropout : float, default 0.1
        Dropout inside each block.
    norm : {"batch", "group", "none"}, default "batch"
        Normalization between convolutions.
    pool : {"masked_avg", "avg"}, default "masked_avg"
        Whether the final average covers only the cells holding a reading.
    mask_between_blocks : bool, default True
        Clear the empty cells between blocks.

    Raises
    ------
    ValueError
        If ``pool`` is unknown or ``hidden_dims`` is empty.
    """

    def __init__(
        self,
        num_channels: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
        norm: str = "batch",
        pool: str = "masked_avg",
        mask_between_blocks: bool = True,
    ):
        super().__init__()
        self.pool = str(pool).lower()
        if self.pool not in {"masked_avg", "avg"}:
            raise ValueError(f"pool must be 'masked_avg' or 'avg', got {pool!r}")

        hidden_dims = _validate_hidden_dims(hidden_dims, type(self).__name__)
        stages: list[nn.Module] = []
        in_channels = int(num_channels)
        for width in hidden_dims:
            stages += _conv_stages(
                lambda c=in_channels, w=width: (
                    # Along the months of one year: the shape of a season.
                    nn.Conv2d(c, w, kernel_size=(1, 3), padding=(0, 1)),
                    # Down the years at one month: how that month changes.
                    nn.Conv2d(w, w, kernel_size=(3, 1), padding=(1, 0)),
                ),
                width,
                norm,
                dropout,
                dims=2,
            )
            in_channels = width
        self.hidden_dims = hidden_dims
        self.blocks = _MaskedConvStack(stages, mask_between=mask_between_blocks)
        self.projection = nn.Linear(hidden_dims[-1], int(output_dim))
        self.output_dim = int(output_dim)

    def forward(self, grid: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        """Summarize one :term:`calendar grid` per point into ``output_dim`` values."""
        features = self.blocks(grid, cell_mask)
        pooled = masked_global_pool(features, cell_mask, masked=self.pool == "masked_avg")
        return self.projection(pooled)


class ConcatGatedFusion(nn.Module):
    """Gated :term:`fusion`: join the branches, then scale each value by a learned weight.

    ``fusion: gated``, the default. The branch summaries are put side by side and every value is
    multiplied by a weight between 0 and 1 that the model computes from all of them together. So a
    covariate can be turned down where the time series says more, and the location - when it is one
    of the branches - can decide how much of either is let through: a reflectance band means one
    thing on an irrigated plain and another on a limestone slope.

    Parameters
    ----------
    static_dim : int
        Width of the covariate summary.
    temporal_dim : int
        Width of the time-series summary.
    coordinate_dim : int, default 0
        Width of the location summary; 0 leaves that branch out.
    """

    def __init__(self, static_dim: int, temporal_dim: int, coordinate_dim: int = 0):
        super().__init__()
        self.output_dim = int(static_dim) + int(temporal_dim) + int(coordinate_dim)
        self.gate = nn.Linear(self.output_dim, self.output_dim)

    def forward(
        self,
        static_features: torch.Tensor,
        temporal_features: Optional[torch.Tensor],
        coordinate_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combine the branch summaries into one vector of width ``output_dim``."""
        parts = [part for part in (static_features, temporal_features, coordinate_features) if part is not None]
        joined = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return joined * torch.sigmoid(self.gate(joined))


class AttentionFusion(nn.Module):
    """Attention :term:`fusion`: let each branch summary be re-read in light of the others.

    ``fusion: attention``. Each branch summary becomes a "token" - a vector of a common width - and a
    small transformer lets the tokens read one another, so the Sentinel-2 summary can be rebuilt
    knowing what the climate summary and the covariates say. The gated fusion can only turn a value
    up or down; this can change what it means.

    Interchangeable with :class:`ConcatGatedFusion`: the model calls either without knowing which.

    Parameters
    ----------
    static_dims : sequence of int
        How the covariate summary is cut into tokens. ``[64]`` makes it one token; a width of 1 per
        column makes every covariate its own token.
    temporal_dims : sequence of int
        The width of each :term:`data source`\'s summary, one token each.
    coordinate_dim : int, default 0
        Width of the location summary; 0 leaves it out.
    d_model : int, default 64
        Width every token is projected to.
    nhead : int, default 4
        How many attention heads; must divide ``d_model``.
    num_layers : int, default 2
        How many transformer layers.
    ff_multiplier : int, default 2
        Width of each layer's internal step, as a multiple of ``d_model``.
    dropout : float, default 0.1
        Dropout inside the transformer.
    readout : {"cls", "mean", "flatten"}, default "cls"
        What is passed on: a summary token, the average of the tokens, or all of them side by side.

    Raises
    ------
    ValueError
        If ``d_model`` is not divisible by ``nhead``, ``readout`` is unknown, or there are no tokens.
    """

    #: The ways the transformer's output can be read.
    READOUTS = ("cls", "mean", "flatten")

    def __init__(
        self,
        static_dims: Sequence[int],
        temporal_dims: Sequence[int],
        coordinate_dim: int = 0,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff_multiplier: int = 2,
        dropout: float = 0.1,
        readout: str = "cls",
    ):
        super().__init__()
        d_model, nhead = int(d_model), int(nhead)
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead for multi-head attention; got d_model={d_model} and nhead={nhead}"
            )
        if int(ff_multiplier) < 1:
            raise ValueError(f"ff_multiplier must be at least 1, got {ff_multiplier}")
        self.readout = str(readout).lower()
        if self.readout not in self.READOUTS:
            raise ValueError(f"readout must be one of {list(self.READOUTS)}, got {readout!r}")

        self.static_dims = [int(width) for width in static_dims]
        self.temporal_dims = [int(width) for width in temporal_dims]
        self.coordinate_dim = int(coordinate_dim or 0)
        widths = self.static_dims + self.temporal_dims + ([self.coordinate_dim] if self.coordinate_dim > 0 else [])
        if not widths:
            raise ValueError(
                "AttentionFusion needs at least one input token; got no static, temporal or coordinate chunks."
            )
        if any(width <= 0 for width in widths):
            raise ValueError(f"AttentionFusion chunk widths must all be positive, got {widths}")

        self.d_model = d_model
        self.num_tokens = len(widths)
        self.tokenizers = nn.ModuleList(nn.Linear(width, d_model) for width in widths)
        # Attention has no sense of order, so each token carries a learned marker saying which
        # branch it came from.
        self.token_type = nn.Parameter(torch.zeros(self.num_tokens, d_model))
        nn.init.normal_(self.token_type, std=0.02)
        if self.readout == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        # A standard transformer layer, plus a closing normalization, since what comes out feeds
        # the prediction head rather than another layer.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(ff_multiplier) * d_model,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        # That optimization does not apply to this kind of layer and would only warn.
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=max(1, int(num_layers)),
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.output_dim = d_model * self.num_tokens if self.readout == "flatten" else d_model

    @staticmethod
    def _split(features: Optional[torch.Tensor], dims: list[int], label: str) -> list[torch.Tensor]:
        """Cut one branch's summary into its tokens, refusing a width it was not built for."""
        if not dims:
            return []
        if features is None:
            raise ValueError(f"AttentionFusion was built with {label} chunks {dims} but received no {label} features")
        if features.size(-1) != sum(dims):
            # A wrong width would still split, leaving every token after the first reading its
            # neighbour's values.
            raise ValueError(
                f"AttentionFusion expected {sum(dims)} {label} feature(s) (chunks {dims}), got {features.size(-1)}"
            )
        return list(torch.split(features, dims, dim=-1))

    def forward(
        self,
        static_features: torch.Tensor,
        temporal_features: Optional[torch.Tensor],
        coordinate_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combine the branch summaries into one vector of width ``output_dim``."""
        chunks = self._split(static_features, self.static_dims, "static")
        chunks += self._split(temporal_features, self.temporal_dims, "temporal")
        if self.coordinate_dim > 0:
            chunks += self._split(coordinate_features, [self.coordinate_dim], "coordinate")

        tokens = (
            torch.stack([tokenizer(chunk) for tokenizer, chunk in zip(self.tokenizers, chunks)], dim=1)
            + self.token_type
        )
        if self.readout == "cls":
            tokens = torch.cat([self.cls_token.expand(tokens.size(0), -1, -1), tokens], dim=1)

        encoded = self.encoder(tokens)
        if self.readout == "cls":
            return encoded[:, 0]
        if self.readout == "mean":
            return encoded.mean(dim=1)
        return encoded.flatten(1)
