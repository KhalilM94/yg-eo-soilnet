"""The branch that turns a point's coordinates into something the model can use.

It receives coordinates already placed on a -1 to 1 range, measured against the area the training
points cover (see
:meth:`SoilSequenceDataModule._normalize_coords
<yg_eo_soilnet.datamodules.sequence.sequence_datamodule.SoilSequenceDataModule>`), and turns each
into a set of waves of different wavelengths. Nearby points then differ in the short waves and
distant ones in the long waves, which a network can read far better than two raw numbers.

Unlike the dates, location is deliberately absolute: where a point is *is* the signal, so the area
the model was trained on is saved with it.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn

from yg_eo_soilnet.models.lightningmodules.mlp import build_mlp_stack


class HarmonicPositionEncoder(nn.Module):
    """Turn normalized coordinates into waves of several wavelengths.

    Each frequency is twice the one before, so the first spans the whole study area and the last
    about a ``2**num_frequencies``-th of it. The long waves let the model express a regional trend,
    the short ones let it tell neighbouring points apart. Because the waves are measured against the
    study area rather than in kilometres, the same setting means the same level of detail on a
    province and on a continent.

    Parameters
    ----------
    num_coordinates : int, default 2
        How many coordinates come in - latitude and longitude.
    num_frequencies : int, default 6
        How many wavelengths, each half the previous one. More means finer detail.
    include_input : bool, default True
        Also pass the coordinates themselves through, which carries a plain north-south or
        east-west gradient.
    hidden_dims : sequence of int, optional
        Widths of the layers summarizing the waves. Empty passes them on as they are.
    dropout : float, default 0.0
        Dropout in those layers.

    Raises
    ------
    ValueError
        If ``num_coordinates`` or ``num_frequencies`` is not positive. A model with no coordinates
        leaves this branch out entirely (``USE_HARMONIC_COORDS: false``).

    Examples
    --------
    >>> import torch
    >>> encoder = HarmonicPositionEncoder(num_frequencies=3)
    >>> encoder(torch.zeros(4, 2)).shape       # 4 points, 2 + 3 x 4 channels
    torch.Size([4, 14])
    """

    def __init__(
        self,
        num_coordinates: int = 2,
        num_frequencies: int = 6,
        include_input: bool = True,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_coordinates = int(num_coordinates)
        if self.num_coordinates <= 0:
            raise ValueError(
                f"num_coordinates must be positive, got {num_coordinates}. Callers with no "
                "coordinates should skip this encoder entirely rather than build a zero-wide one."
            )
        self.num_frequencies = int(num_frequencies)
        if self.num_frequencies <= 0:
            raise ValueError(
                f"num_frequencies must be positive, got {num_frequencies}. A bank with no "
                "frequencies encodes nothing; switch the branch off with USE_HARMONIC_COORDS "
                "instead, which also stops the coordinates being carried at all."
            )
        self.include_input = bool(include_input)

        # Not saved with the weights: the wavelengths follow from num_frequencies, which is, so a
        # checkpoint stays loadable if the way they are worked out ever changes.
        exponents = torch.arange(self.num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", (2.0**exponents) * math.pi, persistent=False)

        self.embedding_dim = self.num_coordinates * (
            2 * self.num_frequencies + (1 if self.include_input else 0)
        )
        hidden_dims = [int(width) for width in (hidden_dims or [])]
        self.hidden_dims = hidden_dims
        # No widths makes this a pass-through, so there is one code path either way.
        self.projection = build_mlp_stack(
            self.embedding_dim,
            hidden_dims,
            None,
            dropout=float(dropout),
            activation="gelu",
            use_layer_norm=True,
            # This block feeds the fusion, not a prediction, so both are on.
            norm_final=True,
            dropout_final=True,
        )
        self.output_dim = hidden_dims[-1] if hidden_dims else self.embedding_dim

    def channel_layout(self) -> dict[str, list[int]]:
        """Which output channel carries what, before the summarizing layers.

        Returns
        -------
        dict of str to list of int
            Channel numbers under ``"input"``, ``"sin"`` and ``"cos"``. Read by the SHAP figures,
            which is why the order is stated here rather than guessed.
        """
        layout: dict[str, list[int]] = {"input": [], "sin": [], "cos": []}
        cursor = 0
        if self.include_input:
            layout["input"] = list(range(cursor, cursor + self.num_coordinates))
            cursor += self.num_coordinates
        for _ in range(self.num_frequencies):
            for _ in range(self.num_coordinates):
                layout["sin"].append(cursor)
                layout["cos"].append(cursor + 1)
                cursor += 2
        return layout

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode a batch of coordinates.

        Parameters
        ----------
        coords : torch.Tensor of shape (batch, num_coordinates)
            Normalized coordinates.

        Returns
        -------
        torch.Tensor of shape (batch, output_dim)

        Raises
        ------
        ValueError
            If the shape is not the one this encoder was built for.
        """
        if coords.dim() != 2:
            raise ValueError(f"coords must be 2-D (batch, coordinates), got {coords.dim()}-D")
        if coords.size(-1) != self.num_coordinates:
            raise ValueError(
                f"coords has {coords.size(-1)} column(s) but this encoder was built for "
                f"{self.num_coordinates}"
            )

        parts: list[torch.Tensor] = [coords] if self.include_input else []
        # Every coordinate against every wavelength, sine and cosine interleaved so the channel
        # order matches channel_layout above.
        angles = coords.unsqueeze(-1) * self.frequencies.to(dtype=coords.dtype)
        for index in range(self.num_frequencies):
            angle = angles[..., index]
            parts.append(torch.stack([angle.sin(), angle.cos()], dim=-1).flatten(1))
        return self.projection(torch.cat(parts, dim=-1))
