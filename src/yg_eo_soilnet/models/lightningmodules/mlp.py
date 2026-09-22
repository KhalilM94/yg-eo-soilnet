"""The stack of layers every branch of the deep-learning model is built from.

The covariate branch, the prediction head and every other block have the same shape - one layer per
width, each optionally followed by normalization, an activation and dropout - so it is built here
once. The widths are always given as an explicit list, so the network can be read off the
configuration: ``head_hidden_dims: [64, 32]`` is two layers, 64 units then 32.
"""

from __future__ import annotations

from typing import Optional, Sequence

from torch import nn

#: The activation functions a configuration may name.
ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


def build_mlp_stack(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: Optional[int] = None,
    *,
    dropout: float,
    use_layer_norm: bool = True,
    activation: str = "relu",
    norm_final: bool = False,
    dropout_final: bool = False,
) -> nn.Module:
    """Build a stack of layers: ``input_dim`` in, one block per width, optionally a final output.

    Each block is a layer of that width, then normalization, the activation and dropout.

    Parameters
    ----------
    input_dim : int
        How many values come in.
    hidden_dims : sequence of int
        One width per layer. Empty gives a single layer from ``input_dim`` to ``output_dim``, or
        nothing at all when there is no ``output_dim``.
    output_dim : int, optional
        Width of a final layer. Left out, the stack ends ``hidden_dims[-1]`` wide.
    dropout : float
        Share of values dropped during training, between 0 and 1.
    use_layer_norm : bool, default True
        Normalize inside each block.
    activation : {"relu", "gelu"}, default "relu"
        The activation function.
    norm_final, dropout_final : bool, default False
        Whether the last block is normalized and dropped out too. A prediction head leaves both off:
        normalizing there would throw away the size of the values the last layer reads, which is
        what a prediction needs. A block feeding into the :term:`fusion` turns both on, since no
        prediction is read from it directly.

    Returns
    -------
    torch.nn.Module

    Raises
    ------
    ValueError
        If the activation is not one of :data:`ACTIVATIONS`.

    Examples
    --------
    >>> stack = build_mlp_stack(8, [32, 16], output_dim=3, dropout=0.1)
    >>> [type(layer).__name__ for layer in stack][-1]
    'Linear'
    """
    try:
        activation_cls = ACTIVATIONS[str(activation).lower()]
    except KeyError:
        raise ValueError(
            f"Unknown activation {activation!r}; expected one of: {', '.join(sorted(ACTIVATIONS))}."
        ) from None

    hidden_dims = [int(width) for width in hidden_dims]
    if not hidden_dims:
        return nn.Identity() if output_dim is None else nn.Linear(input_dim, int(output_dim))

    layers: list[nn.Module] = []
    dim = input_dim
    for index, width in enumerate(hidden_dims):
        layers.append(nn.Linear(dim, width))
        is_last_block = index == len(hidden_dims) - 1
        if use_layer_norm and (not is_last_block or norm_final):
            layers.append(nn.LayerNorm(width))
        layers.append(activation_cls())
        if not is_last_block or dropout_final:
            layers.append(nn.Dropout(dropout))
        dim = width
    if output_dim is not None:
        layers.append(nn.Linear(dim, int(output_dim)))
    return nn.Sequential(*layers)
