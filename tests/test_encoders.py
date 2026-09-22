"""The building blocks under soil_cnn: the MLP stack, the calendar grid and its temporal encoders,
AttentionFusion, and the entity-embedding and static encoders.

The shared regression head, which replaced four near-identical copies.

Each model used to build its own, and two of them took a `head_num_layers` + halving `head_hidden_dim`
+ `head_min_hidden_dim` triple rather than a width list - so the built network could not be read off
the config, and every registry comment describing one had drifted from what it actually built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from yg_eo_soilnet.datamodules.sequence.sequence_builder import to_decimal_year
from yg_eo_soilnet.models.lightningmodules.mlp import build_mlp_stack
from yg_eo_soilnet.models.lightningmodules.tabular_encoders import (
    EntityEmbeddingBlock,
    TabularStaticEncoder,
)
from yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders import (
    AnnualGrid2DEncoder,
    AttentionFusion,
    CalendarGridRasterizer,
    ConcatGatedFusion,
    DilatedTempCNNEncoder,
    decimal_year_to_month_index,
    masked_global_pool,
)


def _widths(head: nn.Module) -> list[int]:
    return [layer.out_features for layer in head.modules() if isinstance(layer, nn.Linear)]


def test_an_empty_width_list_is_a_single_linear_readout() -> None:
    head = build_mlp_stack(16, [], 1, dropout=0.1)

    assert isinstance(head, nn.Linear)
    assert (head.in_features, head.out_features) == (16, 1)


def test_widths_are_built_exactly_as_listed() -> None:
    head = build_mlp_stack(16, [64, 32, 32], 2, dropout=0.1)

    assert _widths(head) == [64, 32, 32, 2]


def test_the_block_feeding_the_readout_carries_no_norm_and_no_dropout() -> None:
    """Normalizing there leaves the final Linear only a direction; magnitude reaches the tails."""
    modules = list(build_mlp_stack(16, [64, 32], 1, dropout=0.1))
    last_hidden_linear = max(i for i, m in enumerate(modules[:-1]) if isinstance(m, nn.Linear))
    tail = modules[last_hidden_linear + 1 : -1]

    assert not any(isinstance(m, (nn.LayerNorm, nn.Dropout)) for m in tail)
    assert any(isinstance(m, nn.LayerNorm) for m in modules[:last_hidden_linear])
    assert any(isinstance(m, nn.Dropout) for m in modules[:last_hidden_linear])


def test_norm_final_restores_the_norm_on_that_block_but_not_the_dropout() -> None:
    modules = list(build_mlp_stack(16, [64, 32], 1, dropout=0.1, norm_final=True))
    last_hidden_linear = max(i for i, m in enumerate(modules[:-1]) if isinstance(m, nn.Linear))
    tail = modules[last_hidden_linear + 1 : -1]

    assert any(isinstance(m, nn.LayerNorm) for m in tail)
    assert not any(isinstance(m, nn.Dropout) for m in tail)


def test_use_layer_norm_false_removes_every_norm() -> None:
    head = build_mlp_stack(16, [64, 32], 1, dropout=0.1, use_layer_norm=False, norm_final=True)

    assert not any(isinstance(m, nn.LayerNorm) for m in head.modules())


@pytest.mark.parametrize("activation,expected", [("relu", nn.ReLU), ("gelu", nn.GELU), ("GELU", nn.GELU)])
def test_the_activation_is_selectable(activation: str, expected: type) -> None:
    head = build_mlp_stack(16, [64], 1, dropout=0.1, activation=activation)

    assert any(isinstance(m, expected) for m in head.modules())


def test_an_unknown_activation_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="Unknown activation 'swish'"):
        build_mlp_stack(16, [64], 1, dropout=0.1, activation="swish")


def test_the_cnn_head_layout_is_reproduced() -> None:
    """LayerNorm + GELU per block, dropout on every block but the last, then the readout."""
    head = build_mlp_stack(10, [64], 1, dropout=0.2, activation="gelu", norm_final=False)

    assert [type(module) for module in head] == [nn.Linear, nn.GELU, nn.Linear]


def test_the_default_layout_norms_and_drops_only_the_first_blocks() -> None:
    head = build_mlp_stack(10, [64, 32], 1, dropout=0.1)

    assert [type(module) for module in head] == [
        nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout, nn.Linear, nn.ReLU, nn.Linear
    ]


def test_widths_given_as_strings_or_floats_are_coerced() -> None:
    """Values arrive from YAML and from Optuna, which is not always as much of an int as it looks."""
    assert _widths(build_mlp_stack(16, ["64", 32.0], 1, dropout=0.1)) == [64, 32, 1]


# --- build_mlp_stack beyond a head -------------------------------------------


def test_no_output_dim_leaves_the_stack_at_its_last_width() -> None:
    """A branch feeding a fusion has no readout to project to."""
    stack = build_mlp_stack(16, [64, 32], None, dropout=0.1)

    assert _widths(stack) == [64, 32]
    assert isinstance(list(stack)[-1], nn.ReLU)


def test_an_empty_stack_with_no_output_dim_is_an_identity() -> None:
    assert isinstance(build_mlp_stack(16, [], None, dropout=0.1), nn.Identity)


def test_dropout_final_makes_every_block_a_full_block() -> None:
    modules = list(
        build_mlp_stack(16, [64, 32], None, dropout=0.1, norm_final=True, dropout_final=True)
    )

    assert [type(m) for m in modules] == [
        nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout,
        nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout,
    ]


def test_the_static_encoder_layout_is_reproduced() -> None:
    """What TabularStaticEncoder used to hand-build, projection included."""
    modules = list(
        build_mlp_stack(16, [64], 32, dropout=0.1, norm_final=True, dropout_final=True)
    )

    assert [type(m) for m in modules] == [nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout, nn.Linear]


# --- the calendar grid: month lookup, rasteriser, pooling and the temporal encoders -----------
# x


ENCODERS = ["dilated_tempcnn", "annual_grid2d"]
ENCODER_CLASSES = [DilatedTempCNNEncoder, AnnualGrid2DEncoder]


def _times(dates) -> torch.Tensor:
    return torch.as_tensor(to_decimal_year(pd.Series(pd.to_datetime(dates)))).unsqueeze(0)


# --- month lookup ----------------------------------------------------------


@pytest.mark.parametrize("year", [2017, 2019, 2020, 2024, 2000, 2100])
def test_every_day_of_a_year_maps_to_its_calendar_month(year: int) -> None:
    """Includes leap and century years: March onward shifts a day once February has 29."""
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    resolved = decimal_year_to_month_index(_times(days))[0].numpy()
    np.testing.assert_array_equal(resolved, days.month.to_numpy() - 1)


def test_month_lookup_beats_the_naive_fraction_scaling() -> None:
    """1 November is the canonical break: floor(0.8323 * 12) == 9, i.e. October."""
    times = _times(["2021-11-01", "2021-12-01"])
    assert decimal_year_to_month_index(times)[0].tolist() == [10, 11]

    naive = ((times - times.floor()) * 12).floor().long()[0].tolist()
    assert naive == [9, 10], "the naive scaling mis-assigns both, which is why it is not used"


def test_first_of_month_stamps_are_exact_in_a_leap_year() -> None:
    dates = [f"2020-{month:02d}-01" for month in range(1, 13)]
    assert decimal_year_to_month_index(_times(dates))[0].tolist() == list(range(12))


# --- rasteriser ------------------------------------------------------------


def test_rasterizer_places_observations_in_the_right_cells() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=3)
    times = _times(["2021-03-01", "2022-07-01", "2023-11-01", "2023-12-01"])
    values = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    mask = torch.ones(1, 4, dtype=torch.bool)

    grid, cell_mask = rasterizer(values, mask, times)

    assert grid.shape == (1, rasterizer.output_channels, 3, 12)
    # Rows count back from the point's own latest year (2023 -> row 2).
    assert cell_mask[0].nonzero().tolist() == [[0, 2], [1, 6], [2, 10], [2, 11]]
    assert [float(grid[0, 0, row, month]) for row, month in cell_mask[0].nonzero().tolist()] == [1.0, 2.0, 3.0, 4.0]


def test_rasterizer_leaves_missing_months_unoccupied() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=1)
    present = [1, 2, 3, 9, 10, 11, 12]  # months 4..8 absent, the sub-12-month case
    times = _times([f"2022-{month:02d}-01" for month in present])
    values = torch.ones(1, len(present), 1)

    _, cell_mask = rasterizer(values, torch.ones(1, len(present), dtype=torch.bool), times)

    occupied = sorted(month for _, month in cell_mask[0].nonzero().tolist())
    assert occupied == [month - 1 for month in present]
    assert cell_mask[0, 0, 3:8].sum() == 0


def test_rasterizer_drops_observations_older_than_the_window() -> None:
    """Older readings must fall off the grid, not wrap into an occupied cell."""
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=2)
    times = _times(["2015-06-01", "2022-06-01", "2023-06-01"])
    values = torch.tensor([[[9.0], [1.0], [2.0]]])

    grid, cell_mask = rasterizer(values, torch.ones(1, 3, dtype=torch.bool), times)

    assert cell_mask[0].nonzero().tolist() == [[0, 5], [1, 5]]
    assert [float(grid[0, 0, row, month]) for row, month in cell_mask[0].nonzero().tolist()] == [1.0, 2.0]
    assert 9.0 not in grid[0, 0].flatten().tolist()


def test_rasterizer_averages_repeated_readings_in_one_cell() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=1)
    times = _times(["2022-05-02", "2022-05-20"])
    values = torch.tensor([[[2.0], [4.0]]])

    grid, cell_mask = rasterizer(values, torch.ones(1, 2, dtype=torch.bool), times)

    assert cell_mask[0].nonzero().tolist() == [[0, 4]]
    assert float(grid[0, 0, 0, 4]) == pytest.approx(3.0)


def test_rasterizer_handles_a_point_with_no_observations() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=2, grid_years=3)
    times = _times(["2022-01-01", "2022-02-01"])
    values = torch.randn(1, 2, 2)

    grid, cell_mask = rasterizer(values, torch.zeros(1, 2, dtype=torch.bool), times)

    assert not bool(cell_mask.any())
    assert torch.isfinite(grid).all()


def test_rasterizer_is_invariant_to_the_calendar_era() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=3)
    times = _times(["2021-03-01", "2022-07-01", "2023-11-01"])
    values = torch.tensor([[[1.0], [2.0], [3.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    base_grid, base_mask = rasterizer(values, mask, times)
    shifted_grid, shifted_mask = rasterizer(values, mask, _times(["2031-03-01", "2032-07-01", "2033-11-01"]))

    torch.testing.assert_close(base_grid, shifted_grid)
    assert torch.equal(base_mask, shifted_mask)


def test_rasterizer_handles_irregular_cadence() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=1)
    times = _times(["2022-01-03", "2022-01-10", "2022-02-19", "2022-02-22"])  # 7, 40, 3 days apart
    values = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]])

    grid, cell_mask = rasterizer(values, torch.ones(1, 4, dtype=torch.bool), times)

    assert cell_mask[0].nonzero().tolist() == [[0, 0], [0, 1]]
    assert float(grid[0, 0, 0, 0]) == pytest.approx(2.0)  # (1+3)/2
    assert float(grid[0, 0, 0, 1]) == pytest.approx(6.0)  # (5+7)/2


def test_rasterizer_emits_validity_and_positional_channels() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=2, grid_years=1)
    assert rasterizer.output_channels == 2 + 2 + 1 + 2  # data, validity, cell flag, month sin/cos

    times = _times(["2022-04-01"])
    values = torch.tensor([[[1.0, 2.0]]])
    validity = torch.tensor([[[True, False]]])
    grid, _ = rasterizer(values, torch.ones(1, 1, dtype=torch.bool), times, validity)

    assert float(grid[0, 2, 0, 3]) == pytest.approx(1.0)  # channel 0 measured
    assert float(grid[0, 3, 0, 3]) == pytest.approx(0.0)  # channel 1 median-filled
    assert float(grid[0, 4, 0, 3]) == pytest.approx(1.0)  # cell occupied

    bare = CalendarGridRasterizer(num_channels=2, grid_years=1, use_validity_channels=False, month_positional=False)
    assert bare.output_channels == 3


def test_rasterizer_infers_its_span_when_none_is_configured() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=None)
    times = _times(["2019-01-01", "2021-01-01"])
    _, cell_mask = rasterizer(times.new_ones(1, 2, 1), torch.ones(1, 2, dtype=torch.bool), times)
    assert cell_mask.shape == (1, 3, 12)  # 2019, 2020, 2021


# --- pooling ---------------------------------------------------------------


def test_masked_pool_ignores_empty_cells() -> None:
    features = torch.tensor([[[[2.0, 0.0], [0.0, 0.0]]]])  # (B=1, C=1, 2, 2)
    cell_mask = torch.tensor([[[True, False], [False, False]]])

    assert float(masked_global_pool(features, cell_mask)) == pytest.approx(2.0)
    # Plain averaging divides by every cell, shrinking a sparse point toward zero.
    assert float(masked_global_pool(features, cell_mask, masked=False)) == pytest.approx(0.5)


def test_masked_pool_zeroes_a_fully_empty_grid() -> None:
    features = torch.randn(2, 3, 4, 4)
    cell_mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    assert bool((masked_global_pool(features, cell_mask) == 0).all())


# --- encoders --------------------------------------------------------------


def test_dilated_conv_reaches_exactly_twelve_months_back() -> None:
    """The point of dilation=12: a spike must resurface 12 cells either side of where it landed."""
    encoder = DilatedTempCNNEncoder(num_channels=1, output_dim=4, hidden_dims=[2], norm="none").eval()
    near_conv, far_conv = encoder.blocks.stages[0][0], encoder.blocks.stages[1][0]
    assert (near_conv.kernel_size, near_conv.dilation, near_conv.padding) == ((3,), (1,), (1,))
    assert (far_conv.kernel_size, far_conv.dilation, far_conv.padding) == ((3,), (12,), (12,))
    torch.nn.init.constant_(near_conv.weight, 0.5)
    torch.nn.init.constant_(far_conv.weight, 0.5)

    baseline = torch.zeros(1, 1, 36)
    spike = baseline.clone()
    spike[0, 0, 18] = 1.0
    cell_mask = torch.ones(1, 36, dtype=torch.bool)

    with torch.no_grad():
        response = (encoder.blocks(spike, cell_mask) - encoder.blocks(baseline, cell_mask)).abs().sum(dim=1)[0]

    touched = set(response.nonzero().flatten().tolist())
    assert {17, 18, 19} <= touched, "the k=3 layer must reach its immediate neighbours"
    assert {6, 30} <= touched, "the dilation=12 layer must reach one year either side"
    assert 12 not in touched, "positions outside the receptive field must stay untouched"


@pytest.mark.parametrize("encoder_cls", ENCODER_CLASSES)
@pytest.mark.parametrize("grid_years", [1, 3, 9, 15])
def test_encoders_accept_any_grid_span(encoder_cls, grid_years: int) -> None:
    encoder = encoder_cls(num_channels=4, output_dim=8, hidden_dims=[6]).eval()
    grid = torch.randn(2, 4, grid_years, 12)
    cell_mask = torch.ones(2, grid_years, 12, dtype=torch.bool)

    with torch.no_grad():
        embedding = encoder(grid, cell_mask)

    assert embedding.shape == (2, 8)
    assert torch.isfinite(embedding).all()


@pytest.mark.parametrize("encoder_cls", ENCODER_CLASSES)
def test_masked_pooling_makes_extra_empty_years_free(encoder_cls) -> None:
    """Padding the grid with empty years must not move the embedding.

    This is what allows grid_years to be inferred rather than frozen into the checkpoint.
    """
    encoder = encoder_cls(num_channels=3, output_dim=5, hidden_dims=[4], norm="none").eval()
    torch.manual_seed(0)
    payload = torch.randn(1, 3, 2, 12)
    payload_mask = torch.ones(1, 2, 12, dtype=torch.bool)

    padded = torch.cat([torch.zeros(1, 3, 4, 12), payload], dim=2)
    padded_mask = torch.cat([torch.zeros(1, 4, 12, dtype=torch.bool), payload_mask], dim=1)

    with torch.no_grad():
        torch.testing.assert_close(encoder(payload, payload_mask), encoder(padded, padded_mask))


def test_concat_gated_fusion_matches_the_specified_formula() -> None:
    fusion = ConcatGatedFusion(static_dim=3, temporal_dim=2)
    static = torch.randn(4, 3)
    temporal = torch.randn(4, 2)

    joined = torch.cat([static, temporal], dim=-1)
    expected = joined * torch.sigmoid(fusion.gate(joined))

    torch.testing.assert_close(fusion(static, temporal), expected)
    assert fusion.output_dim == 5


# --- per-block conv widths ---------------------------------------------------


@pytest.mark.parametrize("encoder_cls", [DilatedTempCNNEncoder, AnnualGrid2DEncoder])
def test_a_repeated_width_reproduces_the_old_num_blocks_stack(encoder_cls) -> None:
    """`hidden_dims=[64, 64]` must be exactly what `hidden_dim=64, num_blocks=2` built."""
    encoder = encoder_cls(num_channels=7, output_dim=8, hidden_dims=[64, 64])
    convs = [m for m in encoder.blocks.modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d))]

    # Two conv stages per block, and the first of each block reads the previous block's width.
    assert [(c.in_channels, c.out_channels) for c in convs] == [
        (7, 64), (64, 64),      # block 1: raw channels in, then the dilated/inter-annual pass
        (64, 64), (64, 64),     # block 2
    ]
    assert encoder.projection.in_features == 64


@pytest.mark.parametrize("encoder_cls", [DilatedTempCNNEncoder, AnnualGrid2DEncoder])
def test_the_conv_stack_can_widen_across_blocks(encoder_cls) -> None:
    """The shape a single shared width could not express, and the conventional one for convs."""
    encoder = encoder_cls(num_channels=7, output_dim=8, hidden_dims=[32, 64, 128])
    convs = [m for m in encoder.blocks.modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d))]

    assert [(c.in_channels, c.out_channels) for c in convs] == [
        (7, 32), (32, 32),
        (32, 64), (64, 64),
        (64, 128), (128, 128),
    ]
    assert encoder.projection.in_features == 128


@pytest.mark.parametrize("encoder_cls", [DilatedTempCNNEncoder, AnnualGrid2DEncoder])
def test_a_widening_stack_still_runs_end_to_end(encoder_cls) -> None:
    encoder = encoder_cls(num_channels=4, output_dim=8, hidden_dims=[8, 16], norm="none").eval()
    grid = torch.randn(2, 4, 3, 12)
    cell_mask = torch.ones(2, 3, 12, dtype=torch.bool)

    assert encoder(grid, cell_mask).shape == (2, 8)


@pytest.mark.parametrize("encoder_cls", [DilatedTempCNNEncoder, AnnualGrid2DEncoder])
def test_an_empty_conv_width_list_is_refused(encoder_cls) -> None:
    """Zero blocks would pool the raw rasterized channels: a different model, not a smaller one."""
    with pytest.raises(ValueError, match="at least one hidden width"):
        encoder_cls(num_channels=4, output_dim=8, hidden_dims=[])


# --- AttentionFusion on its own ---------------------------------------------------------------
# x


def _fusion(**kwargs) -> AttentionFusion:
    defaults = dict(
        static_dims=[5], temporal_dims=[4, 3], coordinate_dim=2, d_model=8, nhead=2, num_layers=1, dropout=0.0
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return AttentionFusion(**defaults).eval()


def _inputs(batch_size=3, static=5, temporal=7, coords=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(batch_size, static, generator=generator),
        torch.randn(batch_size, temporal, generator=generator),
        torch.randn(batch_size, coords, generator=generator) if coords else None,
    ]


@pytest.mark.parametrize("readout, width", [("cls", 8), ("mean", 8), ("flatten", 4 * 8)])
def test_the_output_width_follows_the_readout(readout, width) -> None:
    fusion = _fusion(readout=readout)
    with torch.no_grad():
        out = fusion(*_inputs())

    assert fusion.num_tokens == 4  # static, two modalities, location
    assert fusion.output_dim == width
    assert out.shape == (3, width)


def test_per_feature_static_dims_give_one_token_per_column() -> None:
    """[1]*5 is FT-Transformer's tokenizer: a width-1 chunk through a Linear is x*w + b."""
    fusion = _fusion(static_dims=[1] * 5)

    assert fusion.num_tokens == 5 + 2 + 1
    assert [tokenizer.in_features for tokenizer in fusion.tokenizers] == [1] * 5 + [4, 3, 2]
    with torch.no_grad():
        assert torch.isfinite(fusion(*_inputs())).all()


def test_no_coordinate_token_without_coordinates() -> None:
    fusion = _fusion(coordinate_dim=0)
    static, temporal, _ = _inputs(coords=0)

    assert [tokenizer.in_features for tokenizer in fusion.tokenizers] == [5, 4, 3]
    with torch.no_grad():
        assert fusion(static, temporal, None).shape == (3, 8)


def test_an_empty_static_dims_reads_no_static_tokens_whatever_arrives() -> None:
    """per_feature mode relies on this when a dataset has no static features at all."""
    fusion = _fusion(static_dims=[])
    static, temporal, coords = _inputs()

    with torch.no_grad():
        assert torch.equal(fusion(static, temporal, coords), fusion(torch.zeros(3, 11), temporal, coords))


@pytest.mark.parametrize(
    "position, columns",
    [(0, slice(0, 5)), (1, slice(0, 4)), (1, slice(4, 7)), (2, slice(0, 2))],
    ids=["static", "first_modality", "second_modality", "location"],
)
def test_every_chunk_reaches_the_cls_readout(position, columns) -> None:
    fusion = _fusion(readout="cls")
    inputs = _inputs()

    with torch.no_grad():
        before = fusion(*inputs)
        inputs[position] = inputs[position].clone()
        inputs[position][:, columns] += 1.0
        after = fusion(*inputs)

    assert not torch.allclose(before, after)


def test_branches_are_rebuilt_from_each_other_before_the_head() -> None:
    """What attention adds over the gate: a modality token's OUTPUT depends on the static input.

    Under flatten every token keeps its own slot, so this reads token 1 - the first modality - alone.
    """
    fusion = _fusion(readout="flatten")
    static, temporal, coords = _inputs()

    with torch.no_grad():
        before = fusion(static, temporal, coords)[:, 8:16]
        after = fusion(static + 1.0, temporal, coords)[:, 8:16]

    assert not torch.allclose(before, after)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(d_model=10, nhead=4), "divisible by nhead"),
        (dict(readout="max"), "readout must be one of"),
        (dict(static_dims=[], temporal_dims=[], coordinate_dim=0), "at least one input token"),
        (dict(static_dims=[0]), "must all be positive"),
        (dict(ff_multiplier=0), "ff_multiplier"),
    ],
)
def test_bad_settings_are_refused_at_construction(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _fusion(**kwargs)


def test_a_chunk_width_mismatch_is_refused_rather_than_resliced() -> None:
    fusion = _fusion()
    static, _, coords = _inputs()

    with pytest.raises(ValueError, match="expected 7 temporal"):
        fusion(static, torch.randn(3, 6), coords)


# --- entity embeddings and the static encoder -------------------------------------------------
# x


def test_embedding_block_concatenates_per_feature_vectors():
    block = EntityEmbeddingBlock([7, 13], feature_names=["texture_20cm", "landform_class"])

    assert block.embedding_dims == [4, 7]
    assert block.output_dim == 11

    output = block(torch.tensor([[0, 0], [3, 12]], dtype=torch.long))
    assert output.shape == (2, 11)
    assert torch.isfinite(output).all()


def test_embedding_block_is_a_lookup_not_a_magnitude():
    """Codes 1 and 2 must be unrelated vectors; that is the whole point over ordinal codes."""
    block = EntityEmbeddingBlock([4], embedding_dims=1)
    with torch.no_grad():
        block.embeddings[0].weight.copy_(torch.tensor([[0.0], [5.0], [-3.0], [1.0]]))

    output = block(torch.tensor([[0], [1], [2], [3]], dtype=torch.long))

    assert output.squeeze(-1).tolist() == [0.0, 5.0, -3.0, 1.0]


def test_embedding_block_gradients_reach_only_the_looked_up_rows():
    block = EntityEmbeddingBlock([5])

    block(torch.tensor([[2]], dtype=torch.long)).sum().backward()

    grad = block.embeddings[0].weight.grad
    assert grad[2].abs().sum() > 0
    assert grad[[0, 1, 3, 4]].abs().sum() == 0


def test_embedding_block_with_no_features_returns_an_empty_tensor():
    block = EntityEmbeddingBlock([])

    assert block.output_dim == 0
    assert block(torch.zeros((3, 0), dtype=torch.long)).shape == (3, 0)


def test_embedding_block_rejects_a_wrong_column_count():
    block = EntityEmbeddingBlock([7, 13], feature_names=["texture_20cm", "landform_class"])

    with pytest.raises(ValueError, match="1 column"):
        block(torch.tensor([[0]], dtype=torch.long))


def test_embedding_dropout_is_inactive_in_eval():
    block = EntityEmbeddingBlock([7], dropout=0.9).eval()
    indices = torch.tensor([[3]], dtype=torch.long)

    assert torch.equal(block(indices), block(indices))


# --- TabularStaticEncoder -------------------------------------------------


def test_static_encoder_concatenates_continuous_and_embedded_blocks():
    encoder = TabularStaticEncoder(num_continuous=11, hidden_dims=[64], cardinalities=[7, 13])

    assert encoder.embedding_dims == [4, 7]
    assert encoder.input_dim == 22  # 11 continuous + 4 + 7
    assert encoder.output_dim == 64

    output = encoder(torch.randn(5, 11), torch.zeros((5, 2), dtype=torch.long))
    assert output.shape == (5, 64)


def test_static_encoder_projects_to_output_dim_when_asked():
    """The sequence model projects to fusion_dim; the CNN model does not."""
    projected = TabularStaticEncoder(num_continuous=11, hidden_dims=[64], output_dim=32, cardinalities=[7])
    unprojected = TabularStaticEncoder(num_continuous=11, hidden_dims=[64], cardinalities=[7])

    assert projected(torch.randn(4, 11), torch.zeros((4, 1), dtype=torch.long)).shape == (4, 32)
    assert unprojected(torch.randn(4, 11), torch.zeros((4, 1), dtype=torch.long)).shape == (4, 64)


def test_static_encoder_works_with_no_categoricals():
    encoder = TabularStaticEncoder(num_continuous=11, hidden_dims=[16])

    assert encoder.input_dim == 11
    assert encoder(torch.randn(3, 11)).shape == (3, 16)


def test_static_encoder_works_with_no_continuous_features():
    encoder = TabularStaticEncoder(num_continuous=0, hidden_dims=[16], cardinalities=[7, 13])

    assert encoder.input_dim == 11
    assert encoder(torch.zeros((3, 0)), torch.zeros((3, 2), dtype=torch.long)).shape == (3, 16)


def test_static_encoder_raises_when_it_would_have_no_input():
    with pytest.raises(ValueError, match="at least one feature"):
        TabularStaticEncoder(num_continuous=0, hidden_dims=[16])


def test_static_encoder_raises_when_categoricals_are_missing_from_the_batch():
    encoder = TabularStaticEncoder(
        num_continuous=11, hidden_dims=[16], cardinalities=[7], feature_names=["texture_20cm"]
    )

    with pytest.raises(KeyError, match="texture_20cm"):
        encoder(torch.randn(2, 11))


@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("continuous_norm", ["none", "batch", "layer"])
def test_static_encoder_variants_build_and_run(activation, continuous_norm):
    encoder = TabularStaticEncoder(
        num_continuous=6,
        hidden_dims=[8],
        cardinalities=[4],
        activation=activation,
        continuous_norm=continuous_norm,
    )

    output = encoder(torch.randn(4, 6), torch.zeros((4, 1), dtype=torch.long))

    assert output.shape == (4, 8)
    assert torch.isfinite(output).all()


def test_static_encoder_rejects_an_unknown_activation():
    with pytest.raises(ValueError, match="activation"):
        TabularStaticEncoder(num_continuous=4, hidden_dims=[8], activation="swish")


def test_static_encoder_rejects_an_unknown_continuous_norm():
    with pytest.raises(ValueError, match="continuous_norm"):
        TabularStaticEncoder(num_continuous=4, hidden_dims=[8], continuous_norm="instance")


def test_static_encoder_backprops_into_the_embedding_tables():
    encoder = TabularStaticEncoder(num_continuous=3, hidden_dims=[8], cardinalities=[5])

    encoder(torch.randn(4, 3), torch.tensor([[1], [2], [1], [3]], dtype=torch.long)).sum().backward()

    assert encoder.embeddings.embeddings[0].weight.grad.abs().sum() > 0


# --- static encoder depth ----------------------------------------------------


def test_a_single_width_reproduces_the_old_hand_built_stack() -> None:
    """`hidden_dims=[64]` must be exactly what `hidden_dim=64` built before it took a list."""
    from torch import nn

    encoder = TabularStaticEncoder(num_continuous=11, hidden_dims=[64], cardinalities=[7])

    assert [type(m) for m in encoder.encoder] == [nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout]
    assert encoder.output_dim == 64


def test_the_static_branch_can_be_deepened() -> None:
    from torch import nn

    encoder = TabularStaticEncoder(num_continuous=11, hidden_dims=[128, 64], cardinalities=[7])
    widths = [layer.out_features for layer in encoder.encoder if isinstance(layer, nn.Linear)]

    assert widths == [128, 64]
    assert encoder.output_dim == 64  # the fusion is sized off the LAST block


def test_a_deep_static_branch_still_projects_to_the_requested_width() -> None:
    encoder = TabularStaticEncoder(num_continuous=11, hidden_dims=[128, 64], output_dim=32, cardinalities=[7])

    assert encoder.output_dim == 32
    assert encoder(torch.randn(4, 11), torch.zeros(4, 1, dtype=torch.long)).shape == (4, 32)


def test_an_empty_static_width_list_is_refused() -> None:
    """A zero-layer static branch feeds the raw concatenation into the fusion - a different model."""
    with pytest.raises(ValueError, match="at least one hidden width"):
        TabularStaticEncoder(num_continuous=11, hidden_dims=[], cardinalities=[7])
