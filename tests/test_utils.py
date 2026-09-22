import numpy as np
import pandas as pd
import pytest
from geopandas import GeoDataFrame

from yg_eo_soilnet.utils import LogTransformer, assign_grid_ids


def test_log_transformer_round_trip() -> None:
    transformer = LogTransformer()
    values = np.array([0.0, 1.5, 10.0])

    transformed = transformer.transform(values)
    restored = transformer.inverse_transform(transformed)

    assert np.allclose(restored, values)


def test_assign_grid_ids_returns_grid_and_gdf() -> None:
    frame = pd.DataFrame(
        {
            "lat": [0.0, 0.01, 0.02],
            "lon": [0.0, 0.0, 0.0],
            "value": [1, 2, 3],
        }
    )

    grid_ids, grid_gdf = assign_grid_ids(frame, cell_size_m=1000)

    assert len(grid_ids) == len(frame)
    assert grid_gdf.crs.to_epsg() == 4326
    assert isinstance(grid_gdf, GeoDataFrame)
    assert grid_gdf["Grid_ID"].dtype.kind in {"i", "u"}


@pytest.mark.parametrize("cell_size_m", [None, 0, 50])
def test_assign_grid_ids_rejects_invalid_cell_size(cell_size_m) -> None:
    with pytest.raises(ValueError):
        assign_grid_ids(pd.DataFrame({"lat": [0.0], "lon": [0.0]}), cell_size_m=cell_size_m)
