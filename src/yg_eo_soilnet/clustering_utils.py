from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
from sklearn.cluster import KMeans
from yg_eo_soilnet.plot_style import (
    BASELINE,
    FIG_WIDTH_COLUMN,
    GRID,
    INK_2,
    PROJECT_COLORS,
    SAVE_DPI,
    panel_subtitle,
    style_context,
)
from yg_eo_soilnet.utils import assign_grid_ids

import matplotlib.pyplot as plt
import tempfile
import os
import mlflow
import numpy as np

class BaseSpatialClusterStrategy(ABC):
    """
    Abstract base class for spatial clustering strategies.
    """

    @abstractmethod
    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clusters the input DataFrame spatially and adds a 'cluster' column.
        """
        pass

    def plot_train_test(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        lon_col: str = "lon",
        lat_col: str = "lat",
        title: str = "Train/Test Split",
        artifact_path: str = "splits_plots",
        filename: str = "train_test_split.png",
        show: bool = False,
    ):
        """
        Plot clustered points with train/test coloring, save to temp file,
        and log to MLflow as an artifact.

        Unlike every other plotter in this project this one saves itself, because it is called from
        inside the splitter rather than from a logger that would have somewhere to put a Figure.
        """
        with style_context():
            fig, ax = plt.subplots(
                figsize=(FIG_WIDTH_COLUMN, FIG_WIDTH_COLUMN), layout="constrained"
            )

            # Train is context and test is the focus, so train takes the grey and test the colour -
            # the eye should land on the held-out points, which are the ones the picture is about.
            ax.scatter(
                df.loc[train_idx, lon_col],
                df.loc[train_idx, lat_col],
                color=BASELINE,
                s=1.5,
                lw=0,
                label="Train",
                zorder=2,
                rasterized=True,
            )

            ax.scatter(
                df.loc[test_idx, lon_col],
                df.loc[test_idx, lat_col],
                color=PROJECT_COLORS["Al Moutmir"],
                s=6,
                edgecolor="white",
                linewidth=0.25,
                label="Test",
                zorder=3,
                rasterized=True,
            )

            # overlay grid boundaries if available on this strategy
            grid_gdf = getattr(self, "grid_gdf_", None)
            if grid_gdf is not None:
                grid_gdf.boundary.plot(ax=ax, color=GRID, linewidth=0.5, zorder=1)

            # A map, not a chart: the grid goes, all four spines come back, and the aspect is fixed
            # so the coastline is not stretched.
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(INK_2)
                spine.set_linewidth(0.8)
            ax.tick_params(
                which="both", direction="out", length=3, color=INK_2, labelsize=8,
                top=True, right=True, labeltop=False, labelright=False,
            )
            ax.set_aspect("equal")

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            panel_subtitle(ax, title)
            ax.legend(
                loc="upper left", frameon=True, framealpha=0.9, edgecolor=BASELINE,
                fontsize=7.5, handletextpad=0.4,
            )

            # save to temp file and log to MLflow
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, filename)
                fig.savefig(filepath, dpi=SAVE_DPI, bbox_inches="tight")
                mlflow.log_artifact(filepath, artifact_path=artifact_path)

            if show:
                plt.show()

            plt.close(fig)

@dataclass
class KMeansClusterStrategy(BaseSpatialClusterStrategy):
    """
    KMeans-based spatial clustering.

    Parameters:
    -----------
    n_clusters : int, default=12
        Number of spatial clusters to form.
    lat_col : str, default='lat'
        Name of the latitude column in the DataFrame.
    lon_col : str, default='lon'
        Name of the longitude column in the DataFrame.
    random_state : int, default=42
        Random seed for reproducibility of clustering.
    """
    n_clusters: int = 12
    lat_col: str = 'lat'
    lon_col: str = 'lon'
    random_state: int = 42

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        coords = df[[self.lat_col, self.lon_col]].dropna()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(coords) + 1  # 1-indexed

        df = df.copy()
        df.loc[coords.index, 'cluster'] = labels.astype(int)
        return df.dropna(subset=['cluster'])

@dataclass
class SpatialGridClusterStrategy(BaseSpatialClusterStrategy):
    """
    Regular grid-based spatial clustering.

    Parameters
    ----------
    cell_size_m : int
        Size of each grid cell in meters.
    lat_col : str, default='lat'
        Latitude column in the DataFrame.
    lon_col : str, default='lon'
        Longitude column in the DataFrame.
    """

    cell_size_m: int
    lat_col: str = "lat"
    lon_col: str = "lon"
    random_state: int = 42

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        grid_ids, grid_gdf = assign_grid_ids(
            df, cell_size_m=self.cell_size_m, lon_col=self.lon_col, lat_col=self.lat_col
        )
        df["cluster"] = grid_ids.astype(int)
        self.grid_gdf_ = grid_gdf  # save polygons for later plotting
        return df
