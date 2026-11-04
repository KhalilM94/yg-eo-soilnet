"""The relief basemap the project's maps sit on, and the chrome that frames them.

Ported from the paper-figures notebook so a diagnostic map in a run and a published one are drawn
the same way. It is deliberately hand-rolled matplotlib: there is no contextily, cartopy or
rasterio in this project's environments, and the pieces needed here are small.

Everything is drawn in Web Mercator (EPSG:3857) through :func:`to_mercator`, which is the closed
form of the projection rather than a pyproj call. That choice is what makes :func:`add_north_arrow`
valid - north is straight up in Web Mercator - and what makes :func:`add_scale_bar` need its
latitude correction, since a Mercator metre is not a ground metre except at the equator.

The relief comes from AWS Terrain Tiles and the borders from Natural Earth. Both are downloaded
once into a cache folder and reused. With neither the cache nor the network, the map still draws:
:func:`draw_basemap` falls back to flat ground and says so, because a figure is never worth failing
a run over.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

import numpy as np
from matplotlib.patches import Polygon, Rectangle

from yg_eo_soilnet.plot_style import INK, INK_2, MUTED

logger = logging.getLogger(__name__)

#: Radius of the sphere Web Mercator is defined on, in metres.
R_EARTH = 6378137.0

TERRAIN_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
NATURAL_EARTH_MAR = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries_mar.zip"
NATURAL_EARTH_FILE = "ne_10m_admin_0_countries_mar.zip"

#: Where the tiles and the borders are kept when a caller names no other folder. The paper figures
#: filled this one, so a run that uses the default zoom starts with every Morocco tile already
#: cached and never reaches the network.
DEFAULT_CACHE_DIR = os.path.join("notebooks", "basemap_cache")

#: The zoom the cached tiles were fetched at. A different zoom is a different set of files.
DEFAULT_ZOOM = 8

#: Warm light grey for land, and the pale blue of the sea. The land is modulated by the hillshade;
#: the range below is what keeps a shaded slope from going black.
LAND_COLOR = np.array([0.93, 0.91, 0.88])
SHADE_FLOOR = 0.6
OCEAN_COLOR = "#dfe8ef"

#: The one line that credits both data sources. Drawn by :func:`finish_map` on every map.
CREDIT = "Relief: Terrain Tiles (SRTM, GMTED2010) · Borders: Natural Earth"


def to_mercator(lon, lat):
    """Project longitude and latitude to Web Mercator metres.

    Parameters
    ----------
    lon, lat : array_like
        Degrees east and north.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)`` in metres.

    Examples
    --------
    >>> x, y = to_mercator(0.0, 0.0)
    >>> float(x), float(y)
    (0.0, 0.0)
    """
    return np.radians(lon) * R_EARTH, np.arcsinh(np.tan(np.radians(lat))) * R_EARTH


def cached_download(url: str, path: str) -> str:
    """Download ``url`` to ``path`` unless it is already there, and return the path.

    Raises
    ------
    Exception
        Whatever the request raised. Callers that can do without the file catch it.
    """
    if os.path.exists(path):
        return path
    import requests

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    response = requests.get(url, timeout=60, headers={"User-Agent": "yg-eo-soilnet"})
    response.raise_for_status()
    # Written under a temporary name first: an interrupted download that landed at the real path
    # would be cached as if it were complete, and every later run would reuse the truncated file.
    partial = f"{path}.partial"
    with open(partial, "wb") as handle:
        handle.write(response.content)
    os.replace(partial, path)
    return path


def elevation_mosaic(west, south, east, north, zoom=DEFAULT_ZOOM, cache_dir=DEFAULT_CACHE_DIR):
    """Stitch the Terrain Tiles covering a lon/lat box.

    Parameters
    ----------
    west, south, east, north : float
        The box, in degrees.
    zoom : int, default 8
        Tile zoom level. Higher is more detailed and many more files.
    cache_dir : str
        Where the tiles are kept between runs.

    Returns
    -------
    elevation : numpy.ndarray
        Metres above sea level, one value per pixel.
    extent : tuple of float
        ``(left, right, bottom, top)`` in Web Mercator metres, for ``imshow``.
    """
    from PIL import Image

    n = 2**zoom
    first_x = int((west + 180) / 360 * n)
    last_x = int((east + 180) / 360 * n)
    first_y = int((1 - np.arcsinh(np.tan(np.radians(north))) / np.pi) / 2 * n)
    last_y = int((1 - np.arcsinh(np.tan(np.radians(south))) / np.pi) / 2 * n)
    xs = range(first_x, last_x + 1)
    ys = range(first_y, last_y + 1)

    rows = []
    for y in ys:
        row = []
        for x in xs:
            path = os.path.join(cache_dir, f"terrarium_{zoom}_{x}_{y}.png")
            tile = cached_download(TERRAIN_TILES.format(z=zoom, x=x, y=y), path)
            row.append(np.asarray(Image.open(tile).convert("RGB")))
        rows.append(np.hstack(row))
    rgb = np.vstack(rows).astype(float)

    # The terrarium encoding: metres packed into the three channels, offset by 32768.
    elevation = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
    size, origin = 2 * np.pi * R_EARTH / n, np.pi * R_EARTH
    extent = (
        -origin + xs[0] * size,
        -origin + (xs[-1] + 1) * size,
        origin - (ys[-1] + 1) * size,
        origin - ys[0] * size,
    )
    return elevation, extent


def hillshade(elevation, extent, azimuth=315, altitude=45, exaggeration=3):
    """Lambertian shading lit from the north-west, scaled so flat ground is 1.

    Parameters
    ----------
    elevation : numpy.ndarray
        Metres, as :func:`elevation_mosaic` returns.
    extent : tuple of float
        The mosaic's extent in Web Mercator metres.
    azimuth, altitude : float
        Where the light comes from, in degrees.
    exaggeration : float
        Vertical stretch. Morocco's relief is subtle at country scale without it.

    Returns
    -------
    numpy.ndarray
        Shading factors, 0 (fully shadowed) upwards, 1 on flat ground.
    """
    x_min, x_max, _, y_max = extent
    cell = (x_max - x_min) / elevation.shape[1]  # Web Mercator metres per pixel
    # A Mercator pixel covers cell / cosh(y / R) metres of ground, so each row gets its own
    # spacing - without this the shading is too strong in the north and too weak in the south.
    row_y = y_max - (np.arange(elevation.shape[0]) + 0.5) * cell
    ground = (cell / np.cosh(row_y / R_EARTH))[:, None]
    d_rows, d_cols = np.gradient(elevation * exaggeration)
    dz_east, dz_north = d_cols / ground, -d_rows / ground  # rows run southwards
    az, alt = np.radians(azimuth), np.radians(altitude)
    light = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
    shade = (light[2] - dz_east * light[0] - dz_north * light[1]) / np.sqrt(dz_east**2 + dz_north**2 + 1)
    return np.clip(shade, 0, None) / light[2]


def _relief_image(bounds, zoom, cache_dir):
    """The shaded land colour and its extent, or ``(None, None)`` if the tiles cannot be had."""
    west, south, east, north = bounds
    try:
        elevation, extent = elevation_mosaic(west, south, east, north, zoom=zoom, cache_dir=cache_dir)
    except Exception as exc:
        logger.info(
            f"No relief on this map: the terrain tiles are neither cached in {cache_dir} nor "
            f"reachable ({type(exc).__name__}: {exc}). Drawing flat ground instead."
        )
        return None, None
    shaded = LAND_COLOR * (SHADE_FLOOR + (1 - SHADE_FLOOR) * hillshade(elevation, extent)[..., None])
    return np.clip(shaded, 0, 1), extent


def _ensure_proj_data() -> None:
    """Point GDAL at PROJ's database when nothing else has.

    A process started without the environment's activation script - a Jupyter kernel, or a run
    launched straight from the env's python - leaves ``PROJ_DATA`` unset, and GDAL then fails to
    read the ``.prj`` inside the Natural Earth archive. pyproj knows where the database is even
    when GDAL does not, so it is asked rather than guessed.
    """
    if os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB"):
        return
    try:
        import pyproj

        os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()
    except Exception:
        # Nothing to do about it here; geopandas will say so if it matters.
        pass


#: The Natural Earth archive, once per cache folder. Reading it is a 4.9 MB parse, and a map with a
#: locator inset wants it twice - so a run drawing one figure per target was reading the same file
#: a dozen times.
_COUNTRIES_CACHE: dict[str, Any] = {}


def _all_countries(cache_dir: str):
    """Every country in Natural Earth's Morocco point-of-view edition, read at most once.

    That edition matters: in it Morocco is a single feature reaching 20.77 degrees north, so
    Western Sahara is drawn as part of the country and there is no disputed-boundary geometry to
    render as a dotted line.
    """
    if cache_dir not in _COUNTRIES_CACHE:
        _ensure_proj_data()
        import geopandas as gpd

        path = cached_download(NATURAL_EARTH_MAR, os.path.join(cache_dir, NATURAL_EARTH_FILE))
        _COUNTRIES_CACHE[cache_dir] = gpd.read_file(path)
    return _COUNTRIES_CACHE[cache_dir]


def _countries(bounds, cache_dir):
    """Natural Earth's countries around ``bounds``, in Web Mercator, or ``None``."""
    west, south, east, north = bounds
    try:
        frame = _all_countries(cache_dir)
        # Clipped in lon/lat first: Antarctica has no finite Web Mercator coordinates.
        return frame.clip((west - 1, south - 1, east + 1, north + 1)).to_crs("EPSG:3857")
    except Exception as exc:
        logger.info(
            f"No borders on this map: Natural Earth is neither cached in {cache_dir} nor reachable "
            f"({type(exc).__name__}: {exc})."
        )
        return None


def draw_basemap(axis, bounds, *, zoom=DEFAULT_ZOOM, cache_dir=DEFAULT_CACHE_DIR, relief=True):
    """Draw relief, sea, veiled neighbours and borders: the ground a map sits on.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Drawn on in Web Mercator metres.
    bounds : tuple of float
        ``(west, south, east, north)`` in degrees - the area to cover, not the final zoom, which
        :func:`finish_map` sets.
    zoom : int
        Tile zoom for the relief.
    cache_dir : str
        Where the tiles and borders are cached.
    relief : bool, default True
        False draws flat ground instead of shaded hills, and fetches no elevation tiles. The
        borders are still drawn when they can be had.

    Returns
    -------
    dict
        What was actually drawn: ``{"relief": bool, "borders": bool}``. Either can be False
        without that being an error - see the module docstring.
    """
    from shapely.geometry import box

    west, south, east, north = bounds
    x_min, y_min = to_mercator(west, south)
    x_max, y_max = to_mercator(east, north)

    drawn = {"relief": False, "borders": False}

    image, extent = _relief_image(bounds, zoom, cache_dir) if relief else (None, None)
    if image is not None:
        axis.imshow(image, extent=extent, interpolation="bilinear", zorder=0)
        drawn["relief"] = True
    else:
        # Flat ground, so the points still sit on land-coloured paper rather than on white.
        axis.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor=tuple(LAND_COLOR),
                edgecolor="none",
                zorder=0,
            )
        )

    countries = _countries(bounds, cache_dir)
    if countries is not None and not countries.empty:
        import geopandas as gpd

        view = box(x_min, y_min, x_max, y_max)
        ocean = gpd.GeoSeries([view], crs="EPSG:3857").difference(countries.union_all())
        # A view with no coast in it leaves nothing to draw, and geopandas warns when asked to plot
        # an empty geometry. Warnings are errors here, so an entirely inland map would fail on it.
        if not ocean.is_empty.all():
            ocean.plot(ax=axis, color=OCEAN_COLOR, lw=0, zorder=1)
        if "ADM0_A3" in countries.columns:
            # The neighbours are veiled rather than hidden: the map is about Morocco, but a blank
            # border would read as sea.
            neighbours = countries[countries["ADM0_A3"] != "MAR"]
            if not neighbours.empty:
                neighbours.plot(ax=axis, color="white", alpha=0.55, lw=0, zorder=1)
        countries.boundary.plot(ax=axis, color=INK_2, lw=0.5, zorder=2)
        drawn["borders"] = True

    return drawn


def add_degree_ticks(axis, lons: Iterable[float], lats: Iterable[float]) -> None:
    """Label the frame in degrees, at the given longitudes and latitudes."""
    lons = np.asarray(list(lons), dtype=float)
    lats = np.asarray(list(lats), dtype=float)
    if lons.size:
        axis.set_xticks(to_mercator(lons, 0)[0], [f"{abs(v):g}°{'W' if v < 0 else 'E'}" for v in lons])
    if lats.size:
        axis.set_yticks(to_mercator(0, lats)[1], [f"{abs(v):g}°{'S' if v < 0 else 'N'}" for v in lats])
    axis.tick_params(
        which="both",
        direction="out",
        length=3,
        color=INK_2,
        labelsize=8,
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
    )


def add_scale_bar(axis, lon: float, lat: float, marks_km=(0, 50, 100, 200), height_km: float = 7) -> None:
    """An alternating black and white bar starting at ``(lon, lat)``.

    It is true to scale at that latitude only, which is why it is anchored by coordinates rather
    than by a corner of the axes: a Web Mercator kilometre grows with latitude.
    """
    x0, y0 = to_mercator(lon, lat)
    stretch = 1000 / np.cos(np.radians(lat))  # Web Mercator metres per ground kilometre here
    height = height_km * stretch
    for index, (start, end) in enumerate(zip(marks_km[:-1], marks_km[1:])):
        axis.add_patch(
            Rectangle(
                (x0 + start * stretch, y0),
                (end - start) * stretch,
                height,
                lw=0.6,
                zorder=6,
                facecolor=INK if index % 2 == 0 else "white",
                edgecolor=INK,
            )
        )
    for km in marks_km:
        axis.text(
            x0 + km * stretch, y0 - height * 0.6, f"{km:g}", ha="center", va="top", fontsize=7, color=INK, zorder=6
        )
    axis.text(x0 + marks_km[-1] * stretch + height, y0 + height / 2, "km", va="center", fontsize=7, color=INK, zorder=6)


def add_north_arrow(axis, x: float = 0.94, y: float = 0.84, size: float = 0.07) -> None:
    """A half-filled arrow in axes coordinates; valid because north is straight up here."""
    tip, base = (x, y + size), y + size * 0.3
    for side, fill in ((-1, INK), (1, "white")):
        axis.add_patch(
            Polygon(
                [tip, (x + side * size * 0.32, y), (x, base)],
                closed=True,
                lw=0.6,
                facecolor=fill,
                edgecolor=INK,
                transform=axis.transAxes,
                zorder=6,
            )
        )
    axis.text(
        x,
        y + size + 0.01,
        "N",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=INK,
        transform=axis.transAxes,
        zorder=6,
    )


def degree_ticks_for(bounds, max_ticks: int = 6) -> tuple[list[float], list[float]]:
    """Whole-degree tick positions inside ``bounds``, thinned to at most ``max_ticks`` each.

    Examples
    --------
    >>> degree_ticks_for((-10.2, 30.1, -5.8, 34.4), max_ticks=6)
    ([-10.0, -9.0, -8.0, -7.0, -6.0], [31.0, 32.0, 33.0, 34.0])
    """
    west, south, east, north = bounds

    def whole(low: float, high: float) -> list[float]:
        """Whole degrees between two bounds, thinned to fit ``max_ticks``."""
        values = list(range(int(np.ceil(low)), int(np.floor(high)) + 1))
        if not values:
            return []
        step = max(1, int(np.ceil(len(values) / max_ticks)))
        return [float(value) for value in values[::step]]

    return whole(west, east), whole(south, north)


def finish_map(
    axis,
    bounds,
    *,
    lons: Optional[Iterable[float]] = None,
    lats: Optional[Iterable[float]] = None,
    scale_bar_at: Optional[tuple] = None,
    north_arrow_at: Optional[tuple] = None,
    credit: Optional[str] = CREDIT,
) -> None:
    """Frame, degree ticks, scale bar, north arrow and credit, then zoom to ``bounds``.

    The house style is built for charts - no left spine, a horizontal grid - and a map needs the
    opposite, so this undoes it rather than working around it.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
    bounds : tuple of float
        ``(west, south, east, north)`` in degrees.
    lons, lats : iterable of float, optional
        Where to put the degree ticks; worked out from ``bounds`` when not given.
    scale_bar_at : tuple, optional
        Arguments for :func:`add_scale_bar` after the axes; omitted when None.
    north_arrow_at : tuple, optional
        Arguments for :func:`add_north_arrow` after the axes; omitted when None.
    credit : str, optional
        The attribution line. Only pass None when nothing from those sources was drawn.
    """
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(INK_2)
        spine.set_linewidth(0.8)

    if lons is None or lats is None:
        default_lons, default_lats = degree_ticks_for(bounds)
        lons = default_lons if lons is None else lons
        lats = default_lats if lats is None else lats
    add_degree_ticks(axis, lons, lats)

    if scale_bar_at is not None:
        add_scale_bar(axis, *scale_bar_at)
    if north_arrow_at is not None:
        add_north_arrow(axis, *north_arrow_at)
    if credit:
        axis.text(
            0.995,
            0.005,
            credit,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.5,
            color=INK_2,
            zorder=6,
        )

    west, south, east, north = bounds
    left, bottom = to_mercator(west, south)
    right, top = to_mercator(east, north)
    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)
    axis.set_aspect("equal")


def axes_fraction(bounds, *, lon=None, lat=None):
    """Where a coordinate sits across the axes, 0 at one edge and 1 at the other.

    Longitude is linear in Web Mercator and latitude is not, so the two are worked out differently.
    It is what lets a piece of chrome placed in axes coordinates - the north arrow - be lined up
    with one placed in real coordinates, such as the scale bar.

    Parameters
    ----------
    bounds : tuple of float
        ``(west, south, east, north)`` in degrees.
    lon, lat : float, optional
        The coordinate to locate; give one or both.

    Returns
    -------
    float or tuple of float
        The fraction for whichever was given, or both as ``(x, y)``.

    Examples
    --------
    >>> axes_fraction((-10.0, 30.0, -2.0, 36.0), lon=-6.0)
    0.5
    """
    west, south, east, north = bounds
    x = None if lon is None else float((lon - west) / (east - west))
    y = None
    if lat is not None:
        bottom = to_mercator(0, south)[1]
        top = to_mercator(0, north)[1]
        y = float((to_mercator(0, lat)[1] - bottom) / (top - bottom))
    if lon is not None and lat is not None:
        return x, y
    return x if lon is not None else y


def north_arrow_placement(bounds, scale_bar_at, *, size: float = 0.06, gap: float = 0.04, height_km: float = 7):
    """Where the north arrow goes: centred over the scale bar, just above it.

    This is the arrangement the paper figures use. Figure 2 anchors its bar at 4.4 degrees west and
    its arrow at 0.755 across the axes, and the bar's own centre works out to 0.757 - the alignment
    is deliberate. Computing it from the bar rather than hard-coding those numbers keeps it true
    for any extent.

    Parameters
    ----------
    bounds : tuple of float
        ``(west, south, east, north)`` in degrees.
    scale_bar_at : tuple
        What :func:`scale_bar_placement` returned: ``(lon, lat, marks)``.
    size : float
        The arrow's height, as a fraction of the axes.
    gap : float
        Clearance between the top of the bar and the foot of the arrow.
    height_km : float
        The bar's own height, as passed to :func:`add_scale_bar`.

    Returns
    -------
    tuple of float
        ``(x, y, size)``, ready to splat into :func:`add_north_arrow`.
    """
    lon, lat, marks = scale_bar_at
    total_km = marks[-1]
    # The bar is drawn from its anchor rightwards, so its centre is half its length along.
    centre_lon = lon + (total_km / 2) / (111.32 * np.cos(np.radians(lat)))
    x = axes_fraction(bounds, lon=centre_lon)

    # The bar's height is set in ground kilometres at its own latitude, like its length.
    bottom = to_mercator(0, bounds[1])[1]
    top = to_mercator(0, bounds[3])[1]
    bar_height = height_km * 1000 / np.cos(np.radians(lat)) / (top - bottom)
    y = axes_fraction(bounds, lat=lat) + bar_height + gap
    return float(x), float(y), size


def scale_bar_placement(bounds, *, width_fraction: float = 0.22, margin: float = 0.08):
    """Where to put the scale bar and what marks to give it, so it fits inside ``bounds``.

    The bar is sized to about ``width_fraction`` of the map, rounded to a readable set of marks,
    then anchored far enough from the right edge that its last label stays inside the frame.

    Parameters
    ----------
    bounds : tuple of float
        ``(west, south, east, north)`` in degrees.
    width_fraction : float
        Roughly how much of the map's width the bar should span.
    margin : float
        Clearance from the right edge, as a fraction of the map. The bar sits higher than this
        above the bottom, because its labels hang below it and the credit line runs along the
        very bottom of the frame.

    Returns
    -------
    tuple
        ``(lon, lat, marks)``, ready to splat into :func:`add_scale_bar`.

    Examples
    --------
    >>> _, _, marks = scale_bar_placement((-10.0, 30.0, -2.0, 36.0))
    >>> marks
    (0, 50, 100, 200)
    """
    west, south, east, north = bounds
    mid_lat = (south + north) / 2
    span_km = (east - west) * 111.32 * np.cos(np.radians(mid_lat))
    target = max(span_km * width_fraction, 1.0)

    # Each ladder step is a quarter, a half and then the whole of its last mark, which is what
    # gives the bar its alternating blocks without any of them being an odd width.
    ladders = [
        (1, 2, 5, 10),
        (0, 5, 10, 20),
        (0, 10, 25, 50),
        (0, 25, 50, 100),
        (0, 50, 100, 200),
        (0, 100, 250, 500),
        (0, 250, 500, 1000),
    ]
    marks = min(ladders, key=lambda ladder: abs(ladder[-1] - target))
    total_km = marks[-1]

    # The bar is drawn from its anchor rightwards, and the final label sits at its end, so the
    # anchor has to leave the bar's own length plus the margin.
    lat = south + (north - south) * 0.10
    bar_degrees = total_km / (111.32 * np.cos(np.radians(lat)))
    # The "km" caption hangs off the right end of the bar, so it gets margin of its own.
    lon = east - (east - west) * margin - bar_degrees
    return lon, lat, marks


#: Which country the locator inset is about, by its Natural Earth code.
LOCATOR_COUNTRY = "MAR"

#: Room left around the country in the locator, as a share of its own span.
LOCATOR_MARGIN = 0.04


def add_locator_inset(
    axis,
    bounds,
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    country: str = LOCATOR_COUNTRY,
    width: float = 0.21,
    loc: str = "upper left",
    pad: float = 0.015,
) -> bool:
    """Draw a small whole-country map in the corner, with the mapped area marked on it.

    A reader who does not know the country cannot place the main map's frame from its degree ticks
    alone. This says where in the country they are looking, and how much of it they are seeing.

    Deliberately a flat silhouette - no relief, no ticks, no scale bar. A locator carries none of
    that, it stays legible at this size, and it needs no elevation tiles.

    The country comes from Natural Earth's Morocco point-of-view edition, in which Morocco is one
    feature reaching 20.77 degrees north. Western Sahara is therefore drawn as part of the country,
    with no dotted internal line to render.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The main map. The inset is placed inside it.
    bounds : tuple of float
        The area the main map covers, ``(west, south, east, north)`` in degrees, drawn as a
        rectangle on the inset.
    cache_dir : str
        Where the Natural Earth archive is cached.
    country : str
        The ``ADM0_A3`` code of the country to draw.
    width : float
        The inset's width, as a fraction of the main map.
    loc : str
        Which corner to sit in, named as :func:`~yg_eo_soilnet.plot_style.metric_box` names them.
    pad : float
        Clearance from the edges of the map, as a fraction of it.

    Returns
    -------
    bool
        Whether anything was drawn. False when the borders cannot be had - offline with an empty
        cache - in which case the main map is simply left without an inset.
    """
    try:
        everything = _all_countries(cache_dir)
    except Exception as exc:
        logger.info(
            f"No locator inset on this map: Natural Earth is neither cached in {cache_dir} nor "
            f"reachable ({type(exc).__name__}: {exc})."
        )
        return False

    if "ADM0_A3" not in everything.columns:
        return False
    subject = everything[everything["ADM0_A3"] == country]
    if subject.empty:
        logger.info(f"No locator inset on this map: Natural Earth has no country {country!r}.")
        return False

    # The frame is the country's own extent, so the inset does not depend on a hard-coded box.
    west, south, east, north = subject.total_bounds
    pad_x = (east - west) * LOCATOR_MARGIN
    pad_y = (north - south) * LOCATOR_MARGIN
    frame = (west - pad_x, south - pad_y, east + pad_x, north + pad_y)

    left, bottom = to_mercator(frame[0], frame[1])
    right, top = to_mercator(frame[2], frame[3])
    height = width * float((top - bottom) / (right - left))

    corners = {
        "upper left": (pad, 1 - pad - height),
        "upper right": (1 - pad - width, 1 - pad - height),
        "lower left": (pad, pad),
        "lower right": (1 - pad - width, pad),
    }
    if loc not in corners:
        raise ValueError(f"loc must be one of {sorted(corners)}; got {loc!r}")
    inset = axis.inset_axes([*corners[loc], width, height], transform=axis.transAxes, zorder=7)

    import geopandas as gpd
    from shapely.geometry import box

    # The subject is excluded from the neighbours, or it would be drawn twice - once veiled in
    # white and once filled - and its own outline would be overdrawn in the neighbours' lighter ink.
    around = everything[everything["ADM0_A3"] != country]
    around = around.clip((frame[0] - 1, frame[1] - 1, frame[2] + 1, frame[3] + 1)).to_crs("EPSG:3857")
    # The locator's frame is the country's own extent, so there is always some sea or neighbour
    # around it; drawn as a flat backdrop rather than cut out, since nothing sits under it.
    view = box(left, bottom, right, top)
    gpd.GeoSeries([view], crs="EPSG:3857").plot(ax=inset, color=OCEAN_COLOR, lw=0, zorder=0)
    if not around.empty:
        around.plot(ax=inset, facecolor="white", edgecolor=MUTED, lw=0.3, zorder=1)
    subject.to_crs("EPSG:3857").plot(ax=inset, facecolor=tuple(LAND_COLOR), edgecolor=INK_2, lw=0.5, zorder=2)

    # The mapped area, over the top.
    area_left, area_bottom = to_mercator(bounds[0], bounds[1])
    area_right, area_top = to_mercator(bounds[2], bounds[3])
    inset.add_patch(
        Rectangle(
            (area_left, area_bottom),
            area_right - area_left,
            area_top - area_bottom,
            facecolor="none",
            edgecolor=INK,
            lw=1.0,
            zorder=4,
        )
    )

    inset.set_xlim(left, right)
    inset.set_ylim(bottom, top)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.grid(False)
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(INK_2)
        spine.set_linewidth(0.6)
    return True
