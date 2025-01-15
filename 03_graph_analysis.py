# ruff: noqa: E402
# mypy: python_version = 3.12, disallow_untyped_calls = False
# %%
from cartes.crs import EuroPP  # type: ignore

import pandas as pd
from traffic.algorithms.graphs import AirportGraph
from traffic.data import airports

result = pd.read_pickle("cache/20240109_LSZH_result.pkl")
airport = airports["LSZH"]
ag = AirportGraph.from_airport(airport, EuroPP())


# %%

from compute_analysis import fill_missing_timestamps

filled_result = (
    result.groupby(["flight_id"], as_index=False)
    .apply(lambda df: fill_missing_timestamps(df, ag), include_groups=False)
    .reset_index(drop=True)
)

# %%
from shapely.ops import linemerge

global_start = filled_result.start.min()
global_stop = filled_result.stop.max()
total_duration_seconds = (global_stop - global_start).total_seconds()
stats = (
    # result.eval("duration = stop - start")
    filled_result.eval("duration = stop - start")
    # .groupby("ref")
    .groupby(["first", "last"])
    .agg(
        dict(
            ref="max",
            aeroway="max",
            geometry=linemerge,
            icao24="nunique",
            start="min",
            stop="max",
            duration="sum",
        )
    )
    .assign(distance=lambda df: df.geometry.apply(ag.length))
    .rename(columns=dict(icao24="count_aircraft"))
    .eval("throughput = count_aircraft / @total_duration_seconds * 3600")
    .query('aeroway == "runway" or aeroway == "taxiway"')
    .sort_values("duration")
)
# stats.loc[stats.duration < pd.Timedelta("10 min"), "throughput"] = 0
stats

# %%
import matplotlib.pyplot as plt
from cartes.crs import EuroPP, PlateCarree  # type: ignore
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colors import Normalize, to_hex

norm_count = Normalize(vmin=0, vmax=stats.count_aircraft.max())
norm = Normalize(vmin=0, vmax=stats.duration.dt.total_seconds().max() / 60)
norm_throughput = Normalize(vmin=0, vmax=max(10, stats.throughput.max()))
cmap = plt.get_cmap("YlOrRd")

# %%

ax: GeoAxes

with plt.style.context("traffic"):
    fig, ax = plt.subplots(subplot_kw=dict(projection=EuroPP()))
    airport.plot(ax)

    for _, row in stats.iterrows():
        ax.add_geometries(
            row["geometry"],
            crs=PlateCarree(),
            facecolor="none",
            lw=5 * min(1, norm_count(row["icao24"])),
            edgecolor=cmap(norm(row["throughput"].total_seconds())),
            zorder=5,
        )

    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label="Duration in aircraft × hours",  # noqa: RUF001
        shrink=0.8,
    )

fig
# %%
import json
from typing import Any, TypedDict

import geopandas as gpd
from ipyleaflet import GeoJSON, Map, Popup
from ipywidgets import HTML, Layout

from shapely.geometry import shape
from shapely.ops import linemerge

m = Map(
    center=airport.latlon,
    zoom=14,
    layout=Layout(height="800px"),
)


class FeatureProperties(TypedDict):
    icao24: str
    duration: str
    color: str
    width: int


class FeatureGeometry(TypedDict):
    type: str
    coordinates: Any


class Feature(TypedDict):
    properties: FeatureProperties
    geometry: FeatureGeometry


class Style(TypedDict):
    color: str
    weight: int
    fillColor: str
    fillOpacity: float


def style_callback(feature: Feature) -> Style:
    return {
        "color": feature["properties"]["color"],
        "weight": feature["properties"]["width"],
        "fillColor": feature["properties"]["color"],
        "fillOpacity": 0.5,
    }


def create_popup(feature: Feature) -> HTML:
    properties = feature["properties"]
    html_content = f"""
    <div>
        <strong>Aircraft count:</strong> {properties.get('icao24', 'N/A')}<br>
        <strong>Duration:</strong> {properties.get('duration', 'N/A')}<br>
        <strong>Throughput:</strong> {properties.get('throughput', 'N/A')}<br>
        <strong>Taxiway:</strong> {properties.get('ref', 'N/A')}
    </div>
    """
    return HTML(html_content)


def on_hover(event: Any, feature: Feature, **kwargs: Any) -> None:
    geometry = shape(feature["geometry"])
    c0, c1 = geometry.centroid.coords[0]

    popup = Popup(
        location=(c1, c0),
        child=create_popup(feature),
        close_button=False,
        auto_close=True,
        close_on_escape_key=True,
    )
    m.add_layer(popup)


gdf = gpd.GeoDataFrame(
    stats.reset_index()
    .assign(
        # geometry=lambda df: df.geometry.apply(linemerge),
        width=lambda df: df.count_aircraft.apply(lambda x: 10 * norm_count(x)),
        color=lambda df: df.duration.apply(
            lambda x: to_hex(cmap(norm(x.total_seconds() / 60)))
        ),
        duration=lambda df: df.duration.dt.total_seconds() / 3600,
        throughput=lambda df: 30 * df.throughput.apply(norm_throughput),
    )
    .drop(columns=["start", "stop"]),
    geometry="geometry",
)

geo_data = GeoJSON(
    data=json.loads(gdf.to_json()),
    hover_style={
        "weight": 5,
        "fillOpacity": 0.7,
    },
    style_callback=style_callback,
    name="taxiway stats",
)


geo_data.on_hover(on_hover)
m.add_layer(geo_data)
m

# %%
