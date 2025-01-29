# ruff: noqa: E402
# mypy: python_version = 3.12, disallow_untyped_calls = False
# %%
from cartes.crs import EuroPP  # type: ignore

import pandas as pd
from traffic.algorithms.graphs import AirportGraph
from traffic.data import airports

result = pd.read_pickle("cache/20240109_LSZH_result.pkl")
# result = pd.read_pickle("cache/20240120_LSZH_result.pkl")
# result = pd.read_pickle("cache/20240201_LSZH_result.pkl")
# result = pd.read_pickle("cache/20240401_LSZH_result.pkl")
# result = pd.read_pickle("cache/20240531_LSZH_result.pkl")
# result = pd.read_pickle("cache/20240728_LSZH_result.pkl")
# result = pd.read_pickle("cache/20240812_LSZH_result.pkl")
# result = pd.read_pickle("cache/20240816_LSZH_result.pkl")
# result = pd.read_pickle("cache/20241001_LSZH_result.pkl")
# result = pd.read_pickle("cache/20241116_LSZH_result.pkl")

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
from shapely.ops import unary_union

stats = (
    filled_result.eval("duration = stop - start")
    .groupby(["first", "last"])
    .agg(
        dict(
            ref="max",
            aeroway="max",
            geometry=unary_union,
            icao24="nunique",
            start="min",
            stop="max",
            duration="sum",
        )
    )
    .assign(length=lambda df: df.geometry.apply(ag.length))
    .rename(columns=dict(icao24="count_aircraft"))
    .eval("congestion = duration.dt.total_seconds() / count_aircraft / 60")
    .query('aeroway == "runway" or aeroway == "taxiway"')
    .sort_values("duration")
)
stats

# %%
import matplotlib.pyplot as plt
from cartes.crs import EuroPP, PlateCarree  # type: ignore
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colors import Normalize, to_hex

norm_count = Normalize(vmin=20, vmax=120, clip=True)
norm = Normalize(vmin=0, vmax=2, clip=True)

cmap = plt.get_cmap("YlOrRd")

ax: GeoAxes

with plt.style.context("traffic"):
    fig, ax = plt.subplots(subplot_kw=dict(projection=EuroPP()))
    airport.plot(ax, labels={"font": "Roboto Condensed", "fontsize": 12})

    for _, row in stats.iterrows():
        ax.add_geometries(
            row["geometry"],
            crs=PlateCarree(),
            facecolor="none",
            lw=2 + 2 * norm_count(row["count_aircraft"]),
            edgecolor=cmap(norm(row["congestion"])),
            zorder=5,
        )
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.8,
    )
    cbar.ax.yaxis.set_tick_params(labelsize=12, pad=10)
    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_family("Roboto Condensed")  # type: ignore

    today = filled_result.start.max()
    ax.set_title(
        f"Duration on segment, in minutes per aircraft ({today:%b %d})",
        loc="center",
        y=-0.1,
        fontdict={"family": "Roboto Condensed"},
    )

fig.savefig(f"figures/map_{today:%b%d}.png", dpi=300, transparent=True)
fig
# %%
import json
from typing import Any, TypedDict

import geopandas as gpd
from ipyleaflet import GeoJSON, Map, Popup
from ipywidgets import HTML, Layout

from shapely.geometry import shape

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
        <strong>First:</strong> {properties.get('first', 'N/A')}<br>
        <strong>Last:</strong> {properties.get('last', 'N/A')}<br>
        <strong>Duration:</strong> {properties.get('duration', 'N/A')}<br>
        <strong>Congestion:</strong> {properties.get('congestion', 'N/A')}<br>
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
        width=lambda df: df.count_aircraft.apply(
            lambda x: 5 + 8 * norm_count(x)
        ),
        color=lambda df: df.congestion.apply(lambda x: to_hex(cmap(norm(x)))),
        duration=lambda df: df.duration.dt.total_seconds() / 3600,
        # throughput=lambda df: 30 * df.throughput.apply(norm_throughput),
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
