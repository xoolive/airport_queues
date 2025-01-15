# ruff: noqa: E402
# %%
from cartes.crs import EuroPP  # type: ignore

import numpy as np
from traffic.algorithms.graphs import AirportGraph
from traffic.core import Traffic
from traffic.data import airports

airport = airports["LSZH"]
ag = AirportGraph.from_airport(airport, EuroPP())
ag.filter_connected_components()

t = Traffic.from_file("data/20240109_LSZH.parquet")

# %%
f = t["SWR468E"]

# %%
from compute_analysis import fill_missing_timestamps, ground_part

ground = ground_part(f)
assert ground is not None
parts = ag.make_parts(ground)
filled = fill_missing_timestamps(parts, ag)

# %%
import altair as alt

base = (
    alt.Chart(
        filled.drop(columns="geometry")
        .assign(ref=lambda df: df.ref.ffill())
        .groupby(["ref"])
        .agg(dict(start="min", stop="max"))
        .reset_index()
    )
    .encode(
        alt.X("start", title=None),
        alt.X2("stop"),
        alt.Y("ref", sort="x", title=None),
        alt.Color("ref", sort="x", legend=None),
    )
    .mark_bar()
    .properties(
        width=500,
        title="Taxiway identifier (resp. runway, parking position)",
    )
    .configure_title(anchor="start", fontSize=16, font="Fira Sans")
    .configure_axis(labelFontSize=16, labelFont="Roboto Condensed")
)
base

# %%
import matplotlib.pyplot as plt
from cartes.crs import PlateCarree  # type: ignore
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.lines import Line2D
from pitot.geodesy import destination

g = ground_part(f)
assert g is not None
inter = ag.intersection_with_flight(g)

ax: GeoAxes

with plt.style.context("traffic"):
    fig, ax = plt.subplots(subplot_kw=dict(projection=EuroPP()))
    airport.plot(ax)
    shift = 70
    text_kw = dict(
        transform=PlateCarree(),
        fontsize=18,
        fontfamily="Fira Sans",
        horizontalalignment="center",
        verticalalignment="center",
        rotation_mode="anchor",
    )
    for thr in airport.runways.list:  # type: ignore
        if thr.name != "28":
            continue
        # Placement of labels
        lat, lon, _ = destination(
            thr.latitude, thr.longitude, thr.bearing + 180, shift
        )

        # Compute the rotation of labels
        lat2, lon2, _ = destination(lat, lon, thr.bearing + 180, 1000)
        x1, y1 = ax.projection.transform_point(
            thr.longitude, thr.latitude, PlateCarree()
        )
        x2, y2 = ax.projection.transform_point(lon2, lat2, PlateCarree())
        rotation = 90 + np.degrees(np.arctan2(y2 - y1, x2 - x1))

        ax.text(lon, lat, thr.name, rotation=rotation, **text_kw)

    # f.plot(ax, color="#4c78a8")
    positions_only = g.handle_last_position().query("latitude.notnull()")
    assert positions_only is not None
    for segment in positions_only.split("10s"):
        segment.plot(ax, lw=9, color="#4c78a8")

    ax.add_geometries(
        inter.geometry,
        facecolor="none",
        edgecolor="#4c78a8",
        lw=2,
        crs=PlateCarree(),
    )
    ax.add_geometries(
        parts.geometry,
        facecolor="none",
        edgecolor="#f58518",
        lw=3,
        crs=PlateCarree(),
        zorder=5,
    )

    legend_elements = [
        Line2D(
            [0], [0], color="#4c78a8", lw=9, label="surface trajectory data"
        ),
        Line2D(
            [0],
            [0],
            color="#4c78a8",
            lw=2,
            label="intersected taxiway elements",
        ),
        Line2D([0], [0], color="#f58518", lw=3, label="reconstructed path"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=12,
        prop={"family": "Fira Sans", "size": 14},
    )

    ax.set_extent(g, buffer=5e-4)

fig.savefig(
    "figures/taxiway_graph_01.png",
    dpi=300,
    transparent=True,
)
fig

# %%
f = t["BAW71FJ"]
ground = ground_part(f)
assert ground is not None
parts = ag.make_parts(ground)

filled = fill_missing_timestamps(parts, ag)

# %%

import altair as alt

base = (
    alt.Chart(
        filled.query("first != 4990745318")
        .drop(columns="geometry")
        .assign(ref=lambda df: df.ref.ffill())
    )
    .encode(
        alt.X("start", title=None),
        alt.X2("stop"),
        alt.Y("ref", sort="x", title=None),
        alt.Color("ref", sort="x", legend=None),
    )
    .mark_bar()
    .properties(
        width=500,
        title="Taxiway identifier (resp. runway, parking position)",
    )
    .configure_title(anchor="start", fontSize=16, font="Fira Sans")
    .configure_axis(labelFontSize=16, labelFont="Roboto Condensed")
)
base

# %%

g = ground_part(f)
assert g is not None
inter = ag.intersection_with_flight(g)

with plt.style.context("traffic"):
    fig, ax = plt.subplots(subplot_kw=dict(projection=EuroPP()))
    airport.plot(ax)
    shift = 100
    text_kw = dict(
        transform=PlateCarree(),
        fontsize=18,
        fontfamily="Fira Sans",
        horizontalalignment="center",
        verticalalignment="center",
        rotation_mode="anchor",
    )
    for thr in airport.runways.list:  # type: ignore
        if thr.name != "10":
            continue
        # Placement of labels
        lat, lon, _ = destination(
            thr.latitude, thr.longitude, thr.bearing + 180, shift
        )

        # Compute the rotation of labels
        lat2, lon2, _ = destination(lat, lon, thr.bearing + 180, 1000)
        x1, y1 = ax.projection.transform_point(
            thr.longitude, thr.latitude, PlateCarree()
        )
        x2, y2 = ax.projection.transform_point(lon2, lat2, PlateCarree())
        rotation = 90 + np.degrees(np.arctan2(y2 - y1, x2 - x1))

        ax.text(lon, lat, thr.name, rotation=rotation, **text_kw)

    # f.plot(ax, color="#4c78a8")
    positions_only = g.handle_last_position().query("latitude.notnull()")
    assert positions_only is not None
    for segment in positions_only.split("10s"):
        segment.plot(ax, lw=9, color="#4c78a8")

    ax.add_geometries(
        inter.geometry,
        facecolor="none",
        edgecolor="#4c78a8",
        lw=3,
        crs=PlateCarree(),
    )
    ax.add_geometries(
        parts.geometry,
        facecolor="none",
        edgecolor="#f58518",
        lw=3,
        crs=PlateCarree(),
        zorder=5,
    )

    legend_elements = [
        Line2D(
            [0], [0], color="#4c78a8", lw=9, label="surface trajectory data"
        ),
        Line2D(
            [0],
            [0],
            color="#4c78a8",
            lw=2,
            label="intersected taxiway elements",
        ),
        Line2D([0], [0], color="#f58518", lw=3, label="reconstructed path"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=12,
        prop={"family": "Fira Sans", "size": 14},
    )

    ax.set_extent(g, buffer=1e-3)
fig.savefig(
    "figures/taxiway_graph_02.png",
    dpi=300,
    transparent=True,
)
fig
