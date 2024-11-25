# ruff: noqa: E402
# mypy: python_version = 3.12, disallow_untyped_calls = False
# %%
from collections import defaultdict
from itertools import pairwise
from typing import Any, Self

import geopandas as gpd
import networkx as nx
from cartes.crs import EuroPP, Projection
from scipy.spatial import KDTree

import pandas as pd
from pyproj import Proj, Transformer
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import split, transform

projection = EuroPP()


class AirportGraph:
    wgs84 = Proj("EPSG:4326")

    def __init__(self, graph: nx.Graph, projection: Projection) -> None:
        self.graph = graph
        self.projection = projection
        self.project = Transformer.from_proj(
            self.wgs84, self.projection, always_xy=True
        )
        self.to_wgs84 = Transformer.from_proj(
            self.projection, self.wgs84, always_xy=True
        )

        self.node_map = [node for node, _data in self.graph.nodes(data=True)]
        self.node_kdtree = KDTree(
            [
                self.project.transform(*data["pos"])
                for _node, data in self.graph.nodes(data=True)
            ]
        )

    def relabel(self) -> Self:
        pos_to_nodes = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            pos = data.get("pos")
            if pos is not None:
                pos_to_nodes[pos].append(node)
        mapping = {}
        for pos, nodes in pos_to_nodes.items():
            if len(nodes) > 1:
                # Keep the first node and map the others to it
                representative = nodes[0]
                for node in nodes[1:]:
                    mapping[node] = representative

        graph: nx.Graph = nx.relabel_nodes(self.graph, mapping, copy=False)

        # Relabel first and last fields according to the mapping
        for _, _, data in graph.edges(data=True):
            data["first"] = mapping.get(data["first"], data["first"])
            data["last"] = mapping.get(data["last"], data["last"])

        return AirportGraph(graph, self.projection)

    def buffer_meter(
        self, shape: BaseGeometry, buffer: int = 18
    ) -> BaseGeometry:
        projected = transform(self.project.transform, shape)
        buffered = projected.buffer(buffer)
        return transform(self.to_wgs84.transform, buffered)

    def project_shape[T: BaseGeometry](self, shape: T) -> T:
        return transform(self.project.transform, shape)

    def length(self, shape: BaseGeometry) -> float:
        return self.project_shape(shape).length

    def _split_line_with_point(
        self, line: LineString, splitter: BaseGeometry
    ) -> list[LineString]:
        """Split a LineString with a Point"""

        # point is on line, get the distance from the first point on line
        distance_on_line = line.project(splitter)
        coords = list(line.coords)
        # split the line at the point and create two new lines
        current_position = 0.0
        for i in range(len(coords) - 1):
            point1 = coords[i]
            point2 = coords[i + 1]
            dx = point1[0] - point2[0]
            dy = point1[1] - point2[1]
            segment_length = (dx**2 + dy**2) ** 0.5
            current_position += segment_length
            if distance_on_line == current_position:
                # splitter is exactly on a vertex
                return [
                    LineString(coords[: i + 2]),
                    LineString(coords[i + 1 :]),
                ]
            elif distance_on_line < current_position:
                # splitter is between two vertices
                return [
                    LineString(coords[: i + 1] + [splitter.coords[0]]),
                    LineString([splitter.coords[0]] + coords[i + 1 :]),
                ]
        return [line]

    def snap_and_split(
        self,
        graph: nx.Graph,
        u: int,
        v: int,
        tolerance: float,
    ) -> None:
        # print("enter", u, v)
        edge_data = graph.get_edge_data(u, v)
        u, v = edge_data["first"], edge_data["last"]
        line_wgs84 = edge_data["geometry"]
        line_proj = self.project_shape(line_wgs84)

        ((x, y),) = line_proj.centroid.coords
        _dist, idx = self.node_kdtree.query(
            (x, y), distance_upper_bound=line_proj.length / 2, k=30
        )
        candidate_idx = [
            self.node_map[i] for i in idx if i < len(self.node_map)
        ]
        min_list = [
            (
                line_proj.distance(self.project_shape(Point(*data["pos"]))),
                i,
                Point(*data["pos"]),
            )
            for i, data in graph.nodes(data=True)
            if i in candidate_idx and i != u and i != v
        ]
        if len(min_list) == 0:
            return
        min_dist, min_idx, min_point = min(min_list)
        # print(min_dist, min_idx, u, v)
        if min_dist < tolerance:
            projected_point = line_wgs84.interpolate(
                line_wgs84.project(min_point)
            )
            if line_wgs84.distance(projected_point) > 0:
                splits = self._split_line_with_point(
                    line_wgs84, projected_point
                )
            else:
                splits = split(line_wgs84, projected_point).geoms

            splits_iter = iter(splits)
            left = next(splits_iter)
            right = next(splits_iter, None)
            if right is None:
                # TODO prepare a warning here
                return

            # print(f"split ({u=}, {v=}) into ({u=}, {min_idx=}, {v=})")

            graph.remove_edge(u, v)
            graph.add_edge(
                u,
                min_idx,
                **{**edge_data, "geometry": left, "first": u, "last": min_idx},
            )
            graph.add_edge(
                min_idx,
                v,
                **{**edge_data, "geometry": right, "first": min_idx, "last": v},
            )

            # print("left", u, min_idx, list(left.coords))
            self.snap_and_split(graph, u, min_idx, tolerance)

            # print("right", min_idx, v, list(right.coords))
            self.snap_and_split(graph, min_idx, v, tolerance)

    def merge_nodes(self) -> Self:
        creator = self.relabel()
        for u, v in creator.graph.edges:
            self.snap_and_split(creator.graph, u, v, 18)
        return creator

    def map_flight(self, g: Any) -> Any:
        df = pd.DataFrame.from_records(
            data for u, v, data in self.graph.edges(data=True)
        )
        gdf = (
            gpd.GeoDataFrame(df)
            .set_crs(epsg=4326)
            .to_crs(projection)
            .buffer(18)
            .to_crs(epsg=4326)
            .intersects(g.shape)
        )
        inter = df.loc[gdf]

        copy_graph = self.graph.copy()
        for u, v, data in copy_graph.edges(data=True):
            data["distance"] = self.length(data["geometry"])
        for u, v in zip(inter["first"], inter["last"]):
            copy_graph.get_edge_data(u, v)["distance"] = 1

        x0 = g.at_ratio(0)
        x1 = g.at_ratio(1)
        ((x, y),) = self.project_shape(Point(x0.longitude, x0.latitude)).coords
        _, node0 = self.node_kdtree.query((x, y))
        ((x, y),) = self.project_shape(Point(x1.longitude, x1.latitude)).coords
        _, node1 = self.node_kdtree.query((x, y))
        path = nx.shortest_path(
            copy_graph,
            self.node_map[node0],
            self.node_map[node1],
            weight="distance",
        )
        for u, v in pairwise(path):
            yield copy_graph.get_edge_data(u, v)

    @property
    def components(self) -> int:
        return len(list(nx.connected_components(self.graph)))


# %%

from traffic.core import Traffic

t = Traffic.from_file("20241108_AA65162271_LFPG_LFBO.jsonl.7z")
f = t["394c10"].assign(callsign="AFR81GM")
f
# %%
(g := f.first("10 min")).map_leaflet(zoom=15)
# %%
from traffic.data import airports

airport = airports["LFPG"]
graph = airport._openstreetmap().network_graph(
    "geometry",
    "aeroway",
    "ref",
    "name",
    query_str='aeroway == "taxiway" or aeroway == "runway"',
)
ag = AirportGraph(graph, EuroPP()).merge_nodes()

# %%

import matplotlib.pyplot as plt
from cartes import tiles
from cartes.crs import Mercator, PlateCarree
from cartopy.mpl.geoaxes import GeoAxes

tiles_ = tiles.Basemaps(variant="light_all")
ax: GeoAxes
projection = Mercator.GOOGLE
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection=projection))

for u, v, data in ag.graph.edges(data=True):
    ax.add_geometries(
        [data["geometry"]],
        crs=PlateCarree(),
        facecolor="none",
        edgecolor="black",
    )

g.plot(ax=ax, color="#f58518")

pos = dict((node, data["pos"]) for (node, data) in ag.graph.nodes(data=True))
lon_, lat_ = zip(*pos.values())
ax.scatter(lon_, lat_, s=10, transform=PlateCarree())

ax.add_image(tiles_, 16)
ax.set_extent(g, buffer=0.01)
ax.set_square_ratio(crs=projection)
ax.spines["geo"].set_visible(False)

# %%
for edge_data in ag.map_flight(g):
    ax.add_geometries(
        [edge_data["geometry"]],
        crs=PlateCarree(),
        facecolor="none",
        edgecolor="red",
        lw=1.5,
    )
fig

# %%
cumul = []
for edge_data in ag.map_flight(g):
    shape = ag.buffer_meter(edge_data["geometry"], 18)
    seg = g.clip(shape)
    res = edge_data.copy()

    if seg is not None:
        res["start"] = seg.start
        res["stop"] = seg.stop

    cumul.append(res)

parts = pd.DataFrame.from_records(cumul)

# %%
import altair as alt

alt.Chart(parts.drop(columns=["geometry"])).mark_bar().encode(
    alt.X("start"),
    alt.X2("stop"),
    alt.Y("ref", sort="x", title="Name of the taxiway"),
    alt.Color("ref", legend=None),
).properties(width=500)

# %%

airport = airports["LFBO"]
graph = airport._openstreetmap().network_graph(
    "geometry",
    "aeroway",
    "ref",
    "name",
    query_str='aeroway == "taxiway" or aeroway == "runway"',
)
ag = AirportGraph(graph, EuroPP()).merge_nodes()

# %%
g = (
    f.after(f.resample("1s").next("aligned_on_ils('LFBO')").stop)
    .query("latitude.notnull()")
    .resample("1s")
)
# %%
cumul = []
for edge_data in ag.map_flight(g):
    shape = ag.buffer_meter(edge_data["geometry"], 18)
    seg = g.clip(shape)
    res = edge_data.copy()

    if seg is not None:
        res["start"] = seg.start
        res["stop"] = seg.stop

    cumul.append(res)

parts = pd.DataFrame.from_records(cumul)
# %%
import altair as alt

alt.Chart(parts.drop(columns=["geometry"])).mark_bar().encode(
    alt.X("start"),
    alt.X2("stop"),
    alt.Y("ref", sort="x", title="Name of the taxiway"),
    alt.Color("ref", legend=None),
).properties(width=500)

# %%
import matplotlib.pyplot as plt
from cartes import tiles
from cartes.crs import Mercator, PlateCarree
from cartopy.mpl.geoaxes import GeoAxes

tiles_ = tiles.Basemaps(variant="light_all")
ax: GeoAxes
projection = Mercator.GOOGLE
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection=projection))

for u, v, data in ag.graph.edges(data=True):
    ax.add_geometries(
        [data["geometry"]],
        crs=PlateCarree(),
        facecolor="none",
        edgecolor="black",
    )

g.plot(ax=ax, color="#f58518")

pos = dict((node, data["pos"]) for (node, data) in ag.graph.nodes(data=True))
lon_, lat_ = zip(*pos.values())
ax.scatter(lon_, lat_, s=10, transform=PlateCarree())


for edge_data in ag.map_flight(g):
    ax.add_geometries(
        [edge_data["geometry"]],
        crs=PlateCarree(),
        facecolor="none",
        edgecolor="red",
        lw=1.5,
    )
fig

ax.add_image(tiles_, 16)
ax.set_extent(g, buffer=0.01)
ax.set_square_ratio(crs=projection)
ax.spines["geo"].set_visible(False)
# %%
