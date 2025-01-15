# mypy: python_version = 3.12

import re
from pathlib import Path

import click
from cartes.crs import EuroPP  # type: ignore

import pandas as pd
from traffic.algorithms.graphs import AirportGraph
from traffic.core import Flight, Traffic
from traffic.data import airports


def ground_part(flight: Flight) -> None | Flight:
    if not pd.isna(vmean := flight.vertical_rate_mean):
        if vmean < 0:
            if landing := flight.next('aligned_on_ils("LSZH")'):
                return flight.after(landing.stop)
    if vrate := flight.query("vertical_rate.notnull()"):
        return flight.before(vrate.start)
    return flight


def aircraft_trajectory(flight: Flight) -> bool:
    return isinstance(flight.altitude_max, float) and flight.altitude_max > 0


def debug_flight(f: Flight) -> Flight:
    print(f)
    return f


def fill_missing_timestamps(df: pd.DataFrame, ag: AirportGraph) -> pd.DataFrame:
    df = df.copy()
    i = 0
    while i < len(df):
        if pd.isna(df.at[i, "start"]):
            prev_stop = df.loc[i - 1, "stop"] if i > 0 else pd.NaT
            next_start_idx = df.loc[i:, "start"].dropna().index
            next_start = (
                df.at[next_start_idx[0], "start"]
                if len(next_start_idx) > 0
                else pd.NaT
            )

            if (
                not pd.isna(prev_stop)
                and not pd.isna(next_start)
                and next_start > prev_stop
            ):
                segment_length = ag.length(df.loc[i, "geometry"])
                total_length = (
                    df.loc[i : next_start_idx[0] - 1, "geometry"]
                    .apply(ag.length)
                    .sum()
                )
                time_diff = (next_start - prev_stop).total_seconds()
                time_fraction = segment_length / total_length

                start_time = prev_stop
                stop_time = start_time + pd.Timedelta(
                    seconds=time_diff * time_fraction
                )

                df.at[i, "start"] = start_time
                df.at[i, "stop"] = stop_time
        i += 1

    return df


@click.command()
@click.argument("parquet_file", type=click.Path(exists=True))
@click.option("--max_workers", default=8, help="Maximum number of workers.")
def main(parquet_file: str, max_workers: int) -> None:
    path = Path(parquet_file)
    filename = path.stem

    # Detect airport from the file name
    match = re.search(r"_(\w+)$", filename)
    if not match:
        raise ValueError("Could not detect airport from the file name.")
    airport_code = match.group(1)

    t = Traffic.from_file(path)
    airport = airports[airport_code]
    ag = AirportGraph.from_airport(airport, EuroPP())
    ag.filter_connected_components()

    cache_file = path.parent.parent / "cache" / f"{filename}_result.pkl"

    _result = (
        t.iterate_lazy(iterate_kw=dict(by="1h"))
        .assign_id()
        .pipe(aircraft_trajectory)
        .pipe(ground_part)
        # .pipe(debug_flight)
        .pipe(ag.make_parts)
        .eval(
            desc="",
            max_workers=max_workers,
            cache_file=cache_file,
        )
    )
    click.echo(f"File {cache_file} written")


if __name__ == "__main__":
    main()
