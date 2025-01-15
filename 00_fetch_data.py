from pathlib import Path

import click

import pandas as pd
from traffic.data import airports, opensky


def convert_to_utc_timestamp(
    _ctx: click.Context,
    _param: click.Parameter,
    value: str,
) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize("UTC")


@click.command()
@click.argument("start_date", callback=convert_to_utc_timestamp)
@click.argument("airport_str", default="LSZH")
def main(start_date: pd.Timestamp, airport_str: str) -> None:
    output_file = Path("data") / f"{start_date:%Y%m%d}_{airport_str}.parquet"

    if output_file.exists():
        click.echo(f"File {output_file} already exists")
        return

    stop_date = start_date + pd.Timedelta(days=1)

    t = opensky.history(
        start=start_date,
        stop=stop_date,
        bounds=airports[airport_str].shape.convex_hull.buffer(0.1),
    )
    assert t is not None

    t.to_parquet(output_file)
    click.echo(f"Data saved to {output_file}")


if __name__ == "__main__":
    main()
