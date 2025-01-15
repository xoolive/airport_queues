# ruff: noqa: E402
# mypy: python_version = 3.12, disallow_untyped_calls = False
# %%
from traffic.core import Traffic

t = Traffic.from_file("20220601_LSZH.parquet")

# %%
stats = (
    t.iterate_lazy(iterate_kw=dict(by="1h"))
    .summary(
        ["icao24", "callsign", "start", "stop", "duration", "altitude_max"]
    )
    .eval()
)
# %%
(stats.duration.dt.total_seconds() / 3600).plot.hist(bins=30)
# %%
stats.sort_values("duration")
# %%
stats.query("altitude_max.notnull()").altitude_max.plot.hist(bins=30)
# %%
(
    restats := stats.assign(
        duration=lambda df: df.duration.dt.total_seconds() / 60
    ).fillna(-10000)
).plot.scatter(x="duration", y="altitude_max", alpha=0.1)

# %%
restats.query("altitude_max < 0").drop_duplicates(["icao24"])

# %%
restats.query("duration > 100")
