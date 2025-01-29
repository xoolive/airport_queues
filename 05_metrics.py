# (1) Saturation rate: Are we able to count the number of aircraft taxiing for take-off at time t as well as the departure rate in the interval t to t+i (e.g. 1h after t)?
# (2) Taxi-out times: first moving until lining up
# (3) fundamental diagram --> I suggest only doing it for one single taxiway (e.g. E from the de-icing pad to runway 16). We need to know the throughput at time t and the number of aircraft on the taxiway at time t

# %%
from pathlib import Path

import pandas as pd

cache_folder = Path("./cache")
# result = pd.concat(
#     [pd.read_pickle(file) for file in cache_folder.glob("*.pkl")]
# )

result = pd.read_pickle("./cache/20240109_LSZH_result.pkl")
# result = pd.read_pickle("./cache/20240120_LSZH_result.pkl")
# result = pd.read_pickle("./cache/20240812_LSZH_result.pkl")
# result = pd.read_pickle("./cache/20240728_LSZH_result.pkl")


# %%


def get_movements(df: pd.DataFrame) -> pd.DataFrame:
    movements = (
        # remove parking position to avoid computing the exact start time
        df.query('aeroway != "parking_position"')
        .groupby(["flight_id"])
        .agg(
            {
                "start": "min",
                "stop": "max",
                "aeroway": ["first", "last"],
                "ref": ["first", "last"],
            }
        )
    )
    movements.columns = [
        "_".join(col).strip() for col in movements.columns.values
    ]
    movements = movements.reset_index().sort_values("start_min")
    return movements


movements = get_movements(result)
movements

# %%
movements.groupby(
    ["aeroway_first", "aeroway_last"]
).flight_id.count().reset_index()

# %%
movements.query('aeroway_last == "runway"').groupby(
    ["ref_last"]
).flight_id.count().reset_index()


# %%
def get_taxi_times(
    movements: pd.DataFrame, runway_id: None | str = None
) -> pd.DataFrame:
    if runway_id:
        taxi_times = movements.query(
            f'aeroway_first != "runway" and ref_last == "{runway_id}"'
        ).eval("taxi_time = (stop_max - start_min).dt.total_seconds()/60")
    else:
        taxi_times = movements.query('aeroway_first != "runway"').eval(
            "taxi_time = (stop_max - start_min).dt.total_seconds()/60"
        )
    return taxi_times


taxi_times = get_taxi_times(movements, "10/28")
# taxi_times = get_taxi_times(movements, "14/32")
# taxi_times = get_taxi_times(movements, "16/34")


# %%
import altair as alt

chart = (
    alt.Chart(
        get_taxi_times(movements).query(
            'ref_last == "14/32" | ref_last == "16/34" | ref_last == "10/28"'
        )
    )
    .mark_bar()
    .encode(
        alt.X(
            "taxi_time",
            bin=alt.Bin(maxbins=30, extent=[0, 30]),
            title="Taxi time (in minutes)",
        ).scale(domain=[0, 30], clamp=True),
        alt.Y("count()", title=None),
        alt.Row("ref_last").title(None),
        alt.Color("ref_last").legend(None),
    )
    .properties(width=600, height=200)
    .configure_axis(
        labelFont="Roboto Condensed",
        labelFontSize=16,
        titleFont="Roboto Condensed",
        titleFontSize=20,
        titleAnchor="end",
    )
    .configure_header(
        labelFont="Roboto Condensed",
        labelFontSize=24,
        labelAnchor="start",
        labelOrient="top",
    )
)

chart.show()


# %%
def taxi_out_per_freq(
    movements: pd.DataFrame,
    freq: str = "10 min",
) -> pd.DataFrame:
    day = movements.start_min.dt.date.unique()[0]
    start_time = pd.Timestamp(f"{day} 05:00Z")
    end_time = pd.Timestamp(f"{day} 23:00Z")

    departures = movements.query(
        'aeroway_last == "runway" and aeroway_first != "runway"'
    ).eval("duration = stop_max - start_min")

    counts = pd.DataFrame.from_records(
        {
            "time": time,
            "count": departures.query(
                "start_min < @time and @time < stop_max"
            ).shape[0],
        }
        for time in pd.date_range(start_time, end_time, freq=freq)
    )

    return counts


chart = (
    alt.Chart(taxi_out_per_freq(movements))
    .mark_bar()
    .encode(alt.X("time"), alt.Y("count"))
    .properties(width=600, title="Aircraft taxi-out movements")
)
chart


# %%
def compute_dep_queue(df: pd.DataFrame) -> pd.DataFrame:
    movements = get_movements(df)
    departures = movements.query(
        'aeroway_last == "runway" and aeroway_first != "runway"'
    ).eval("duration = stop_max - start_min")
    cumul = []
    for current_id in departures.flight_id:
        filtered = df.query('flight_id == @current_id and aeroway == "taxiway"')
        entry = filtered.iloc[0]
        aobt = entry.start  # noqa: F841
        entry = departures.query("flight_id == @current_id").iloc[0]
        atot = entry.stop_max  # noqa: F841
        queue = departures.query("stop_max < @atot and stop_max > @aobt")
        cumul.append({"flight_id": current_id, "queue": queue.shape[0]})

    dep_queue = departures.merge(
        pd.DataFrame.from_records(cumul), on="flight_id"
    )
    return dep_queue


dep_queue = compute_dep_queue(result)
# %%

dep_queue = pd.concat(
    [
        compute_dep_queue(pd.read_pickle(file))
        for file in cache_folder.glob("*.pkl")
    ]
)

# %%
takeoff_10 = list(result.query('ref == "L9" | ref == "B9"').flight_id.unique())
dep_queue = dep_queue.query("flight_id not in @takeoff_10").assign(
    ref_last=lambda df: df.ref_last.str.replace("16/34", "runway 16")
    .str.replace("14/32", "runway 32")
    .str.replace("10/28", "runway 28")
)


# %%
alt.Chart(
    dep_queue.eval("duration = duration.dt.total_seconds()/60")
).mark_bar().encode(
    x="start_min",
    y="queue",
)

# %%
chart = (
    alt.Chart(dep_queue.eval("duration = duration.dt.total_seconds()/60"))
    .mark_point()
    .encode(
        alt.X("queue")
        .axis(
            labelFont="Roboto Condensed",
            labelFontSize=16,
            titleFont="Roboto Condensed",
            titleFontSize=16,
            titleAnchor="end",
        )
        .scale(domain=[0, 10])
        .title("Number of preceding queuing aircraft"),
        alt.Y("duration")
        .axis(labelFont="Roboto Condensed", labelFontSize=16)
        .title(None),
        alt.Row("ref_last").title(None),
        alt.Color("ref_last").legend(None),
    )
    .transform_filter("datum.duration < 45 & datum.queue < 10")
    .transform_filter("datum.ref_last != 'Turn Pad'")
    .properties(
        width=400, height=200, title="Taxi-out time duration (in minutes)"
    )
    .configure_title(font="Roboto Condensed", fontSize=16, anchor="start")
)
chart
# %%
import statsmodels.api as sm

# Example grouped data
grouped = (
    dep_queue.eval("duration = duration.dt.total_seconds()/60")
    .query('ref_last != "Turn Pad"')
    .query("duration < 45 & queue < 10")
    .groupby("ref_last")
)

# Initialize a results dictionary to store the output for each runway
results = {}

# Loop through each group
for runway, data in grouped:
    # Define X (independent variable) and Y (dependent variable)
    X = data["queue"]
    y = data["duration"]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    results[runway] = {
        "R2": model.rsquared,
        "beta_0": model.params["const"],
        "beta_0_t_stat": model.tvalues["const"],
        "beta_0_p_value": model.pvalues["const"],
        "beta_1": model.params["queue"],
        "beta_1_t_stat": model.tvalues["queue"],
        "beta_1_p_value": model.pvalues["queue"],
    }

results

# %%


chart = (
    alt.Chart(
        dep_queue.eval("duration = duration.dt.total_seconds()/60"), width=35
    )
    .transform_density(
        "duration",
        as_=["duration", "density"],
        extent=[0, 40],
        groupby=["queue", "ref_last"],
    )
    .transform_filter("datum.duration < 45 & datum.queue < 10")
    .transform_filter("datum.ref_last != 'Turn Pad' & datum.ref_last != 'E8'")
    .mark_area(orient="horizontal")
    .encode(
        alt.X("density:Q")
        .stack("center")
        .impute(None)
        .title(None)
        .axis(labels=False, values=[0], grid=False, ticks=True),
        alt.Y("duration:Q")
        .title("Taxi-out duration (in minutes)")
        .axis(
            labelFont="Roboto Condensed",
            labelFontSize=16,
            titleFont="Roboto Condensed",
            titleFontSize=18,
            titleAnchor="end",
            titleAlign="left",
            titleAngle=0,
            titleY=-15,
            # titleOrient="top",
        ),
        alt.Color("ref_last:N").legend(None),
        alt.Row("ref_last:N")
        .title(None)
        .header(
            labelFont="Roboto Condensed",
            labelFontSize=20,
            labelAngle=0,
            labelAnchor="start",
            labelOrient="right",
            labelPadding=-40,
            labelBaseline="bottom",
            labelAlign="right",
        )
        .spacing(50),
        alt.Column("queue:N")
        .spacing(0)
        .header(
            title="Number of preceding queueing aircraft",
            titleFont="Roboto Condensed",
            titleFontSize=18,
            titleOrient="bottom",
            titleAnchor="end",
            titleBaseline="bottom",
            labelOrient="bottom",
            labelPadding=0,
            labelFont="Roboto Condensed",
            labelFontSize=16,
        ),
    )
    .transform_filter("datum.density > 2.5e-2")
    .configure_view(stroke=None)
    .properties(height=200)
)
chart


# %%
def compute_takeoff_rate(
    df: pd.DataFrame,
    freq: str = "10 min",
    delta: pd.Timedelta = pd.Timedelta("30 min"),
) -> pd.DataFrame:
    day = df.query("start.notnull()").start.dt.date.max()
    start_time = pd.Timestamp(f"{day} 05:00Z")
    end_time = pd.Timestamp(f"{day} 23:00Z")

    movements = get_movements(df)
    departures = movements.query(
        'aeroway_last == "runway" and aeroway_first != "runway"'
    ).eval("duration = stop_max - start_min")

    return pd.DataFrame.from_records(
        {
            "time": time,
            "count": departures.query(
                "start_min < @time and @time < stop_max"
            ).shape[0],
            "takeoff_rate": departures.assign(
                stop_minus=lambda df: df.stop_max - delta,
                stop_plus=lambda df: df.stop_max + delta,
            )
            .query("stop_minus < @time < stop_plus")
            .shape[0],
        }
        for time in pd.date_range(start_time, end_time, freq=freq)
    )


takeoff_rate = compute_takeoff_rate(result)

# %%

takeoff_rate = pd.concat(
    [
        compute_takeoff_rate(pd.read_pickle(file))
        for file in cache_folder.glob("*.pkl")
    ]
)
# %%
chart = (
    alt.Chart(takeoff_rate, width=400, height=400)
    .mark_circle()
    .encode(
        alt.X("count")
        .title("Number of departure aircraft on ground")
        .axis(
            labelFont="Roboto Condensed",
            labelFontSize=16,
            titleFont="Roboto Condensed",
            titleFontSize=16,
            titleAnchor="end",
        )
        .scale(domain=(0, 12)),
        alt.Y("takeoff_rate")
        .title("Take-off rate")
        .axis(
            labelFont="Roboto Condensed",
            titleFont="Roboto Condensed",
            labelFontSize=16,
            titleFontSize=16,
            titleAngle=0,
            titleAnchor="end",
            titleAlign="left",
            titleY=-15,
        ),
    )
    .transform_filter("datum.count > 0")
    .transform_joinaggregate(
        y_median="median(takeoff_rate)",
        y_mean="mean(takeoff_rate)",
        lower_box="q1(takeoff_rate)",
        upper_box="q3(takeoff_rate)",
        lower_whisk="min(takeoff_rate)",
        upper_whisk="max(takeoff_rate)",
        groupby=["count"],
    )
)
(
    chart.mark_bar(size=15, opacity=0.1, color="#9ecae9").encode(
        alt.Y("lower_box:Q"),
        alt.Y2("upper_box:Q"),
    )
    + chart.mark_rule(color="#9ecae9").encode(
        alt.Y("lower_whisk:Q"), alt.Y2("upper_whisk:Q")
    )
    + chart
    + chart.mark_tick(
        color="#f58518", size=15, thickness=3, orient="horizontal"
    )
    .encode(alt.Y("y_median:Q"))
    .transform_filter("datum.count < 9")
    + chart.mark_line(color="#f58518")
    .encode(alt.Y("y_median:Q"))
    .transform_filter("datum.count < 9")
)

# %%

alt.Chart(takeoff_rate).mark_bar(opacity=1).encode(
    x="utchoursminutes(time)", y="count"
).properties(width=600, title="Aircraft taxi-out movements")
# %%

alt.Chart(takeoff_rate).mark_line().encode(
    x="utchoursminutes(time)", y="takeoff_rate", color="utcmonthdate(time):N"
).properties(width=600, title="take off rate")
# %%

# item 3 + try with more data
x = "aircraft per unit time (flow)"
y = "aircraft per length (density)"
result


def compute_fundamental_diagram(df: pd.DataFrame) -> pd.DataFrame:
    date = df.query("start.notnull()").start.dt.date.max()
    counts = pd.DataFrame.from_records(
        {
            "time": time,
            "count": df.assign(
                start_minus=lambda df: df.start - pd.Timedelta("5min"),
                stop_plus=lambda df: df.stop + pd.Timedelta("5min"),
            )
            .query("start_minus < @time and @time < stop_plus")
            .shape[0],
            "duration": df.assign(
                start_minus=lambda df: df.start - pd.Timedelta("5min"),
                stop_plus=lambda df: df.stop + pd.Timedelta("5min"),
            )
            .query("start_minus < @time and @time < stop_plus")
            .eval("duration = stop - start")
            .duration.mean()
            .total_seconds()
            / 60,
        }
        for time in pd.date_range(
            f"{date} 05:00Z", f"{date} 23:00Z", freq="5 min"
        )
    )
    return counts


fundamental = compute_fundamental_diagram(
    result.query("first == 7611985639 and last == 1806108977")
)

# %%
# from cartes.crs import EuroPP  # type: ignore
#
# from traffic.algorithms.graphs import AirportGraph
# from traffic.data import airports
#
# ag = AirportGraph.from_airport(airports["LSZH"], EuroPP())
# length = ag.length(ag.graph.edges[7611985639, 1806108977, 0]["geometry"])
# length => 362.87

# %%
fundamental = pd.concat(
    [
        compute_fundamental_diagram(
            pd.read_pickle(file).query(
                "first == 7611985639 and last == 1806108977"
            )
        )
        for file in cache_folder.glob("*.pkl")
    ]
)
# %%


chart = (
    alt.Chart(fundamental, width=400)
    .mark_circle(opacity=0.4)
    .encode(
        alt.Y("flow:Q")
        .title("Number of aircraft per minute (flow)")
        .axis(
            labelFont="Roboto Condensed",
            labelFontSize=16,
            titleFont="Roboto Condensed",
            titleFontSize=16,
            titleAnchor="end",
            titleAngle=0,
            titleAlign="left",
            titleY=-15,
        ),
        alt.X("density:Q")
        .title("Number of aircraft per meter (density)")
        .axis(
            labelAngle=0,
            labelFont="Roboto Condensed",
            labelFontSize=16,
            titleFont="Roboto Condensed",
            titleFontSize=16,
            titleAnchor="end",
        ),
    )
    .transform_calculate("flow", "datum.count/datum.duration")
    .transform_calculate("density", "datum.count/362.87")
)
(
    chart.transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())",
        density="datum.density + 1.5e-4 * datum.jitter",
    )
    + alt.Chart(pd.DataFrame({"x": [0, 0.011, 0.02], "y": [1.8, 6, 2.1]}))
    .mark_area(fillOpacity=0.3)
    .encode(alt.X("x"), alt.Y("y"))
    + alt.Chart(pd.DataFrame({"x": [0, 0.011, 0.02], "y": [1.8, 6, 2.1]}))
    .mark_line()
    .encode(alt.X("x"), alt.Y("y"))
)
# %%
