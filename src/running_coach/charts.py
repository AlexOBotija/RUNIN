"""Plotly charts for the app, built ONLY from the analysis functions' results.

The charts never use LLM output: the numbers come straight from weekly_volume(),
pace_trend() and compare_to_peers(), so what the runner sees is exactly what the tools
calculated.

Colours: pastel for backgrounds (the peer band), deeper shades of the same colours for
the data, so bars and lines are easy to see on the light page (contrast at least 3:1).
"""

import math

import plotly.graph_objects as go

from running_coach.analysis.metrics import format_pace

DISTANCE_COLOR = "#2F9670"  # deep sage green: weekly km bars
PACE_COLOR = "#C8693F"  # terracotta (deep peach): pace line
PEER_BAND_COLOR = "#D9D3F0"  # pastel lavender: the p25-p75 range of runners at your level
TEXT_COLOR = "#2B2D42"  # same as textColor in .streamlit/config.toml
MUTED_TEXT_COLOR = "#6B6F80"  # axis labels
GRID_COLOR = "#E6E4DF"  # light gray grid lines

CHART_HEIGHT = 320


def _apply_base_layout(fig: go.Figure, title: str, subtitle: str = "") -> go.Figure:
    """The look shared by every chart: transparent background, quiet axes, small margins."""
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 16, "color": TEXT_COLOR},
            "subtitle": {"text": subtitle, "font": {"size": 12, "color": MUTED_TEXT_COLOR}},
            "x": 0,
        },
        height=CHART_HEIGHT,
        margin={"l": 10, "r": 10, "t": 70 if subtitle else 50, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",  # transparent: the page colour shows through
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": MUTED_TEXT_COLOR, "size": 13},
        showlegend=False,  # one data series per chart: the title and labels name it
        hoverlabel={"bgcolor": "white", "font": {"color": TEXT_COLOR}},
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID_COLOR, tickformat="%d %b")
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig


def weekly_distance_chart(volume: dict, peers: dict | None = None) -> go.Figure:
    """Bars of km per week, with the peer range (p25-p75 weekly km) as a band behind them.

    volume: the result of weekly_volume() with status "ok".
    peers: the result of compare_to_peers(). If it is missing or not "ok", no band is drawn.
    """
    week_starts = [week["week_start"] for week in volume["weekly_km"]]
    km = [week["km"] for week in volume["weekly_km"]]

    fig = go.Figure(
        go.Bar(
            x=week_starts,
            y=km,
            marker={"color": DISTANCE_COLOR},
            hovertemplate="Week from %{x|%d %b %Y}<br><b>%{y:.1f} km</b><extra></extra>",
        )
    )

    subtitle = ""
    if peers is not None and peers.get("status") == "ok":
        weekly_km = peers["comparison"]["weekly_km"]
        fig.add_hrect(
            y0=weekly_km["p25"],
            y1=weekly_km["p75"],
            fillcolor=PEER_BAND_COLOR,
            opacity=0.6,
            line_width=0,
            layer="below",  # behind the bars
        )
        # The subtitle explains the band, so no legend is needed and no text covers the bars.
        subtitle = (
            f"Band: typical for {peers['level']} runners "
            f"({weekly_km['p25']:.0f}–{weekly_km['p75']:.0f} km)"
        )

    fig.update_layout(bargap=0.25, barcornerradius=4)
    fig.update_yaxes(title_text="km", rangemode="tozero")
    return _apply_base_layout(fig, "Distance per week", subtitle)


def _pace_ticks(paces: list[float]) -> list[float]:
    """Tick positions for a pace axis: every 15 s, or every 30 s if the range is wide."""
    low, high = min(paces), max(paces)
    step = 0.25 if high - low <= 1.5 else 0.5  # in minutes: 15 s or 30 s
    first = math.floor(low / step) * step
    last = math.ceil(high / step) * step
    count = round((last - first) / step) + 1
    return [first + i * step for i in range(count)]


def pace_chart(trend: dict) -> go.Figure:
    """A line of the average pace per week (weeks without runs are skipped).

    trend: the result of pace_trend() with status "ok".
    The y-axis is reversed, so UP means FASTER (a lower pace), which is easier to read.
    """
    week_starts = [week["week_start"] for week in trend["weekly_pace"]]
    paces = [week["pace_min_km"] for week in trend["weekly_pace"]]
    pace_texts = [week["pace"] for week in trend["weekly_pace"]]

    fig = go.Figure(
        go.Scatter(
            x=week_starts,
            y=paces,
            mode="lines+markers",
            line={"color": PACE_COLOR, "width": 2},
            marker={"color": PACE_COLOR, "size": 8, "line": {"color": "white", "width": 2}},
            customdata=pace_texts,
            hovertemplate="Week from %{x|%d %b %Y}<br><b>%{customdata} min/km</b><extra></extra>",
        )
    )

    # Plotly can't show 5.5 as "5:30" by itself, so we give it the tick labels.
    ticks = _pace_ticks(paces)
    fig.update_yaxes(
        title_text="min/km (up = faster)",
        autorange="reversed",
        tickvals=ticks,
        ticktext=[format_pace(tick) for tick in ticks],
    )
    return _apply_base_layout(fig, "Average pace per week")
