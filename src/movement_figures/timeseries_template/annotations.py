"""Reusable Matplotlib annotations for session-clock timeseries.

``draw_annotations`` draws ordinary regions and point events. ``qc_annotations``
adapts one session's already-loaded Q_C trial table and raw event frame. The
adapter does no file reading, pose processing, or movement quantification.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes

# Passing a replacement mapping selects the annotation kinds to include. Copy a
# default entry before changing it, e.g. {**QC_REGION_STYLES["outbound"], "alpha": .2}.
QC_REGION_STYLES = {
    "outbound": {"label": "Outbound", "color": "#4c72b0", "alpha": 0.08},
    "inbound": {"label": "Inbound", "color": "#dd8452", "alpha": 0.12},
}
QC_EVENT_STYLES = {
    # These three keys refer to derived trial-table times; all other keys below
    # are case-sensitive prefixes of raw ExperimentEvents strings.
    "trial_start": {
        "label": "Trial boundary", "color": "#4c72b0", "linewidth": 0.8,
        "alpha": 0.6,
    },
    "poke": {
        "label": "Decision poke", "color": "black", "linestyle": "--",
        "linewidth": 0.9, "alpha": 0.65,
    },
    "trial_end": {
        "label": "Trial end", "color": "0.4", "linestyle": "--",
        "linewidth": 0.9, "alpha": 0.65,
    },
    "Start Trial": {
        "label": "Start Trial (event)", "color": "#2ca02c", "linestyle": ":",
        "linewidth": 1.0, "alpha": 0.85,
    },
    "Await poke - Cue Tone ON": {
        "label": "Sound cue ON", "color": "#9467bd", "linewidth": 1.2,
        "alpha": 0.9,
    },
    "NosePokes & CueTone OFF": {
        "label": "Cue OFF", "color": "#9467bd", "linestyle": ":",
        "linewidth": 0.9, "alpha": 0.5,
    },
}
QC_OUTCOME_STYLES = {
    "Success": {"label": "Success", "color": "#2e8b57", "marker": "^"},
    "Failure": {"label": "Failure", "color": "#d1495b", "marker": "v"},
    "Miss": {"label": "Miss", "color": "#888888", "marker": "o"},
}


@dataclass(frozen=True)
class Region:
    """A labelled interval in absolute session seconds; style goes to axvspan."""

    start: float
    end: float
    label: str
    style: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Event:
    """An absolute timestamp drawn as a line, or a marker at an axes height.

    ``height=None`` draws a vertical line. A height between 0 and 1 draws a
    marker using the x-axis transform, so it never depends on the trace units.
    """

    time: float
    label: str
    style: Mapping[str, Any] = field(default_factory=dict)
    height: float | None = None


@dataclass(frozen=True)
class Annotations:
    """Composable annotation records and information about omitted regions."""

    regions: tuple[Region, ...] = ()
    events: tuple[Event, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass
class AnnotationResult:
    """Drawn artists, deduplicated legend handles, and adapter notes."""

    artists: list[Artist]
    legend_handles: list[Artist]
    notes: tuple[str, ...]


def _style(spec, default_label):
    style = dict(spec)
    label = style.pop("label", default_label)
    return label, style


def _single_session(frame, name):
    if frame is None or "session" not in frame:
        return None
    sessions = frame["session"].dropna().unique()
    if len(sessions) > 1:
        raise ValueError(f"{name} must contain only one session; filter it first.")
    return sessions[0] if len(sessions) else None


def qc_annotations(
    trials: pd.DataFrame,
    events: pd.DataFrame | None = None,
    *,
    region_styles=None,
    event_styles=None,
    outcome_styles=None,
) -> Annotations:
    """Build annotations from one session's Q_C trials and optional raw events.

    Trials need finite ``start_time`` and ``end_time``. Phase bounds use the four
    ``outbound_*_time`` / ``inbound_*_time`` columns when present, otherwise
    ``tz_triggered_time`` separates ``start_time`` and ``end_time``. Missing
    target boundaries omit both phase spans and are reported in ``notes``.

    A closing event is labelled Decision poke only when ``ChosenPort`` is a
    nonnegative integer and the outcome is not Miss. Otherwise it is Trial end.
    A Miss therefore never implies a physical poke. Raw Start Trial events are
    separate from derived trial boundaries, which may use the previous close.

    Raw events need an ``Event`` column and numeric ``Time`` column or index in
    the SAME absolute clock as trials and pose. Styles are Matplotlib keyword
    mappings with an optional ``label``. ``None`` selects defaults; ``{}`` hides
    that category. For event styles, trial_start/poke/trial_end are reserved;
    other keys select raw event prefixes. Outcome markers accept ``height``
    (axes fraction, default .96) in addition to ordinary plot styles.
    """
    region_styles = QC_REGION_STYLES if region_styles is None else region_styles
    event_styles = QC_EVENT_STYLES if event_styles is None else event_styles
    outcome_styles = QC_OUTCOME_STYLES if outcome_styles is None else outcome_styles
    unknown_regions = set(region_styles) - {"outbound", "inbound"}
    if unknown_regions:
        raise ValueError(f"Unknown Q_C regions: {sorted(unknown_regions)}. Use Region for custom intervals.")
    missing = {"start_time", "end_time"} - set(trials.columns)
    if missing:
        raise ValueError(f"Trial table is missing columns: {sorted(missing)}.")
    trial_session = _single_session(trials, "trials")
    event_session = _single_session(events, "events")
    if trial_session is not None and event_session is not None and trial_session != event_session:
        raise ValueError("Trials and events belong to different sessions.")

    regions, points, missing_phases = [], [], 0
    for index, row in trials.iterrows():
        start, end = float(row["start_time"]), float(row["end_time"])
        if not np.isfinite([start, end]).all() or end < start:
            raise ValueError(f"Trial {index!r} has invalid start/end times.")
        target = row.get("tz_triggered_time", np.nan)
        bounds = {
            "outbound": (row.get("outbound_start_time", start), row.get("outbound_end_time", target)),
            "inbound": (row.get("inbound_start_time", target), row.get("inbound_end_time", end)),
        }
        flat_bounds = [value for pair in bounds.values() for value in pair]
        if any(pd.isna(value) for value in flat_bounds):
            missing_phases += 1
        else:
            ob_start, ob_end = map(float, bounds["outbound"])
            ib_start, ib_end = map(float, bounds["inbound"])
            if not np.isfinite(flat_bounds).all() or not start <= ob_start <= ob_end <= ib_start <= ib_end <= end:
                raise ValueError(f"Trial {index!r} has invalid or overlapping phase bounds.")
            for name, spec in region_styles.items():
                label, style = _style(spec, name.title())
                a, b = map(float, bounds[name])
                regions.append(Region(a, b, label, style))

        if "trial_start" in event_styles:
            label, style = _style(event_styles["trial_start"], "Trial boundary")
            points.append(Event(start, label, style))
        outcome = row.get("outcome", "")
        port = pd.to_numeric(row.get("ChosenPort", np.nan), errors="coerce")
        has_poke = pd.notna(port) and np.isfinite(port) and port >= 0 and port == int(port) and outcome != "Miss"
        closing_kind = "poke" if has_poke else "trial_end"
        if closing_kind in event_styles:
            label, style = _style(event_styles[closing_kind], closing_kind)
            points.append(Event(end, label, style))
        if outcome in outcome_styles:
            label, style = _style(outcome_styles[outcome], outcome)
            height = style.pop("height", 0.96)
            style = {"linestyle": "None", "markersize": 7, **style}
            points.append(Event(end, label, style, height=height))

    raw_styles = {name: spec for name, spec in event_styles.items()
                  if name not in {"trial_start", "poke", "trial_end"}}
    if events is not None and raw_styles:
        if "Event" not in events:
            raise ValueError("Raw events must have an Event column.")
        times = pd.to_numeric(events.get("Time", events.index), errors="raise")
        times = np.asarray(times, dtype=float)
        if not np.isfinite(times).all():
            raise ValueError("Raw event times must be finite session-clock seconds.")
        for prefix, spec in raw_styles.items():
            label, style = _style(spec, prefix)
            selected = events["Event"].astype("string").str.startswith(prefix, na=False).to_numpy(dtype=bool)
            points.extend(Event(float(t), label, style) for t in times[selected])

    notes = ()
    if missing_phases and region_styles:
        notes = (f"Phase spans omitted for {missing_phases} trial(s) with missing target/phase boundaries.",)
    return Annotations(tuple(regions), tuple(points), notes)


def draw_annotations(
    ax: Axes,
    annotations: Annotations,
    *,
    window: tuple[float, float] | None = None,
    time_offset: float = 0.0,
    legend: bool = True,
) -> AnnotationResult:
    """Draw on an existing timeseries axis and return artists and legend handles.

    ``window`` is an absolute session-time interval [start, end]; point events
    at both boundaries are included so a full trial shows its outcome.
    Display x = absolute time - ``time_offset``. Subtract the same offset from
    the plotted trace yourself.
    An explicit window sets x limits; otherwise the current limits are kept.
    Regions are clipped to that window. Y limits are always preserved. Existing
    labelled traces are retained in a legend with each label included once.
    """
    if not np.isfinite(time_offset):
        raise ValueError("time_offset must be finite seconds.")
    original_xlim, original_ylim = ax.get_xlim(), ax.get_ylim()
    if window is None:
        w0, w1 = sorted(x + time_offset for x in original_xlim)
    else:
        w0, w1 = map(float, window)
    if not np.isfinite([w0, w1]).all() or w1 <= w0:
        raise ValueError("window must contain two increasing finite times.")

    artists = []
    for region in annotations.regions:
        if not np.isfinite([region.start, region.end]).all() or region.end < region.start:
            raise ValueError(f"Invalid bounds for region {region.label!r}.")
        start, end = max(w0, region.start), min(w1, region.end)
        if start < end:
            artists.append(ax.axvspan(start - time_offset, end - time_offset,
                                      label=region.label, **dict(region.style)))
    for event in annotations.events:
        if not np.isfinite(event.time):
            raise ValueError(f"Invalid timestamp for event {event.label!r}.")
        if event.height is not None and not 0 <= event.height <= 1:
            raise ValueError("Event marker height must lie between 0 and 1.")
        if w0 <= event.time <= w1:
            x = event.time - time_offset
            if event.height is None:
                artists.append(ax.axvline(x, label=event.label, **dict(event.style)))
            else:
                defaults = {"marker": "o", "linestyle": "None", "zorder": 5, "clip_on": False}
                artists.extend(ax.plot([x], [event.height], transform=ax.get_xaxis_transform(),
                                       label=event.label, **(defaults | dict(event.style))))

    ax.set_xlim(original_xlim if window is None else (w0 - time_offset, w1 - time_offset))
    ax.set_ylim(original_ylim)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels, strict=True):
        unique.setdefault(label, handle)
    if legend and unique:
        ax.legend(list(unique.values()), list(unique), fontsize="small", framealpha=0.9)
    return AnnotationResult(artists, list(unique.values()), annotations.notes)


def annotate_qc_timeseries(ax, trials, events=None, *, window=None, time_offset=0.0,
                           region_styles=None, event_styles=None, outcome_styles=None,
                           legend=True) -> AnnotationResult:
    """Build the Q_C template and draw it; see qc_annotations and draw_annotations."""
    annotations = qc_annotations(trials, events, region_styles=region_styles,
                                 event_styles=event_styles, outcome_styles=outcome_styles)
    return draw_annotations(ax, annotations, window=window, time_offset=time_offset, legend=legend)
