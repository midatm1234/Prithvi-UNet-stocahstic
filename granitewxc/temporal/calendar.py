"""Calendar-aware temporal metadata for sequence-conditioned downscaling.

The CORDEX/PRISM workflows in this repository mix calendars: the SA ACCESS-CM2
predictor files declare a ``standard`` (Gregorian) calendar but physically omit
29 February, while the matching high-resolution targets retain it. NARR/PRISM is
a true Gregorian daily archive. A temporal model that assumes "adjacent index
implies adjacent day" is therefore wrong on real files, and the errors are
silent.

Everything here works from the *actual* time coordinate values, never from
positional indices:

* :func:`elapsed_days` uses calendar-correct differencing (``cftime`` and
  ``numpy.datetime64`` both supported), so a removed 29 February shows up as a
  two-day step rather than a one-day step.
* :func:`find_discontinuities` flags any step that is not the declared cadence,
  which is what drives recurrent-state resets.
* :func:`year_fraction` divides day-of-year by the length of *that* year in
  *that* calendar, so the seasonal phase does not drift in 360-day or no-leap
  files and does not jump in Gregorian leap years.
* :func:`build_time_features` refuses to emit hour-of-day features when the
  archive has a single constant time-of-day stamp, which is the daily case for
  every workflow currently in this repository. Fabricating a diurnal cycle from
  a constant ``12:00:00`` stamp would inject a feature with no information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "CalendarSpec",
    "TimeFeatureSpec",
    "build_time_features",
    "days_in_year",
    "elapsed_days",
    "find_discontinuities",
    "resolve_calendar",
    "time_key",
    "year_fraction",
]


#: Calendar aliases understood by CF conventions, mapped to a canonical name.
_CALENDAR_ALIASES: dict[str, str] = {
    "standard": "standard",
    "gregorian": "standard",
    "proleptic_gregorian": "proleptic_gregorian",
    "noleap": "noleap",
    "no_leap": "noleap",
    "365_day": "noleap",
    "365day": "noleap",
    "all_leap": "all_leap",
    "allleap": "all_leap",
    "366_day": "all_leap",
    "366day": "all_leap",
    "360_day": "360_day",
    "360day": "360_day",
    "julian": "julian",
}

#: Calendars with a fixed year length (no leap-year branching).
_FIXED_YEAR_LENGTH: dict[str, int] = {
    "noleap": 365,
    "all_leap": 366,
    "360_day": 360,
}


@dataclass(frozen=True)
class CalendarSpec:
    """A resolved CF calendar plus the nominal year length used for phase.

    ``nominal_days_in_year`` is only used for reporting and for choosing
    sensible defaults; :func:`year_fraction` always divides by the true length
    of the specific year in the specific calendar.
    """

    name: str
    nominal_days_in_year: float

    @property
    def has_leap_years(self) -> bool:
        return self.name in {"standard", "proleptic_gregorian", "julian"}

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name}(~{self.nominal_days_in_year:g} d/yr)"


def resolve_calendar(value: Any) -> CalendarSpec:
    """Resolve a CF calendar name, ``cftime`` instance, or array of times.

    Raises
    ------
    ValueError
        If the calendar cannot be determined or is not a CF calendar.
    """
    name: str | None = None

    if isinstance(value, CalendarSpec):
        return value
    if isinstance(value, str):
        name = value.strip().lower()
    elif isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=object) if not isinstance(value, np.ndarray) else value
        if arr.size == 0:
            raise ValueError("Cannot resolve a calendar from an empty time array.")
        if np.issubdtype(arr.dtype, np.datetime64):
            name = "proleptic_gregorian"
        else:
            return resolve_calendar(arr.reshape(-1)[0])
    else:
        cal = getattr(value, "calendar", None)
        if cal is not None:
            name = str(cal).strip().lower()
        elif isinstance(value, np.datetime64):
            name = "proleptic_gregorian"

    if name is None:
        raise ValueError(
            f"Unable to resolve a CF calendar from {type(value).__name__}; "
            "pass an explicit calendar name."
        )

    canonical = _CALENDAR_ALIASES.get(name)
    if canonical is None:
        raise ValueError(
            f"Unsupported calendar {name!r}. Supported: "
            f"{sorted(set(_CALENDAR_ALIASES))}."
        )

    if canonical in _FIXED_YEAR_LENGTH:
        nominal = float(_FIXED_YEAR_LENGTH[canonical])
    else:
        nominal = 365.2425 if canonical != "julian" else 365.25
    return CalendarSpec(name=canonical, nominal_days_in_year=nominal)


def _is_leap(year: int, calendar: str) -> bool:
    if calendar == "julian":
        return year % 4 == 0
    # standard / proleptic_gregorian
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def days_in_year(year: int, calendar: CalendarSpec | str) -> int:
    """Exact number of days in ``year`` under ``calendar``."""
    spec = resolve_calendar(calendar)
    fixed = _FIXED_YEAR_LENGTH.get(spec.name)
    if fixed is not None:
        return fixed
    return 366 if _is_leap(int(year), spec.name) else 365


def time_key(value: Any) -> tuple[int, int, int, int, int, int]:
    """Calendar-agnostic ``(y, m, d, H, M, S)`` key for an exact date join.

    This mirrors :meth:`granitewxc.datasets` behaviour used by the existing
    frame datasets so that sequence windows and frame samples agree on identity.
    """
    if all(hasattr(value, name) for name in ("year", "month", "day")):
        return (
            int(value.year),
            int(value.month),
            int(value.day),
            int(getattr(value, "hour", 0)),
            int(getattr(value, "minute", 0)),
            int(getattr(value, "second", 0)),
        )
    text = np.datetime_as_string(np.datetime64(value), unit="s")
    date_part, time_part = text.split("T")
    year, month, day = (int(p) for p in date_part.split("-"))
    hour, minute, second = (int(p) for p in time_part.split(":"))
    return year, month, day, hour, minute, second


def _day_of_year(value: Any, calendar: CalendarSpec) -> int:
    """1-based day of year for ``value`` under ``calendar``."""
    dayofyr = getattr(value, "dayofyr", None)
    if dayofyr is not None:
        return int(dayofyr)

    year, month, day, *_ = time_key(value)
    if calendar.name == "360_day":
        return (int(month) - 1) * 30 + int(day)
    if calendar.name in _FIXED_YEAR_LENGTH:
        cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        if calendar.name == "all_leap":
            cum = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        return cum[int(month) - 1] + int(day)
    leap = _is_leap(int(year), calendar.name)
    cum = (
        [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        if leap
        else [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    )
    return cum[int(month) - 1] + int(day)


def _fraction_of_day(value: Any) -> float:
    _, _, _, hour, minute, second = time_key(value)
    return (hour * 3600.0 + minute * 60.0 + second) / 86400.0


def year_fraction(
    times: Sequence[Any] | np.ndarray,
    calendar: CalendarSpec | str | None = None,
) -> np.ndarray:
    """Seasonal phase in ``[0, 1)`` for each timestamp.

    The divisor is the length of the *specific* year in the *specific*
    calendar, so a Gregorian leap year uses 366 and a no-leap year uses 365.
    This keeps ``sin``/``cos`` seasonal features continuous across year
    boundaries instead of drifting by a day per leap year.
    """
    spec = resolve_calendar(calendar if calendar is not None else times)
    values = list(np.asarray(times, dtype=object).reshape(-1)) if not isinstance(times, (list, tuple)) else list(times)
    out = np.empty(len(values), dtype=np.float64)
    for idx, value in enumerate(values):
        year = time_key(value)[0]
        doy = _day_of_year(value, spec)
        total = days_in_year(year, spec)
        # ``doy`` is 1-based; subtract one so 1 Jan 00:00 maps exactly to 0.0.
        out[idx] = ((doy - 1) + _fraction_of_day(value)) / float(total)
    return np.mod(out, 1.0)


def elapsed_days(
    times: Sequence[Any] | np.ndarray,
    calendar: CalendarSpec | str | None = None,
) -> np.ndarray:
    """Calendar-correct elapsed days between consecutive timestamps.

    Returns an array of length ``len(times)`` whose first entry is ``nan``
    (there is no preceding sample) and whose entry ``i`` is
    ``times[i] - times[i - 1]`` expressed in days.

    A removed 29 February appears as ``2.0``; the SA historical/end-century
    concatenation appears as a ~36160-day step. Both are discontinuities as far
    as recurrent state is concerned.
    """
    values = list(np.asarray(times, dtype=object).reshape(-1)) if not isinstance(times, (list, tuple)) else list(times)
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out

    first = values[0]
    if isinstance(first, np.datetime64) or (
        isinstance(times, np.ndarray) and np.issubdtype(np.asarray(times).dtype, np.datetime64)
    ):
        as_ns = np.asarray(times, dtype="datetime64[ns]").astype("int64")
        diffs = np.diff(as_ns) / 8.64e13  # ns per day
        out[1:] = diffs
        return out

    for idx in range(1, n):
        delta = values[idx] - values[idx - 1]
        out[idx] = delta.days + delta.seconds / 86400.0 + delta.microseconds / 8.64e10
    return out


def find_discontinuities(
    times: Sequence[Any] | np.ndarray,
    cadence_days: float,
    *,
    calendar: CalendarSpec | str | None = None,
    tolerance_days: float = 1e-6,
) -> np.ndarray:
    """Boolean mask marking samples that do **not** continue the previous one.

    ``result[0]`` is always ``True`` (a sequence must start somewhere). Any
    other ``True`` marks a step whose elapsed time differs from
    ``cadence_days`` by more than ``tolerance_days`` -- a missing date, a
    duplicate timestamp, a period concatenation, or an irregular interval. A
    non-positive step (duplicate or out-of-order time) is also flagged.
    """
    if cadence_days <= 0:
        raise ValueError(f"cadence_days must be positive, got {cadence_days}")
    deltas = elapsed_days(times, calendar)
    mask = np.zeros(deltas.shape[0], dtype=bool)
    if mask.size == 0:
        return mask
    mask[0] = True
    if mask.size > 1:
        finite = np.isfinite(deltas[1:])
        off_cadence = np.abs(deltas[1:] - cadence_days) > tolerance_days
        non_positive = deltas[1:] <= 0
        mask[1:] = (~finite) | off_cadence | non_positive
    return mask


@dataclass(frozen=True)
class TimeFeatureSpec:
    """Which temporal metadata channels are produced, and in what order.

    The list is explicit and introspectable so that a checkpoint can record it
    and a later run can refuse to load weights trained against a different
    feature layout.
    """

    names: tuple[str, ...]
    cadence_days: float
    calendar: str
    include_hour_of_day: bool = False
    include_lead_time: bool = False

    @property
    def dim(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "cadence_days": float(self.cadence_days),
            "calendar": self.calendar,
            "include_hour_of_day": bool(self.include_hour_of_day),
            "include_lead_time": bool(self.include_lead_time),
        }


def _has_subdaily_variation(times: Sequence[Any]) -> bool:
    """True only when the archive actually resolves time of day."""
    seen: set[tuple[int, int, int]] = set()
    for value in times:
        _, _, _, hour, minute, second = time_key(value)
        seen.add((hour, minute, second))
        if len(seen) > 1:
            return True
    return False


def build_time_features(
    times: Sequence[Any],
    *,
    cadence_days: float,
    calendar: CalendarSpec | str | None = None,
    context_length: int | None = None,
    include_hour_of_day: bool | str = "auto",
    include_lead_time: bool = False,
    lead_times_days: Sequence[float] | None = None,
    position_offset: int = 0,
    mark_sequence_start: bool = True,
) -> tuple[np.ndarray, TimeFeatureSpec]:
    """Build the temporal-metadata matrix for one contiguous sequence.

    Parameters
    ----------
    times:
        Timestamps of the sequence, in order. Must already be verified
        contiguous by the caller (see :func:`find_discontinuities`); this
        function reports the actual gaps it sees but does not split.
    cadence_days:
        Nominal sampling interval, e.g. ``1.0`` for daily data. Used to
        normalize the elapsed-time channel so a regular step is exactly ``0``.
    context_length:
        Length used to normalize the relative-position channel. Defaults to
        ``len(times)``.
    include_hour_of_day:
        ``"auto"`` (default) emits hour-of-day sin/cos only when the timestamps
        actually vary within the day. ``True`` forces them and raises if the
        stamps are constant, because a constant stamp carries no diurnal
        information and would train a dead feature.
    include_lead_time:
        Forecasting mode only. Requires ``lead_times_days``.
    position_offset:
        Index of ``times[0]`` measured from the last state reset, **not** from the
        start of this array. Chunked inference must pass the absolute position
        within the contiguous run; otherwise the ``state_age`` and
        ``is_sequence_start`` channels would depend on how the run happened to be
        chunked and the chunked result would not match the single-pass result.
    mark_sequence_start:
        Whether ``position_offset == 0`` really is a fresh sequence. A mid-run
        inference chunk passes ``False`` so it is not mislabelled as a start.

    Returns
    -------
    features:
        ``float32`` array of shape ``[len(times), spec.dim]``.
    spec:
        The resolved :class:`TimeFeatureSpec` describing the channel order.
    """
    values = list(times)
    if not values:
        raise ValueError("build_time_features requires at least one timestamp.")
    spec_cal = resolve_calendar(calendar if calendar is not None else values)
    n = len(values)
    ctx = int(context_length) if context_length else n

    subdaily = _has_subdaily_variation(values)
    if include_hour_of_day == "auto":
        use_hour = subdaily
    else:
        use_hour = bool(include_hour_of_day)
        if use_hour and not subdaily:
            raise ValueError(
                "include_hour_of_day=True but every timestamp shares the same "
                "time of day; a constant stamp carries no diurnal information. "
                "Use 'auto' for daily archives."
            )

    if include_lead_time and lead_times_days is None:
        raise ValueError("include_lead_time=True requires lead_times_days.")
    if include_lead_time and len(lead_times_days) != n:  # type: ignore[arg-type]
        raise ValueError("lead_times_days must match the number of timestamps.")

    phase = year_fraction(values, spec_cal)
    deltas = elapsed_days(values, spec_cal)

    columns: list[np.ndarray] = []
    names: list[str] = []

    columns.append(np.sin(2.0 * math.pi * phase))
    names.append("season_sin")
    columns.append(np.cos(2.0 * math.pi * phase))
    names.append("season_cos")

    # Normalized elapsed time: 0 for a regular step. The first frame has no
    # predecessor; report 0 (as if regular) and flag it separately below so the
    # model can distinguish "sequence start" from "regular step".
    rel_gap = np.zeros(n, dtype=np.float64)
    if n > 1:
        rel_gap[1:] = (deltas[1:] - cadence_days) / cadence_days
    rel_gap = np.nan_to_num(rel_gap, nan=0.0, posinf=0.0, neginf=0.0)
    # Compress the huge SA period-concatenation gap so it cannot dominate the
    # feature scale, while keeping small gaps (a dropped 29 Feb -> 1.0) linear.
    columns.append(np.sign(rel_gap) * np.log1p(np.abs(rel_gap)))
    names.append("log_gap_excess")

    # Both of the next two channels are defined relative to the last state reset,
    # never relative to this array's own extent, so they are invariant to how a
    # long run is split into inference chunks.
    offset = int(position_offset)
    absolute = np.arange(offset, offset + n, dtype=np.float64)

    is_start = np.zeros(n, dtype=np.float64)
    if mark_sequence_start and offset == 0:
        is_start[0] = 1.0
    columns.append(is_start)
    names.append("is_sequence_start")

    # ``state_age``: how much history stands behind this frame, saturating at the
    # configured context length. In a training window (offset 0) this is
    # i/(L-1) -- identical to the previous "context_position". At inference with
    # carried state it saturates at 1.0, which is the correct description of a
    # fully warmed-up state and is what the model saw at the end of every
    # training window.
    denom = float(max(ctx - 1, 1))
    columns.append(np.minimum(absolute, denom) / denom)
    names.append("state_age")

    if use_hour:
        frac = np.array([_fraction_of_day(v) for v in values], dtype=np.float64)
        columns.append(np.sin(2.0 * math.pi * frac))
        names.append("hour_sin")
        columns.append(np.cos(2.0 * math.pi * frac))
        names.append("hour_cos")

    if include_lead_time:
        lead = np.asarray(lead_times_days, dtype=np.float64) / cadence_days
        columns.append(lead)
        names.append("lead_time_steps")

    features = np.stack(columns, axis=1).astype(np.float32)
    resolved = TimeFeatureSpec(
        names=tuple(names),
        cadence_days=float(cadence_days),
        calendar=spec_cal.name,
        include_hour_of_day=use_hour,
        include_lead_time=bool(include_lead_time),
    )
    return features, resolved


def describe_time_axis(
    times: Sequence[Any],
    cadence_days: float,
    *,
    calendar: CalendarSpec | str | None = None,
) -> dict[str, Any]:
    """Human-readable audit of a time axis: gaps, duplicates, calendar.

    Used by the CLI ``describe`` subcommands and by the dataset builders to
    print what they actually found rather than what the file claims.
    """
    spec = resolve_calendar(calendar if calendar is not None else times)
    deltas = elapsed_days(times, spec)
    finite = deltas[1:][np.isfinite(deltas[1:])]
    unique, counts = np.unique(np.round(finite, 6), return_counts=True)
    breaks = find_discontinuities(times, cadence_days, calendar=spec)
    keys = [time_key(v) for v in times]
    duplicates = len(keys) - len(set(keys))
    return {
        "n_times": len(times),
        "calendar": spec.name,
        "cadence_days": float(cadence_days),
        "first": time_key(times[0]) if len(times) else None,
        "last": time_key(times[-1]) if len(times) else None,
        "step_histogram": {float(u): int(c) for u, c in zip(unique, counts)},
        "n_discontinuities": int(breaks.sum()),
        "discontinuity_indices": np.flatnonzero(breaks).tolist()[:64],
        "n_duplicate_timestamps": int(duplicates),
    }
