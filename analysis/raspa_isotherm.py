from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, Optional[float]]:
    n = len(xs)
    if n < 2:
        return {"slope": None, "intercept": None, "r_squared": None}

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0.0:
        return {"slope": None, "intercept": None, "r_squared": None}

    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    predictions = [slope * x + intercept for x in xs]
    residual = sum((y - pred) ** 2 for y, pred in zip(ys, predictions))
    total = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - residual / total if total > 0.0 else None
    return {"slope": slope, "intercept": intercept, "r_squared": r_squared}


def _origin_slope(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    denominator = sum(x * x for x in xs)
    if denominator <= 0.0:
        return None
    return sum(x * y for x, y in zip(xs, ys)) / denominator


def _logspace(start: float, stop: float, count: int) -> List[float]:
    if count <= 1:
        return [10.0 ** start]
    step = (stop - start) / (count - 1)
    return [10.0 ** (start + i * step) for i in range(count)]


def _fit_langmuir(pressures: Sequence[float], uptakes: Sequence[float]) -> Dict[str, Any]:
    if len(pressures) < 4 or min(pressures) <= 0.0 or max(uptakes) <= 0.0:
        return {"status": "insufficient_data"}

    p_min = min(pressures)
    p_max = max(pressures)
    b_min = 1.0 / (100.0 * p_max)
    b_max = 100.0 / p_min
    best: Optional[Tuple[float, float, float]] = None

    for b_value in _logspace(math.log10(b_min), math.log10(b_max), 600):
        fractions = [(b_value * p) / (1.0 + b_value * p) for p in pressures]
        denominator = sum(value * value for value in fractions)
        if denominator <= 0.0:
            continue
        q_sat = sum(value * uptake for value, uptake in zip(fractions, uptakes)) / denominator
        if q_sat <= 0.0:
            continue
        residual = sum(
            (uptake - q_sat * value) ** 2
            for value, uptake in zip(fractions, uptakes)
        )
        if best is None or residual < best[0]:
            best = (residual, b_value, q_sat)

    if best is None:
        return {"status": "fit_failed"}

    residual, b_value, q_sat = best
    uptake_mean = sum(uptakes) / len(uptakes)
    total = sum((uptake - uptake_mean) ** 2 for uptake in uptakes)
    r_squared = 1.0 - residual / total if total > 0.0 else None
    p_half = 1.0 / b_value
    saturation_fraction = max(uptakes) / q_sat if q_sat > 0.0 else None
    boundary_limited = (
        b_value <= b_min * 1.05
        or b_value >= b_max / 1.05
        or p_half < p_min / 3.0
        or p_half > p_max * 3.0
    )

    return {
        "status": "ok",
        "model": "single_site_langmuir",
        "q_saturation": q_sat,
        "affinity_b_per_bar": b_value,
        "half_loading_pressure_bar": p_half,
        "r_squared": r_squared,
        "observed_max_fraction_of_fitted_saturation": saturation_fraction,
        "boundary_limited": boundary_limited,
    }


def _interpolate_uptake(
    points: Sequence[Dict[str, Any]],
    target_pressure: float,
) -> Tuple[Optional[float], Optional[float], str]:
    for point in points:
        if math.isclose(point["pressure_bar"], target_pressure, rel_tol=1e-9, abs_tol=1e-12):
            return point["uptake"], point.get("uptake_error"), "sampled"

    for left, right in zip(points, points[1:]):
        p_left = left["pressure_bar"]
        p_right = right["pressure_bar"]
        if p_left < target_pressure < p_right:
            if p_left > 0.0:
                fraction = (
                    math.log(target_pressure) - math.log(p_left)
                ) / (math.log(p_right) - math.log(p_left))
                method = "log_pressure_interpolation"
            else:
                fraction = (target_pressure - p_left) / (p_right - p_left)
                method = "linear_pressure_interpolation"
            uptake = left["uptake"] + fraction * (right["uptake"] - left["uptake"])
            left_error = left.get("uptake_error")
            right_error = right.get("uptake_error")
            if left_error is not None and right_error is not None:
                error = math.sqrt(
                    ((1.0 - fraction) * left_error) ** 2
                    + (fraction * right_error) ** 2
                )
            else:
                error = None
            return uptake, error, method
    return None, None, "outside_sampled_range"


def _observed_half_max_pressure(points: Sequence[Dict[str, Any]]) -> Optional[float]:
    target = 0.5 * max(point["uptake"] for point in points)
    if points[0]["uptake"] >= target:
        return points[0]["pressure_bar"]

    for left, right in zip(points, points[1:]):
        q_left = left["uptake"]
        q_right = right["uptake"]
        if q_left <= target <= q_right and q_right > q_left:
            fraction = (target - q_left) / (q_right - q_left)
            p_left = left["pressure_bar"]
            p_right = right["pressure_bar"]
            if p_left > 0.0:
                return math.exp(
                    math.log(p_left)
                    + fraction * (math.log(p_right) - math.log(p_left))
                )
            return p_left + fraction * (p_right - p_left)
    return None


def _shape_label(
    point_count: int,
    langmuir: Dict[str, Any],
    slope_decay_ratio: Optional[float],
) -> str:
    if point_count < 4:
        return "insufficient_points_for_shape"

    saturation_fraction = langmuir.get("observed_max_fraction_of_fitted_saturation")
    fit_quality = langmuir.get("r_squared")
    reliable_fit = (
        langmuir.get("status") == "ok"
        and not langmuir.get("boundary_limited")
        and fit_quality is not None
        and fit_quality >= 0.90
    )
    if reliable_fit and saturation_fraction is not None:
        if (
            saturation_fraction >= 0.95
            and slope_decay_ratio is not None
            and slope_decay_ratio <= 0.05
        ):
            return "plateau_reached"
        if saturation_fraction >= 0.80:
            return "approaching_saturation"
    if slope_decay_ratio is not None and slope_decay_ratio <= 0.25:
        return "strongly_curving_but_plateau_unconfirmed"
    return "capacity_still_rising"


def analyze_isotherm_series(series: Dict[str, Any]) -> Dict[str, Any]:
    raw_points = series.get("points") or []
    clean_by_pressure: Dict[float, Dict[str, Any]] = {}
    duplicate_pressures: List[float] = []

    for raw in raw_points:
        pressure = _finite_float(raw.get("pressure_bar"))
        uptake = _finite_float(raw.get("uptake"))
        if uptake is None:
            uptake = _finite_float(raw.get("uptake_excess"))
        if pressure is None or uptake is None or pressure <= 0.0:
            continue
        point = {
            "pressure_bar": pressure,
            "uptake": uptake,
            "uptake_error": _finite_float(raw.get("uptake_error")),
            "source_file": raw.get("source_file") or raw.get("raspa_output_file"),
            "work_dir": raw.get("work_dir"),
        }
        if pressure in clean_by_pressure:
            duplicate_pressures.append(pressure)
        clean_by_pressure[pressure] = point

    points = [clean_by_pressure[key] for key in sorted(clean_by_pressure)]
    if len(points) < 2:
        return {
            "series_id": series.get("series_id"),
            "mof": series.get("mof"),
            "guest": series.get("guest"),
            "temperature_K": _finite_float(series.get("temperature_K")),
            "uptake_units": series.get("uptake_units"),
            "status": "insufficient_data",
            "n_points": len(points),
            "points": points,
            "limitations": ["At least two distinct positive pressure points are required."],
        }

    pressures = [point["pressure_bar"] for point in points]
    uptakes = [point["uptake"] for point in points]
    q_max = max(uptakes)

    low_candidates = [i for i, uptake in enumerate(uptakes) if uptake <= 0.20 * q_max]
    low_count = min(5, max(2 if len(points) == 2 else 3, len(low_candidates)))
    low_count = min(low_count, len(points))
    low_points = points[:low_count]
    low_pressures = [point["pressure_bar"] for point in low_points]
    low_uptakes = [point["uptake"] for point in low_points]
    low_linear = _linear_fit(low_pressures, low_uptakes)
    low_origin_slope = _origin_slope(low_pressures, low_uptakes)

    high_count = min(3, len(points))
    high_points = points[-high_count:]
    high_linear = _linear_fit(
        [point["pressure_bar"] for point in high_points],
        [point["uptake"] for point in high_points],
    )
    high_slope = high_linear.get("slope")
    slope_decay_ratio = None
    if low_origin_slope is not None and low_origin_slope > 0.0 and high_slope is not None:
        slope_decay_ratio = max(0.0, high_slope) / low_origin_slope

    decreasing_intervals = []
    log_elasticities = []
    for left, right in zip(points, points[1:]):
        delta = right["uptake"] - left["uptake"]
        tolerance = 0.0
        if left.get("uptake_error") is not None and right.get("uptake_error") is not None:
            tolerance = math.sqrt(left["uptake_error"] ** 2 + right["uptake_error"] ** 2)
        if delta < -tolerance:
            decreasing_intervals.append(
                {
                    "from_pressure_bar": left["pressure_bar"],
                    "to_pressure_bar": right["pressure_bar"],
                    "uptake_change": delta,
                }
            )
        if left["uptake"] > 0.0 and right["uptake"] > 0.0:
            elasticity = math.log(right["uptake"] / left["uptake"]) / math.log(
                right["pressure_bar"] / left["pressure_bar"]
            )
            log_elasticities.append(
                {
                    "pressure_mid_bar": math.sqrt(
                        left["pressure_bar"] * right["pressure_bar"]
                    ),
                    "dlog_uptake_dlog_pressure": elasticity,
                }
            )

    langmuir = _fit_langmuir(pressures, uptakes)
    observed_half_pressure = _observed_half_max_pressure(points)
    fit_quality = langmuir.get("r_squared")
    use_langmuir_knee = (
        langmuir.get("status") == "ok"
        and not langmuir.get("boundary_limited")
        and fit_quality is not None
        and fit_quality >= 0.90
    )
    if use_langmuir_knee:
        knee_pressure = langmuir.get("half_loading_pressure_bar")
        knee_method = "single_site_langmuir_half_loading"
    else:
        knee_pressure = observed_half_pressure
        knee_method = "observed_half_max_interpolation"

    for point in points:
        if use_langmuir_knee and knee_pressure and knee_pressure > 0.0:
            relative = point["pressure_bar"] / knee_pressure
            point["pressure_over_knee"] = relative
            if relative <= 0.1:
                point["curve_regime"] = "henry_like"
            elif relative <= 10.0:
                point["curve_regime"] = "transition"
            else:
                point["curve_regime"] = "near_saturation"
        else:
            point["pressure_over_knee"] = None
            point["curve_regime"] = "unresolved"

    windows = [(0.1, 1.0), (0.1, 10.0), (1.0, 10.0)]
    working_capacities: List[Dict[str, Any]] = []
    for low_pressure, high_pressure in windows:
        if low_pressure < pressures[0] or high_pressure > pressures[-1]:
            continue
        low_uptake, low_error, low_method = _interpolate_uptake(points, low_pressure)
        high_uptake, high_error, high_method = _interpolate_uptake(points, high_pressure)
        if low_uptake is None or high_uptake is None:
            continue
        error = None
        if low_error is not None and high_error is not None:
            error = math.sqrt(low_error ** 2 + high_error ** 2)
        working_capacities.append(
            {
                "desorption_pressure_bar": low_pressure,
                "adsorption_pressure_bar": high_pressure,
                "working_capacity": high_uptake - low_uptake,
                "working_capacity_error": error,
                "uptake_units": series.get("uptake_units"),
                "endpoint_methods": {
                    "desorption": low_method,
                    "adsorption": high_method,
                },
            }
        )

    measured_range_capacity = uptakes[-1] - uptakes[0]
    measured_range_error = None
    if points[0].get("uptake_error") is not None and points[-1].get("uptake_error") is not None:
        measured_range_error = math.sqrt(
            points[0]["uptake_error"] ** 2 + points[-1]["uptake_error"] ** 2
        )

    shape_label = _shape_label(len(points), langmuir, slope_decay_ratio)
    limitations = [
        "The Langmuir fit is a compact shape descriptor, not proof of a single adsorption-site mechanism.",
        "Fitted saturation is unreliable when the measured range does not bend toward a plateau.",
        "Working capacities using interpolated endpoints are descriptive estimates.",
    ]
    if duplicate_pressures:
        limitations.append(
            "Duplicate pressure points were present; the last value at each duplicate pressure was used."
        )
    if decreasing_intervals:
        limitations.append(
            "Statistically unresolved decreasing intervals can occur from Monte Carlo uncertainty; significant decreases are listed explicitly."
        )

    return {
        "series_id": series.get("series_id"),
        "mof": series.get("mof"),
        "guest": series.get("guest"),
        "temperature_K": _finite_float(series.get("temperature_K")),
        "uptake_units": series.get("uptake_units"),
        "status": "ok",
        "n_points": len(points),
        "pressure_range_bar": [pressures[0], pressures[-1]],
        "uptake_range": [min(uptakes), max(uptakes)],
        "points": points,
        "initial_region": {
            "n_points": low_count,
            "max_pressure_bar": low_pressures[-1],
            "slope_through_origin_per_bar": low_origin_slope,
            "linear_slope_per_bar": low_linear.get("slope"),
            "linear_intercept": low_linear.get("intercept"),
            "linear_r_squared": low_linear.get("r_squared"),
        },
        "high_pressure_region": {
            "n_points": high_count,
            "slope_per_bar": high_slope,
            "linear_r_squared": high_linear.get("r_squared"),
            "slope_to_initial_slope_ratio": slope_decay_ratio,
        },
        "langmuir_shape_fit": langmuir,
        "knee": {
            "pressure_bar": knee_pressure,
            "method": knee_method,
            "observed_half_max_pressure_bar": observed_half_pressure,
        },
        "shape_label": shape_label,
        "monotonicity": {
            "status": "monotonic_within_uncertainty" if not decreasing_intervals else "significant_decrease_detected",
            "significant_decreasing_intervals": decreasing_intervals,
        },
        "log_slope_profile": log_elasticities,
        "working_capacity": {
            "measured_pressure_range": {
                "desorption_pressure_bar": pressures[0],
                "adsorption_pressure_bar": pressures[-1],
                "working_capacity": measured_range_capacity,
                "working_capacity_error": measured_range_error,
                "uptake_units": series.get("uptake_units"),
            },
            "standard_windows": working_capacities,
        },
        "limitations": limitations,
    }


def analyze_isotherm_collection(series_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    analyses = [analyze_isotherm_series(series) for series in series_list]
    successful = [item for item in analyses if item.get("status") == "ok"]

    def ranking(metric_path: Sequence[Any], reverse: bool = True) -> List[Dict[str, Any]]:
        rows = []
        for item in successful:
            value: Any = item
            for key in metric_path:
                if isinstance(value, dict):
                    value = value.get(key)
                elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
                    value = value[key]
                else:
                    value = None
            value = _finite_float(value)
            if value is None:
                continue
            rows.append(
                {
                    "series_id": item.get("series_id"),
                    "mof": item.get("mof"),
                    "guest": item.get("guest"),
                    "temperature_K": item.get("temperature_K"),
                    "value": value,
                }
            )
        return sorted(rows, key=lambda row: row["value"], reverse=reverse)

    if not analyses:
        status = "insufficient_data"
        note = "No RASPA uptake series with multiple pressure points were found."
    elif not successful:
        status = "insufficient_data"
        note = "RASPA uptake series were found, but none had two distinct positive pressure points."
    elif len(successful) == 1:
        status = "single_isotherm"
        note = "One isotherm was characterized; cross-MOF curve comparison requires additional matched isotherms."
    else:
        status = "ok"
        note = "Multiple isotherms were characterized and ranked using curve-derived descriptors."

    return {
        "method": "isotherm_shape_analysis",
        "definition": {
            "initial_slope": "Low-loading uptake increase per bar, used as a finite-pressure affinity indicator.",
            "knee_pressure": "Pressure near half of fitted saturation loading when a reliable Langmuir shape fit is available.",
            "shape_label": "Whether the observed curve reaches, approaches, or does not establish a plateau.",
            "working_capacity": "Uptake difference between adsorption and desorption pressures.",
            "curve_regime": "Pressure classified relative to the fitted knee, avoiding fixed absolute pressure thresholds.",
        },
        "status": status,
        "note": note,
        "series": analyses,
        "rankings": {
            "initial_slope_descending": ranking(
                ["initial_region", "slope_through_origin_per_bar"]
            ),
            "observed_max_uptake_descending": ranking(["uptake_range", 1]),
            "fitted_saturation_descending": ranking(
                ["langmuir_shape_fit", "q_saturation"]
            ),
            "knee_pressure_ascending": ranking(["knee", "pressure_bar"], reverse=False),
        },
        "limitations": [
            "Compare isotherms only when MOF, guest, temperature, uptake basis, and force-field conventions are compatible.",
            "At least four well-spaced pressure points are recommended for knee and shape interpretation.",
        ],
    }
