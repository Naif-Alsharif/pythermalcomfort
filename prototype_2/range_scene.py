"""Prototype 2: class-based threshold-region scene."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


class Output(str, Enum):
    """Common model outputs we want to support in prototype_2."""

    PMV = "pmv"
    PPD = "ppd"
    UTCI = "utci"


class RangeScene:
    """Second-iteration scene API for threshold-filled region plots."""

    def __init__(self, model_func: Any) -> None:
        # Store selected thermal model function (e.g., pmv_ppd_iso, utci)
        self.model_func = model_func

        # Config placeholders
        self.x_name: str | None = None
        self.y_name: str | None = None
        self.x_min: float | None = None
        self.x_max: float | None = None
        self.y_min: float | None = None
        self.y_max: float | None = None
        # Private default link: tr follows tdb unless tr is fixed explicitly.
        self._default_links: dict[str, str] = {"tr": "tdb"}
        self.fixed_values: dict[str, Any] = {}

        # Artists created by plot(), used later by adjust_* helpers.
        self.line_artists: list[Line2D] = []
        self.fill_artists: list[PolyCollection] = []

        # Internal signature cache for safe model calls.
        self._signature: inspect.Signature | None = None
        self._allowed_args: set[str] = set()
        self._required_args: set[str] = set()
        self._accepts_var_kwargs: bool = False

    def axes(self, **axes_ranges: Any) -> RangeScene:
        """Set x/y axes with two keyword ranges.

        Example:
            scene.axes(tdb=(18, 34), rh=(20, 80))
        """
        self._read_model_signature()

        if len(axes_ranges) != 2:
            raise ValueError("axes() requires exactly two keyword ranges.")

        items = list(axes_ranges.items())
        x_name, x_range = items[0]
        y_name, y_range = items[1]

        if x_name == y_name:
            raise ValueError("x and y axis parameters must be different.")

        invalid = [name for name in (x_name, y_name) if name not in self._allowed_args]
        if invalid:
            invalid_str = ", ".join(invalid)
            msg = (
                f"axes() received invalid parameter(s): {invalid_str}. "
                "Use model argument names from the selected function."
            )
            raise ValueError(msg)

        def _parse_axis_range(param_name: str, value: Any) -> tuple[float, float]:
            if not isinstance(value, tuple | list) or len(value) != 2:
                msg = (
                    f"Axis '{param_name}' must be a tuple/list of length 2: (min, max)."
                )
                raise ValueError(msg)
            try:
                min_val = float(value[0])
                max_val = float(value[1])
            except (TypeError, ValueError) as exc:
                msg = f"Axis '{param_name}' range values must be numeric."
                raise ValueError(msg) from exc
            if min_val >= max_val:
                msg = (
                    f"Axis '{param_name}' requires min < max "
                    f"(got {min_val} >= {max_val})."
                )
                raise ValueError(msg)
            return min_val, max_val

        x_min, x_max = _parse_axis_range(x_name, x_range)
        y_min, y_max = _parse_axis_range(y_name, y_range)

        # Do not allow axis variables to be fixed constants.
        axis_conflicts = [
            name for name in (x_name, y_name) if name in self.fixed_values
        ]
        if axis_conflicts:
            conflict_str = ", ".join(axis_conflicts)
            msg = (
                f"fixed() already contains axis parameter(s): {conflict_str}. "
                "Remove them from fixed() when using axes()."
            )
            raise ValueError(msg)

        self.x_name = x_name
        self.y_name = y_name
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        return self

    def fixed(self, **kwargs: Any) -> RangeScene:
        """Set fixed model input values using model-argument names.

        Example:
            scene.fixed(vr=0.1, met=1.2, clo=0.5)
        """
        for key, value in kwargs.items():
            if key == self.x_name or key == self.y_name:
                msg = f"fixed() cannot set axis parameter '{key}'. Set it through axes(...)."
                raise ValueError(msg)
            self.fixed_values[key] = value
        return self

    def _read_model_signature(self) -> None:
        """Read model signature once and cache required/allowed argument names."""
        if self._signature is not None:
            return

        self._signature = inspect.signature(self.model_func)
        self._allowed_args = set(self._signature.parameters.keys())
        self._accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in self._signature.parameters.values()
        )

        required: set[str] = set()
        for name, p in self._signature.parameters.items():
            is_named_input = p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            if is_named_input and p.default is inspect._empty:
                required.add(name)
        self._required_args = required

    def _apply_v_vr_alias(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Map v <-> vr carefully based on what the model accepts.

        This keeps examples simple: user can set one airflow key and still work
        across models that expect either v (UTCI) or vr (PMV-family).
        """
        self._read_model_signature()
        out = dict(kwargs)

        expects_v = "v" in self._allowed_args
        expects_vr = "vr" in self._allowed_args

        if expects_vr and not expects_v and "v" in out and "vr" not in out:
            out["vr"] = out.pop("v")
        elif expects_v and not expects_vr and "vr" in out and "v" not in out:
            out["v"] = out.pop("vr")

        return out

    def _validate_call_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Validate model-call kwargs against signature."""
        self._read_model_signature()

        if not self._accepts_var_kwargs:
            invalid = sorted(k for k in kwargs if k not in self._allowed_args)
            if invalid:
                invalid_str = ", ".join(invalid)
                msg = f"Model does not accept parameter(s): {invalid_str}"
                raise ValueError(msg)

        missing = sorted(k for k in self._required_args if k not in kwargs)
        if missing:
            missing_str = ", ".join(missing)
            msg = f"Missing required parameter(s): {missing_str}"
            raise ValueError(msg)

    def _build_call_kwargs(self, x_value: float, y_value: float) -> dict[str, Any]:
        """Build minimal kwargs for one model call at one x/y point.

        Why build kwargs internally?
        - Public API stays compact (axes + fixed).
        - Model call still needs argument-name keys, so kwargs are assembled
          here in one controlled place.
        """
        if self.x_name is None or self.y_name is None:
            raise ValueError("Axes are not set. Call axes(...) first.")
        self._read_model_signature()

        call_kwargs: dict[str, Any] = {}

        # 1) Start from fixed values.
        call_kwargs.update(self.fixed_values)

        # 2) Add axis values for this point.
        call_kwargs[self.x_name] = float(x_value)
        call_kwargs[self.y_name] = float(y_value)

        # 3) Apply private default links (target gets source value).
        for target, source in self._default_links.items():
            # Fixed/explicit values win over default links.
            if (
                target in self._allowed_args
                and source in self._allowed_args
                and target not in call_kwargs
                and source in call_kwargs
            ):
                call_kwargs[target] = call_kwargs[source]

        # 4) Apply v/vr compatibility mapping and validate.
        call_kwargs = self._apply_v_vr_alias(call_kwargs)
        self._validate_call_kwargs(call_kwargs)
        return call_kwargs

    def _extract_output_value(self, result: Any, output: Output) -> float:
        """Extract scalar value from model result for a selected output."""
        if output is Output.PMV:
            if hasattr(result, "pmv"):
                return float(result.pmv)
            if isinstance(result, dict) and "pmv" in result:
                return float(result["pmv"])
            raise ValueError("Selected output PMV is not available in model result.")

        if output is Output.PPD:
            if hasattr(result, "ppd"):
                return float(result.ppd)
            if isinstance(result, dict) and "ppd" in result:
                return float(result["ppd"])
            raise ValueError("Selected output PPD is not available in model result.")

        if output is Output.UTCI:
            # Some implementations return an object with .utci
            if hasattr(result, "utci"):
                return float(result.utci)
            # Some may return dict-like data
            if isinstance(result, dict) and "utci" in result:
                return float(result["utci"])
            # Some may return UTCI directly as scalar
            try:
                return float(result)
            except Exception as exc:
                raise ValueError(
                    "Selected output UTCI could not be extracted from result."
                ) from exc

        msg = f"Unsupported output: {output}"
        raise ValueError(msg)

    def _compute_threshold_curves(
        self,
        *,
        output: Output,
        thresholds: list[float],
        x_step: float,
        y_step: float,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Compute x(y) threshold curves by scanning temperature for each RH.

        Prototype approach:
        - For each y value, scan x values ONCE to get z(x).
        - Reuse that z(x) for all thresholds to find first crossings.
        - Use linear interpolation between adjacent points.
        - If no crossing exists at this y:
          - use edge fallback (x_min/x_max) for filling,
          - keep NaN for line drawing.
        """
        if self.x_name is None or self.y_name is None:
            raise ValueError("Axes are not set. Call axes(...) first.")
        if x_step <= 0 or y_step <= 0:
            raise ValueError("x_step and y_step must be positive.")

        if (
            self.x_min is None
            or self.x_max is None
            or self.y_min is None
            or self.y_max is None
        ):
            raise ValueError("Axes are not fully set. Call axes(...) first.")

        x_values = np.arange(self.x_min, self.x_max + 1e-12, x_step)
        y_values = np.arange(self.y_min, self.y_max + 1e-12, y_step)

        fill_curves: list[np.ndarray] = []
        line_curves: list[np.ndarray] = []
        x_min = float(self.x_min)
        x_max = float(self.x_max)
        has_any_valid_output = False

        # Allocate one curve array per threshold.
        for _ in thresholds:
            fill_curves.append(np.full(len(y_values), np.nan, dtype=float))
            line_curves.append(np.full(len(y_values), np.nan, dtype=float))

        for i, y in enumerate(y_values):
            # Evaluate output along the x scan line at this y (once).
            z = np.full(len(x_values), np.nan, dtype=float)
            for j, x in enumerate(x_values):
                try:
                    kwargs = self._build_call_kwargs(float(x), float(y))
                    result = self.model_func(**kwargs)
                    z_val = self._extract_output_value(result, output)
                    z[j] = z_val
                    if np.isfinite(z_val):
                        has_any_valid_output = True
                except Exception:
                    # Keep this point as NaN and continue scanning.
                    z[j] = np.nan
                    continue

            # Entire y-row failed: keep as "no data" and continue.
            if not np.isfinite(z).any():
                continue

            # Reuse the same z(x) for each threshold.
            for k, threshold in enumerate(thresholds):
                # Convert to residual around threshold.
                r = z - float(threshold)

                # Find first crossing (or exact hit) along x.
                crossing_x = np.nan
                found_crossing = False
                for j in range(len(x_values) - 1):
                    x0, x1 = x_values[j], x_values[j + 1]
                    r0, r1 = r[j], r[j + 1]

                    if not np.isfinite(r0) or not np.isfinite(r1):
                        continue

                    # Exact hit at left point.
                    if r0 == 0.0:
                        crossing_x = float(x0)
                        found_crossing = True
                        break

                    # Sign change means threshold crossing in interval.
                    if r0 * r1 < 0.0:
                        # Linear interpolation in [x0, x1].
                        # x* = x0 + (0-r0)*(x1-x0)/(r1-r0)
                        crossing_x = float(x0 + (-r0) * (x1 - x0) / (r1 - r0))
                        found_crossing = True
                        break

                    # Exact hit at right point.
                    if r1 == 0.0:
                        crossing_x = float(x1)
                        found_crossing = True
                        break

                if found_crossing:
                    fill_curves[k][i] = crossing_x
                    line_curves[k][i] = crossing_x
                    continue

                # No crossing: decide whether boundary is outside left/right domain
                # so filled regions remain continuous.
                finite = np.isfinite(r)
                if not finite.any():
                    continue

                rf = r[finite]
                if np.all(rf < 0.0):
                    # Entire row is below threshold => boundary is to the right.
                    fill_curves[k][i] = x_max
                elif np.all(rf > 0.0):
                    # Entire row is above threshold => boundary is to the left.
                    fill_curves[k][i] = x_min
                else:
                    # Mixed/ambiguous without clear crossing due NaNs.
                    # Keep NaN (gap) in this row.
                    continue

        if not has_any_valid_output:
            raise ValueError(
                "No valid model outputs could be computed for this plot. "
                "Check fixed parameters and axis ranges."
            )

        return y_values, fill_curves, line_curves

    def plot(
        self,
        *,
        output: Output,
        levels: list[float],
        colors: list[str],
        x_step: float,
        y_step: float,
        ax: plt.Axes | None = None,
        legend: bool = True,
        legend_title: str = "Regions",
        legend_loc: str = "upper left",
    ) -> plt.Axes:
        """Draw threshold-filled regions (no heatmap) for the selected output."""
        if self.x_name is None or self.y_name is None:
            raise ValueError("Axes are not set. Call axes(...) first.")
        if x_step <= 0 or y_step <= 0:
            raise ValueError("x_step and y_step must be positive.")
        if len(levels) == 0:
            raise ValueError("levels must contain at least one threshold.")
        if (
            self.x_min is None
            or self.x_max is None
            or self.y_min is None
            or self.y_max is None
        ):
            raise ValueError("Axes are not fully set. Call axes(...) first.")

        # Keep thresholds sorted so regions are ordered from low to high.
        sorted_levels = sorted(float(v) for v in levels)

        # Number of regions is thresholds + 1.
        needed_colors = len(sorted_levels) + 1
        if len(colors) != needed_colors:
            msg = f"colors must have length {needed_colors} (got {len(colors)})."
            raise ValueError(msg)

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))

        y_values, fill_curves, line_curves = self._compute_threshold_curves(
            output=output,
            thresholds=sorted_levels,
            x_step=x_step,
            y_step=y_step,
        )

        x_lo = float(self.x_min)
        x_hi = float(self.x_max)
        left_const = np.full_like(y_values, x_lo, dtype=float)
        right_const = np.full_like(y_values, x_hi, dtype=float)

        # Store created artists for later access/customization.
        self.fill_artists = []
        self.line_artists = []

        # Build region boundaries:
        #   region 0:   [left_const, curve0]
        #   region i:   [curve(i-1), curve(i)]
        #   last region:[curve_last, right_const]
        region_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        if len(fill_curves) == 0:
            region_pairs = [(left_const, right_const)]
        else:
            region_pairs.append((left_const, fill_curves[0]))
            for i in range(len(fill_curves) - 1):
                region_pairs.append((fill_curves[i], fill_curves[i + 1]))
            region_pairs.append((fill_curves[-1], right_const))

        # Fill each region.
        for i, (x_left, x_right) in enumerate(region_pairs):
            valid = np.isfinite(x_left) & np.isfinite(x_right)
            if valid.any():
                poly = ax.fill_betweenx(
                    y_values[valid],
                    x_left[valid],
                    x_right[valid],
                    color=colors[i],
                )
                self.fill_artists.append(poly)

        # Draw threshold boundary lines.
        for curve in line_curves:
            valid = np.isfinite(curve)
            if valid.any():
                (line,) = ax.plot(
                    curve[valid],
                    y_values[valid],
                )
                self.line_artists.append(line)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(float(self.y_min), float(self.y_max))
        ax.set_xlabel(self.x_name)
        ax.set_ylabel(self.y_name)

        # Optional auto-legend for filled threshold regions.
        if legend:
            out_name = output.value.upper()
            labels: list[str] = []
            for i in range(len(colors)):
                if i == 0:
                    labels.append(f"{out_name} < {sorted_levels[0]:g}")
                elif i == len(colors) - 1:
                    labels.append(f"{out_name} > {sorted_levels[-1]:g}")
                else:
                    lo = sorted_levels[i - 1]
                    hi = sorted_levels[i]
                    labels.append(f"{lo:g} <= {out_name} <= {hi:g}")

            handles = [
                Patch(facecolor=colors[i], alpha=0.65, label=labels[i])
                for i in range(len(colors))
            ]
            ax.legend(handles=handles, title=legend_title, loc=legend_loc)

        return ax

    def adjust_lines(self, **kwargs: Any) -> RangeScene:
        """Apply native Matplotlib line kwargs to all threshold lines."""
        if not self.line_artists:
            raise ValueError("Call plot() first.")
        for line in self.line_artists:
            line.set(**kwargs)
        return self

    def adjust_fills(self, **kwargs: Any) -> RangeScene:
        """Apply native Matplotlib fill kwargs to all threshold regions."""
        if not self.fill_artists:
            raise ValueError("Call plot() first.")
        for fill in self.fill_artists:
            fill.set(**kwargs)
        return self
