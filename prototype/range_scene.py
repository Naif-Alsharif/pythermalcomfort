"""Class-based prototype for thermal range scenes."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


class RangeScene:
    """Small class wrapper for setting up, computing, and plotting a range scene."""

    def __init__(self, model_func: Callable[..., Any]) -> None:
        # Store the model function (for example: pmv_ppd_iso, utci)
        self.model_func = model_func

        # User-configurable state (filled later by helper methods)
        self.x_axis: dict[str, Any] = {}
        self.y_axis: dict[str, Any] = {}
        self.linked_inputs: dict[str, str] = {}
        self.fixed_params: dict[str, Any] = {}
        self.metric_name: str | None = None

        # Computed arrays and plotting handles (set later)
        self.grid_x: np.ndarray | None = None
        self.grid_y: np.ndarray | None = None
        self.metric_grid: np.ndarray | None = None
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None

        # Internal helper state used when preparing model calls
        self._signature: inspect.Signature | None = None
        self._allowed_args: set[str] = set()
        self._accepts_var_kwargs: bool = False
        self._x_var: str | None = None
        self._y_var: str | None = None

    def axes(
        self,
        *,
        x_name: str,
        x_min: float,
        x_max: float,
        x_step: float,
        y_name: str,
        y_min: float,
        y_max: float,
        y_step: float,
    ) -> RangeScene:
        # Configure x/y axis names and ranges for later grid creation
        if x_step <= 0 or y_step <= 0:
            raise ValueError("x_step and y_step must be positive.")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("Axis min must be smaller than max.")

        self.x_axis = {
            "name": x_name,
            "min": float(x_min),
            "max": float(x_max),
            "step": float(x_step),
        }
        self.y_axis = {
            "name": y_name,
            "min": float(y_min),
            "max": float(y_max),
            "step": float(y_step),
        }
        self._x_var = x_name
        self._y_var = y_name
        # Config changed: drop cached computed grids.
        self.grid_x = None
        self.grid_y = None
        self.metric_grid = None
        return self

    def link(self, **kwargs: str) -> RangeScene:
        # Link model inputs (example: tr="tdb")
        # Meaning: when the grid uses "tdb", also pass that value to "tr"
        self.linked_inputs.update(kwargs)
        # Config changed: drop cached computed grids.
        self.grid_x = None
        self.grid_y = None
        self.metric_grid = None
        return self

    def parameters(self, **kwargs: Any) -> RangeScene:
        # Store fixed parameters that do not change across the grid
        self.fixed_params.update(kwargs)
        # Config changed: drop cached computed grids.
        self.grid_x = None
        self.grid_y = None
        self.metric_grid = None
        return self

    def metric(self, name: str) -> RangeScene:
        # Store the metric field name to extract from model result
        self.metric_name = name
        # Config changed: drop cached computed grids.
        self.grid_x = None
        self.grid_y = None
        self.metric_grid = None
        return self

    def _extract_metric_value_for_name(self, result: Any, metric_name: str) -> float:
        """Extract one scalar metric value using an explicit metric name."""
        # 1) Most pythermalcomfort model results are dataclass-like objects.
        if hasattr(result, metric_name):
            return float(getattr(result, metric_name))

        # 2) Some outputs may be dict-like.
        if isinstance(result, dict) and metric_name in result:
            return float(result[metric_name])

        # 3) Common fallback names for quick PMV/UTCI-style usage.
        if metric_name == "pmv" and hasattr(result, "pmv"):
            return float(result.pmv)
        if metric_name == "utci" and hasattr(result, "utci"):
            return float(result.utci)

        msg = (
            f"Could not extract metric '{metric_name}' from result type "
            f"{type(result).__name__}."
        )
        raise ValueError(msg)

    def _read_model_signature(self) -> None:
        """Read and cache the model function signature once."""
        if self._signature is not None:
            return

        self._signature = inspect.signature(self.model_func)
        self._allowed_args = set(self._signature.parameters.keys())
        self._accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in self._signature.parameters.values()
        )

    def _validate_allowed_arguments(self, call_params: dict[str, Any]) -> None:
        """Raise an error if a parameter is not accepted by the model."""
        self._read_model_signature()
        if self._accepts_var_kwargs:
            return

        invalid = [name for name in call_params if name not in self._allowed_args]
        if invalid:
            invalid_str = ", ".join(sorted(invalid))
            msg = (
                f"Model does not accept parameter(s): {invalid_str}. "
                "Check axes/link/parameters settings."
            )
            raise ValueError(msg)

    def _apply_v_vr_alias(self, call_params: dict[str, Any]) -> dict[str, Any]:
        """Map v <-> vr if needed for model compatibility."""
        self._read_model_signature()
        out = dict(call_params)

        # If model expects vr but user provided v, remap v -> vr.
        if "vr" in self._allowed_args and "v" not in self._allowed_args:
            if "v" in out and "vr" not in out:
                out["vr"] = out.pop("v")

        # If model expects v but user provided vr, remap vr -> v.
        if "v" in self._allowed_args and "vr" not in self._allowed_args:
            if "vr" in out and "v" not in out:
                out["v"] = out.pop("vr")

        return out

    def _build_call_params(self, x_value: float, y_value: float) -> dict[str, Any]:
        """Build kwargs for one model call at a single grid point."""
        x_var = self._x_var or self.x_axis.get("name")
        y_var = self._y_var or self.y_axis.get("name")
        if not x_var or not y_var:
            raise ValueError("Axes are not configured. Call axes(...) first.")

        # Start from fixed inputs, then overwrite x/y for this grid point.
        params = dict(self.fixed_params)
        params[x_var] = float(x_value)
        params[y_var] = float(y_value)

        # Apply input links (example: tr='tdb').
        for target_name, source_name in self.linked_inputs.items():
            if source_name not in params:
                msg = (
                    f"Linked source '{source_name}' not found while setting "
                    f"'{target_name}'."
                )
                raise ValueError(msg)
            params[target_name] = params[source_name]

        params = self._apply_v_vr_alias(params)
        self._validate_allowed_arguments(params)
        return params

    def _extract_metric_value(self, result: Any) -> float:
        """Extract one scalar metric value from a model result."""
        if self.metric_name is None:
            raise ValueError("Metric is not configured. Call metric(...) first.")
        return self._extract_metric_value_for_name(result, self.metric_name)

    def _compute_metric_grid(self) -> RangeScene:
        """Compute metric values on a 2D grid defined by axes settings."""
        if not self.x_axis or not self.y_axis:
            raise ValueError("Axes are not configured. Call axes(...) first.")
        if self.metric_name is None:
            raise ValueError("Metric is not configured. Call metric(...) first.")

        # Build 1D axis arrays from configured min/max/step.
        # + small epsilon helps include the max endpoint when floating values are used.
        x_values = np.arange(
            self.x_axis["min"],
            self.x_axis["max"] + 1e-12,
            self.x_axis["step"],
            dtype=float,
        )
        y_values = np.arange(
            self.y_axis["min"],
            self.y_axis["max"] + 1e-12,
            self.y_axis["step"],
            dtype=float,
        )

        # Store mesh grid for later contour/heatmap plotting.
        self.grid_x, self.grid_y = np.meshgrid(x_values, y_values)

        # Create output matrix with same shape as the mesh.
        # Each cell [i, j] stores the metric at (x_values[j], y_values[i]).
        self.metric_grid = np.full(self.grid_x.shape, np.nan, dtype=float)

        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                call_params = self._build_call_params(x_val, y_val)
                result = self.model_func(**call_params)
                self.metric_grid[i, j] = self._extract_metric_value(result)

        return self

    def plot(
        self,
        *,
        ax: plt.Axes | None = None,
        cmap: str = "coolwarm",
        diverging_center: float | None = None,
        contour_metric: str | None = None,
        contour_levels: list[float] | None = None,
        contour_colors: list[str] | None = None,
        colorbar: bool = True,
    ) -> plt.Axes:
        """Compute data (if needed) and draw a heatmap with optional contours."""
        # Build the main metric grid if it does not exist yet.
        if self.metric_grid is None or self.grid_x is None or self.grid_y is None:
            self._compute_metric_grid()

        # At this point, grids must exist.
        if self.metric_grid is None or self.grid_x is None or self.grid_y is None:
            raise RuntimeError("Grid computation failed.")

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 7))
            ax = self.ax
        else:
            self.ax = ax
            self.fig = ax.figure

        # Optional diverging center (useful for PMV centered at 0).
        norm = None
        if diverging_center is not None:
            if np.isnan(self.metric_grid).all():
                raise ValueError(
                    "No valid metric values to plot. "
                    "Current scene configuration produced an all-NaN grid."
                )
            vmin = float(np.nanmin(self.metric_grid))
            vmax = float(np.nanmax(self.metric_grid))
            norm = TwoSlopeNorm(vmin=vmin, vcenter=diverging_center, vmax=vmax)

        # Draw heatmap using pcolormesh on the computed mesh.
        mesh = ax.pcolormesh(
            self.grid_x,
            self.grid_y,
            self.metric_grid,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )

        # Optional contour overlay.
        if contour_levels:
            # Use main metric contours by default.
            contour_grid = self.metric_grid

            # If contour_metric is different, compute a temporary grid for it.
            if contour_metric and contour_metric != self.metric_name:
                temp_grid = np.full(self.metric_grid.shape, np.nan, dtype=float)
                y_values = self.grid_y[:, 0]
                x_values = self.grid_x[0, :]
                for i, y_val in enumerate(y_values):
                    for j, x_val in enumerate(x_values):
                        call_params = self._build_call_params(
                            float(x_val), float(y_val)
                        )
                        result = self.model_func(**call_params)
                        temp_grid[i, j] = self._extract_metric_value_for_name(
                            result, contour_metric
                        )
                contour_grid = temp_grid

            contours = ax.contour(
                self.grid_x,
                self.grid_y,
                contour_grid,
                levels=contour_levels,
                colors=contour_colors or "black",
                linewidths=1.5,
            )
            ax.clabel(contours, inline=True, fontsize=9)

        # Simple labels from axis config.
        ax.set_xlabel(str(self.x_axis.get("name", "x")))
        ax.set_ylabel(str(self.y_axis.get("name", "y")))

        # Optional colorbar.
        if colorbar and self.fig is not None:
            label = self.metric_name or "metric"
            cbar = self.fig.colorbar(mesh, ax=ax)
            cbar.set_label(label)

        return ax
