"""Adaptive comfort chart plotting for ASHRAE 55 and EN 16798."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import is_color_like
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pythermalcomfort.plots.matplotlib._shared import BasePlotResult

# ── constants ──────────────────────────────────────────────────────────────

_N_POINTS: int = 200

_CENTER_LINE_LABEL: str = "Comfort Temperature"
_CENTER_LINE_DEFAULTS: dict[str, Any] = {
    "color": "#333333",
    "linewidth": 1.5,
    "linestyle": "--",
}


# ── cooling effect ─────────────────────────────────────────────────────────


def _compute_ce(v: float | None) -> float:
    """Compute cooling effect offset from air speed.

    Based on EN 16798 Table B.3.  Applied to upper comfort boundaries
    when ``v`` >= 0.6 m/s.

    Returns 0.0 when *v* is ``None`` or below 0.6 m/s.
    """
    if v is None or v < 0.6:
        return 0.0
    if v < 0.9:
        return 1.2
    if v < 1.2:
        return 1.8
    return 2.2


def _apply_ce_to_band(
    t_rm: np.ndarray,
    t_cmf: np.ndarray,
    *,
    upper_offset: float,
    lower_offset: float,
    ce: float,
    slope: float,
    intercept: float,
    t_rm_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply cooling effect with a vertical transition edge.

    Returns (t_rm_out, lower, upper) arrays ready for fill_between.
    """
    lower = t_cmf + lower_offset
    upper_base = t_cmf + upper_offset

    if ce <= 0.0:
        return t_rm, lower, upper_base

    t_rm_transition = (25.0 - intercept - upper_offset) / slope

    if not (t_rm_range[0] < t_rm_transition < t_rm_range[1]):
        # Transition outside visible range
        upper = np.where(upper_base >= 25.0, upper_base + ce, upper_base)
        return t_rm, lower, upper

    # Insert transition point twice for vertical edge
    idx = int(np.searchsorted(t_rm, t_rm_transition))
    t_val = slope * t_rm_transition + intercept

    t_rm_out = np.insert(t_rm, idx, [t_rm_transition, t_rm_transition])
    lower_out = np.insert(
        lower,
        idx,
        [
            t_val + lower_offset,
            t_val + lower_offset,
        ],
    )
    upper_out = np.insert(upper_base, idx, [25.0, 25.0])

    # Apply ce from second inserted point onward
    ce_mask = np.zeros(len(upper_out), dtype=bool)
    ce_mask[idx + 1 :] = True
    upper_out = np.where(ce_mask, upper_out + ce, upper_out)

    return t_rm_out, lower_out, upper_out


# ── band definitions ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class _BandDef:
    """Internal definition of one comfort band."""

    key: str
    upper_offset: float
    lower_offset: float
    default_label: str
    default_color: str


_STANDARD_CONFIGS: dict[str, dict[str, Any]] = {
    "ashrae": {
        "slope": 0.31,
        "intercept": 17.8,
        "t_rm_range": (10.0, 33.5),
        "bands": [
            _BandDef("80", 3.5, -3.5, "80% Acceptability", "#B3D9FF"),
            _BandDef("90", 2.5, -2.5, "90% Acceptability", "#6BB3FF"),
        ],
    },
    "en": {
        "slope": 0.33,
        "intercept": 18.8,
        "t_rm_range": (10.0, 33.5),
        "bands": [
            _BandDef("cat_iii", 4.0, -5.0, "Category III", "#C5E0B4"),
            _BandDef("cat_ii", 3.0, -4.0, "Category II", "#A9D18E"),
            _BandDef("cat_i", 2.0, -3.0, "Category I", "#70AD47"),
        ],
    },
}

_BAND_KEYS: dict[str, list[str]] = {
    std: [b.key for b in cfg["bands"]] for std, cfg in _STANDARD_CONFIGS.items()
}


# ── public config ──────────────────────────────────────────────────────────


@dataclass
class BandsConfig:
    """Reusable comfort band configuration.

    Controls which bands are displayed and their appearance.
    Follows the same pattern as :class:`ThresholdsConfig`.

    Attributes:
        show: Band keys to display.  If ``None``, all bands are shown.

            - ASHRAE keys: ``"80"``, ``"90"``
            - EN keys: ``"cat_i"``, ``"cat_ii"``, ``"cat_iii"``

        labels: Custom labels for the visible bands.  Must have the same
            length as *show* (or the total number of bands if *show* is
            ``None``).  If ``None``, default labels are used.
        colors: Custom colors for the visible bands.  Same length rule
            as *labels*.  If ``None``, default colors are used.

    Example::

        config = BandsConfig(
            show=["90"],
            labels=["90% Comfort Zone"],
            colors=["#FF6B6B"],
        )
    """

    show: Sequence[str] | None = None
    labels: Sequence[str] | None = None
    colors: Sequence[str] | None = None

    _validated: bool = field(init=False, repr=False, compare=False, default=False)

    def _validate(self, standard: str) -> None:
        """Validate against a specific standard's band keys."""
        valid_keys = set(_BAND_KEYS[standard])
        n_visible = (
            len(self.show) if self.show is not None else len(_BAND_KEYS[standard])
        )

        if self.show is not None:
            invalid = [k for k in self.show if k not in valid_keys]
            if invalid:
                msg = (
                    f"Invalid band key(s): {', '.join(invalid)}. "
                    f"Valid keys for '{standard}': "
                    f"{', '.join(sorted(valid_keys))}"
                )
                raise ValueError(msg)

        if self.labels is not None:
            if len(self.labels) != n_visible:
                msg = f"labels must have length {n_visible} (got {len(self.labels)})."
                raise ValueError(msg)

        if self.colors is not None:
            if len(self.colors) != n_visible:
                msg = f"colors must have length {n_visible} (got {len(self.colors)})."
                raise ValueError(msg)
            bad = [c for c in self.colors if not is_color_like(c)]
            if bad:
                msg = f"Invalid color value(s): {', '.join(str(c) for c in bad)}"
                raise ValueError(msg)

        self._validated = True


# ── resolved band (internal) ──────────────────────────────────────────────


@dataclass(frozen=True)
class _ResolvedBand:
    """A band with all overrides applied, ready to render."""

    upper_offset: float
    lower_offset: float
    label: str
    color: str


# ── result container ───────────────────────────────────────────────────────


@dataclass
class AdaptivePlotResult(BasePlotResult):
    """Result from :meth:`AdaptivePlot.plot`.

    Attributes:
        fig: Matplotlib figure.
        ax: Matplotlib axes.
        center_line: The comfort temperature center line artist, or ``None``.
        fills: Filled comfort band artists (outermost first).
        legend: Legend artist, or ``None``.
    """

    center_line: Line2D | None
    fills: list[PolyCollection]
    legend: Legend | None


# ── main class ─────────────────────────────────────────────────────────────


class AdaptivePlot:
    """Adaptive comfort chart for ASHRAE 55 or EN 16798.

    Renders comfort bands directly from the standard's published formulas:

    - **ASHRAE 55**: ``t_cmf = 0.31 * t_running_mean + 17.8``
    - **EN 16798**: ``t_cmf = 0.33 * t_running_mean + 18.8``

    Unlike :class:`ThresholdPlot`, this class does **not** call the model
    function.  The comfort bands are defined by simple linear equations in
    the standards, so no grid evaluation is needed.

    An optional air speed parameter ``v`` can be set via :meth:`set_params`
    to apply the **cooling effect (ce)** to upper boundaries.  When
    ``v`` >= 0.6 m/s, the upper limits shift upward according to
    EN 16798 Table B.3:

    - 0.6 <= v < 0.9: ce = 1.2 °C
    - 0.9 <= v < 1.2: ce = 1.8 °C
    - v >= 1.2: ce = 2.2 °C

    Example::

        from pythermalcomfort.plots.matplotlib import AdaptivePlot

        result = AdaptivePlot("ashrae").plot(title="Adaptive Comfort (ASHRAE 55)")

    Band keys for selection and customization:

    - **ASHRAE**: ``"80"`` (80% acceptability), ``"90"`` (90% acceptability)
    - **EN**: ``"cat_i"`` (Category I), ``"cat_ii"`` (Category II),
      ``"cat_iii"`` (Category III)
    """

    def __init__(
        self,
        standard: Literal["ashrae", "en"],
        *,
        t_running_mean_range: tuple[float, float] | None = None,
    ) -> None:
        """Initialize an adaptive comfort chart builder.

        Args:
            standard: Comfort standard, ``'ashrae'`` or ``'en'``.
            t_running_mean_range: Optional ``(min, max)`` override for the
                x-axis.  Defaults to the standard's applicability range
                (10-33.5 °C).
        """
        std = standard.lower().strip()
        if std not in _STANDARD_CONFIGS:
            msg = f"Unknown standard '{standard}'. Must be 'ashrae' or 'en'."
            raise ValueError(msg)

        self._standard = std
        self._cfg = _STANDARD_CONFIGS[std]
        self._v: float | None = None
        self._bands_config: BandsConfig | None = None

        if t_running_mean_range is not None:
            lo, hi = float(t_running_mean_range[0]), float(t_running_mean_range[1])
            if lo >= hi:
                raise ValueError("t_running_mean_range must have min < max.")
            self._t_rm_range = (lo, hi)
        else:
            self._t_rm_range = self._cfg["t_rm_range"]

    def set_params(self, *, v: float | None = None) -> AdaptivePlot:
        """Set optional parameters.

        Args:
            v: Air speed in m/s.  When >= 0.6, a cooling effect (ce) is
                applied to upper comfort boundaries, shifting them upward.
                When ``None`` or < 0.6, no cooling effect is applied.

        Returns:
            Self, to support method chaining.
        """
        if v is not None:
            self._v = float(v)
        return self

    def set_bands(
        self,
        *,
        show: BandsConfig | Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
    ) -> AdaptivePlot:
        """Customize which comfort bands are displayed and their appearance.

        Accepts either a pre-built :class:`BandsConfig` or raw parameters.

        Args:
            show: A :class:`BandsConfig` instance, **or** a list of band
                keys to display.  When a ``BandsConfig`` is supplied,
                *labels* and *colors* must not be given separately.

                If ``None`` (and not a ``BandsConfig``), all bands are shown.

                - ASHRAE keys: ``"80"``, ``"90"``
                - EN keys: ``"cat_i"``, ``"cat_ii"``, ``"cat_iii"``

            labels: Custom labels for the visible bands.  Must have the
                same length as *show* (or the total band count if *show*
                is ``None``).
            colors: Custom colors for the visible bands.  Same length
                rule as *labels*.

        Returns:
            Self, to support method chaining.

        Example::

            # Raw parameters
            .set_bands(show=["90"], labels=["90% Zone"], colors=["#FF6B6B"])

            # BandsConfig (reusable)
            config = BandsConfig(show=["90"], labels=["90% Zone"])
            .set_bands(show=config)
        """
        if isinstance(show, BandsConfig):
            if labels is not None or colors is not None:
                raise ValueError(
                    "labels and colors must not be provided separately when "
                    "show is a BandsConfig instance.  Set them inside the "
                    "BandsConfig instead."
                )
            config = show
        else:
            config = BandsConfig(show=show, labels=labels, colors=colors)

        config._validate(self._standard)
        self._bands_config = config
        return self

    def _resolve_bands(self) -> list[_ResolvedBand]:
        """Return bands with all user overrides applied."""
        all_defs: list[_BandDef] = self._cfg["bands"]
        cfg = self._bands_config

        if cfg is not None and cfg.show is not None:
            show_set = set(cfg.show)
            visible_defs = [d for d in all_defs if d.key in show_set]
        else:
            visible_defs = list(all_defs)

        resolved: list[_ResolvedBand] = []
        for i, d in enumerate(visible_defs):
            label = d.default_label
            color = d.default_color
            if cfg is not None:
                if cfg.labels is not None:
                    label = str(cfg.labels[i])
                if cfg.colors is not None:
                    color = str(cfg.colors[i])
            resolved.append(
                _ResolvedBand(
                    upper_offset=d.upper_offset,
                    lower_offset=d.lower_offset,
                    label=label,
                    color=color,
                )
            )
        return resolved

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        xlabel: str | None = "Prevailing Mean Outdoor Temperature [°C]",
        ylabel: str | None = "Operative Temperature [°C]",
        legend: bool = True,
        grid: bool = True,
        show_center_line: bool = True,
        center_line_kws: Mapping[str, Any] | None = None,
        fill_kws: Mapping[str, Any] | None = None,
        legend_kws: Mapping[str, Any] | None = None,
    ) -> AdaptivePlotResult:
        """Render the adaptive comfort chart.

        Args:
            ax: Existing axes.  If ``None``, a new figure is created.
            title: Optional chart title.
            xlabel: X-axis label.  ``None`` to omit.
            ylabel: Y-axis label.  ``None`` to omit.
            legend: Whether to draw a legend.
            grid: Whether to display background grid lines.
            show_center_line: Whether to draw the comfort temperature
                center line.
            center_line_kws: Overrides for the center line (``ax.plot``).
            fill_kws: Shared overrides for all bands
                (``ax.fill_between``).  Per-band colors are set via
                :meth:`set_bands`.
            legend_kws: Overrides for the legend (``ax.legend``).

        Returns:
            :class:`AdaptivePlotResult` with figure, axes, and artists.
        """
        bands = self._resolve_bands()
        ce = _compute_ce(self._v)

        # Compute comfort temperature from standard formula
        t_rm = np.linspace(self._t_rm_range[0], self._t_rm_range[1], _N_POINTS)
        slope: float = self._cfg["slope"]
        intercept: float = self._cfg["intercept"]
        t_cmf = slope * t_rm + intercept

        # Create axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        fill_opts = dict(fill_kws or {})
        fill_opts.setdefault("alpha", 0.7)

        # Draw bands (outermost first for correct layering)
        fills: list[PolyCollection] = []
        for band in bands:
            t_rm_band, lower, upper = _apply_ce_to_band(
                t_rm,
                t_cmf,
                upper_offset=band.upper_offset,
                lower_offset=band.lower_offset,
                ce=ce,
                slope=slope,
                intercept=intercept,
                t_rm_range=self._t_rm_range,
            )
            fill = ax.fill_between(
                t_rm_band,
                lower,
                upper,
                color=band.color,
                **fill_opts,
            )
            fills.append(fill)

        # Draw center line
        center_line_artist: Line2D | None = None
        if show_center_line:
            cl_opts = dict(_CENTER_LINE_DEFAULTS)
            if center_line_kws:
                cl_opts.update(center_line_kws)
            (center_line_artist,) = ax.plot(t_rm, t_cmf, **cl_opts)

        # Legend (innermost band first, then center line)
        legend_artist: Legend | None = None
        if legend:
            lg_opts = dict(legend_kws or {})
            lg_opts.setdefault("loc", "lower right")
            lg_opts.setdefault("frameon", True)
            lg_opts.setdefault("framealpha", 0.9)

            handles: list[Any] = []
            for band in reversed(bands):
                handles.append(
                    Patch(
                        facecolor=band.color,
                        alpha=fill_opts.get("alpha", 0.7),
                        label=band.label,
                    )
                )
            if center_line_artist is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        label=_CENTER_LINE_LABEL,
                        **dict(_CENTER_LINE_DEFAULTS),
                    )
                )
            legend_artist = ax.legend(handles=handles, **lg_opts)

        # Grid
        if grid:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Labels and limits
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.set_xlim(self._t_rm_range)

        return AdaptivePlotResult(
            fig=fig,
            ax=ax,
            center_line=center_line_artist,
            fills=fills,
            legend=legend_artist,
        )
