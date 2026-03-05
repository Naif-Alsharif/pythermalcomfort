"""Simple PMV threshold-fill example using prototype_2 RangeScene."""

from __future__ import annotations

import matplotlib.pyplot as plt
from range_scene import Output, RangeScene

from pythermalcomfort.models import pmv_ppd_iso

scene = (
    RangeScene(model_func=pmv_ppd_iso)
    .axes(tdb=(18.0, 34.0), rh=(20.0, 100.0))
    .fixed(
        vr=0.10,
        met=1.2,
        clo=0.5,
        wme=0.0,
        limit_inputs=False,
        round_output=False,
    )
)

ax = scene.plot(
    output=Output.PMV,
    levels=[-0.5, 0.5],
    colors=["#4c78a8", "#f2f2f2", "#e15759"],
    x_step=0.5,
    y_step=2.0,
)

ax.set_title("PMV Threshold-Filled Regions")
ax.set_xlabel("Air temperature [degC]")
ax.set_ylabel("Relative humidity [%]")

# Extra: Native Matplotlib styling through scene helpers
scene.adjust_lines(color="black", linewidth=1.5, linestyle="--")
scene.adjust_fills(alpha=0.65)


plt.tight_layout()
plt.show()
