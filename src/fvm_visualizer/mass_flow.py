from fvm_solver.solution import Solution
from fvm_solver.models.fluid_config import FluidConfiguration
from matplotlib.patches import Rectangle
from fvm_solver.mesh import Mesh
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from typing import Optional
import numpy as np

plt.style.use('seaborn-v0_8')

def render_mass_flow(
    mass_in: np.ndarray,
    mass_out: np.ndarray,
    ax: Optional[Axes] = None,
    title: Optional[str] = "Mass Flow"
) -> Axes:

    x = np.arange(len(mass_in))
    mass_flow_min, mass_flow_max = sorted([mass_in[-1], mass_out[-1]])
    mass_flow_avg = (mass_flow_min + mass_flow_max) / 2
    y_min, y_max = np.min([mass_in, mass_out]), np.max([mass_in, mass_out])

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))

    label = (
        "$\dot{m}_{in} =$ " + f"{mass_in[-1]:4.2f} kg/s\n"
        "$\dot{m}_{out} =$ " + f"{mass_out[-1]:4.2f} kg/s\n"
        "$\Delta \ \dot{m} =$ "+ f"{abs(round(mass_in[-1], 2) - round(mass_out[-1], 2)):3.2f} kg/s"
    )

    # ax.fill_between(x, mass_in[-1], mass_out[-1], alpha=0.1, color="tab:blue",
    #                 label="Envelope (Δ Mass-Flow Convergence)")
    ax.plot(x, mass_in,  label='Mass-Flow at Inlet')
    ax.plot(x, mass_out, label='Mass-Flow at Outlet')
    ax.axvline(x[-1], label="Iteration Limit", color='tab:red', linestyle='--')

    ax.set_xlabel("Iteration Step")
    ax.set_ylabel("Mass-Flow [kg / s]")
    ax.set_title(title)
    ax.legend(loc='best')

    x_text_offset = abs(x[-1] - x[0]) * 0.05  # prct of x-range 
    y_text_offset = abs(y_max - y_min) * 0.10 # prct of y-range
    x_bbox_width = abs(x[-1] - x[0]) * 0.01  # prct of x-range

    convergence_bbox = Rectangle(
        xy=(x[-1] - x_bbox_width / 2, mass_flow_min),
        width=x_bbox_width,
        height=abs(mass_flow_max - mass_flow_min),
        edgecolor='tab:orange',
        lw=1.0,
        facecolor=None,
        fill=False,
    )

    ax.add_patch(convergence_bbox)
    ax.annotate(
        xy=(x[-1], mass_flow_avg),
        xytext=(x[-1]-x_text_offset, mass_flow_min-y_text_offset),
        text=label,
        arrowprops=dict(arrowstyle="-", lw=0.5, color='k'),
        ha="right", 
        va="top",   
    )

    return ax
