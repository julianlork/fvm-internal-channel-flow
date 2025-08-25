from fvm_solver.solution import Solution
from fvm_solver.models.fluid_config import FluidConfiguration
from fvm_solver.mesh import Mesh
from fvm_solver.utils import get_mach_number
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from typing import Optional


plt.style.use('seaborn-v0_8')


def render_mach_number(solution: Solution, mesh: Mesh, cfg: FluidConfiguration, ax: Optional[Axes] = None, title: Optional[str] = "Local Mach Number") -> Axes:
    Xg, Yg = mesh.vertices_x, mesh.vertices_y
    U = solution.U[1:-1, 1:-1]
    gamma = cfg.gamma
    mach = get_mach_number(U, gamma)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))
        
    cax = ax.pcolormesh(Xg, Yg, mach, shading='auto', cmap='magma')
    _   = ax.set_title(title)
    _   = ax.set_aspect("equal")
    _   = ax.set_axis_off()
    _   = ax.figure.colorbar(cax, ax=ax, label="$M_{\infty}$", fraction=0.046, pad=0.04)

    return ax

