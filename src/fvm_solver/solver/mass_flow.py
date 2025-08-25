import numpy as np
from typing import Literal
from numba import jit
from fvm_solver.solution import Solution
from fvm_solver.mesh import Mesh


def compute_mass_flow_rate(solution: Solution, mesh: Mesh, boundary: Literal["inlet", "outlet"]) -> float:
    return _compute_mass_flow_rate(solution.U, mesh.vertices_y, boundary)


#@jit(nopython=True, fastmath=True, inline="always")
def _compute_mass_flow_rate(U: np.ndarray, Yg: np.ndarray, boundary: Literal["inlet", "outlet"]) -> float:

    if boundary == 'inlet':
        i = 1  # Leftmost face
    elif boundary == 'outlet':
        i = -2  # Rightmost face
    else:
        raise ValueError("Boundary must be either 'inlet' or 'outlet'.")
    
    j_max = U.shape[1]
    dy = np.diff(Yg[i, :])
    mass_flow = U[i, 1:j_max-1, 1] * dy # exclude ghost cells in U

    return np.sum(mass_flow)
