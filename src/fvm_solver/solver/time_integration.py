from fvm_solver.models.fluid_config import FluidConfiguration
from fvm_solver.models.solver_config import SolverConfiguration
from fvm_solver.mesh import Mesh
from fvm_solver.solution import Solution
from fvm_solver.solver.boundary_condition import apply_boundary_conditions
from fvm_solver.solver.flux import compute_cell_fluxes
from fvm_solver.solver.residual import compute_flux_residual
import numpy as np


def update_in_time(R: float, dt: float, areas: np.ndarray, alpha: float):
    dU = np.zeros_like(R)
    dU[1:-1, 1:-1, :] = alpha * dt * R[1:-1,1:-1,:] / areas[..., np.newaxis]
    return dU


def get_rk4_step(
        solution: Solution,
        mesh: Mesh,
        fluid_cfg: FluidConfiguration,
        solver_cfg: SolverConfiguration,
        dt: float) -> Solution:
    
    """Stage 1"""
    Y1 = solution.U.copy()
    Y1 = apply_boundary_conditions(Y1, mesh, fluid_cfg)

    """Stage 2"""
    F, G = compute_cell_fluxes(Y1, fluid_cfg)
    R1 = compute_flux_residual(Y1, F, G, mesh, fluid_cfg)
    Y2 = solution.U - update_in_time(R1, dt, mesh.cell_areas, solver_cfg.rk4_coefficients[0])
    Y2 = apply_boundary_conditions(Y2, mesh, fluid_cfg)

    """Stage 3"""
    F, G = compute_cell_fluxes(Y2, fluid_cfg)
    R2 = compute_flux_residual(Y2, F, G, mesh, fluid_cfg)
    Y3 = solution.U - update_in_time(R2, dt, mesh.cell_areas, solver_cfg.rk4_coefficients[1])
    Y3 = apply_boundary_conditions(Y3, mesh, fluid_cfg)

    """Stage 4"""
    F, G = compute_cell_fluxes(Y3, fluid_cfg)
    R3 = compute_flux_residual(Y3, F, G, mesh, fluid_cfg)
    Y4 = solution.U - update_in_time(R3, dt, mesh.cell_areas, solver_cfg.rk4_coefficients[2])
    Y4 = apply_boundary_conditions(Y4, mesh, fluid_cfg)

    """Update"""
    F, G = compute_cell_fluxes(Y4, fluid_cfg)
    R4 = compute_flux_residual(Y4, F, G, mesh, fluid_cfg)
    U_new = solution.U - update_in_time(R4, dt, mesh.cell_areas, 1.0)
    U_new = apply_boundary_conditions(U_new, mesh, fluid_cfg)

    solution.U = U_new
    F, G = compute_cell_fluxes(U_new, fluid_cfg)
    residual = compute_flux_residual(U_new, F, G, mesh, fluid_cfg)
    L2_norm = np.sqrt(np.mean(residual[:,:,1:3] ** 2))

    return solution, L2_norm


