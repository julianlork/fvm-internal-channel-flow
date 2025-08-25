from fvm_solver.models.mesh_config import MeshConfig
from fvm_solver.models.fluid_config import FluidConfiguration
from typing import Self
import numpy as np


class Solution():

    def __init__(self, init_solution: np.ndarray):
        self.U = init_solution

    @staticmethod
    def create_with_initial_flow(mesh_cfg: MeshConfig, fluid_cfg: FluidConfiguration) -> Self:
        """
        Initialize solution with uniform freestream flow based on fluid config with one surrounding
        ghost layer.
        """
        nx = mesh_cfg.num_grid_lines_x + 1
        ny = mesh_cfg.num_grid_lines_y + 1
        U = np.zeros((nx, ny, 4))
        U = _apply_initial_flow(U, fluid_cfg)
        return Solution(init_solution=U)
    

def _apply_initial_flow(U: np.ndarray, fluid_cfg: FluidConfiguration) -> np.ndarray:
    # Extract from context
    rho_atm = fluid_cfg.rho_atm
    T_atm   = fluid_cfg.T_atm
    p_atm   = fluid_cfg.p_atm
    gamma   = fluid_cfg.gamma
    M_inf   = fluid_cfg.M_inf
    R       = fluid_cfg.R

    # Compute primitive variables
    rho_init = rho_atm
    c_init   = np.sqrt(gamma * R * T_atm)  
    u_init   = M_inf * c_init
    v_init   = 0.0                      
    E_init   = p_atm / ((gamma - 1) * rho_init) + 0.5*(u_init**2 + v_init**2)

    # Compute & Assign conserved variables
    U[:, :, 0] = rho_init
    U[:, :, 1] = rho_init * u_init
    U[:, :, 2] = rho_init * v_init
    U[:, :, 3] = rho_init * E_init
    return U