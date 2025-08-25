from fvm_solver.models.fluid_config import FluidConfiguration
import numpy as np


def compute_cell_fluxes(U: np.ndarray, fluid_cfg: FluidConfiguration) -> tuple[np.ndarray]:
    # Extract from context
    gamma = fluid_cfg.gamma

    # Alocate arrays for x-flux (F) and y-flux (G)
    F = np.zeros_like(U)
    G = np.zeros_like(U)

    # Extract conservative vars
    rho  = U[:, :, 0]
    rhou = U[:, :, 1]
    rhov = U[:, :, 2]
    rhoE = U[:, :, 3]

    # Compute primitive vars for x- and y-fluxes
    u = rhou / rho
    v = rhov / rho
    E = rhoE / rho
    p = (gamma - 1.0) * rho * ( E - 0.5*(u**2 + v**2) )
    p = (gamma - 1) * (rhoE - 0.5 * rho * (u**2 + v**2))
    H = E + p / rho

    # Physical flux in x
    F[:, :, 0] = rhou
    F[:, :, 1] = rhou*u + p
    F[:, :, 2] = rhou*v
    F[:, :, 3] = rhou*H
    
    # Physical flux in y
    G[:, :, 0] = rhov
    G[:, :, 1] = rhov*u
    G[:, :, 2] = rhov*v + p
    G[:, :, 3] = rhov*H

    return F, G
