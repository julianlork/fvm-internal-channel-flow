import numpy as np


def get_mach_number(U: np.ndarray, gamma: float) -> np.ndarray:

    # Extract conservative variables
    rho  = U[:, :, 0]  # Density
    rhou = U[:, :, 1]  # Momentum in x-direction
    rhov = U[:, :, 2]  # Momentum in y-direction
    rhoE = U[:, :, 3]  # Total energy

    # Compute primitive variables
    u = rhou / rho
    v = rhov / rho
    velocity_magnitude = np.sqrt(u**2 + v**2)

    # Compute pressure
    p = (gamma - 1) * (rhoE - 0.5 * rho * (u**2 + v**2))

    # Compute speed of sound
    c = np.sqrt(gamma * p / rho)

    # Compute Mach number
    mach = velocity_magnitude / c

    return mach