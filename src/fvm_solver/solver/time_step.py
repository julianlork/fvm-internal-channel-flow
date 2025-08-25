import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def get_time_step(U: np.ndarray, Xg: np.ndarray, Yg: np.ndarray, gamma: float, cfl: float):

    num_cells_x, num_cells_y = Xg.shape[0]- 1, Xg.shape[1]-1 # number of physical cells
    dt_min = np.ones((num_cells_x, num_cells_y))

    for i in prange(num_cells_x):
        for j in range(num_cells_y):
            # Extract conservative variables
            rho   = U[i+1, j+1, 0]
            rhou  = U[i+1, j+1, 1]
            rhov  = U[i+1, j+1, 2]
            rhoE  = U[i+1, j+1, 3]

            # Convert to primitive variables
            u = rhou / rho
            v = rhov / rho
            p = (gamma - 1.0) * (rhoE - 0.5*rho*(u*u + v*v)) 
            c = (gamma * p / rho)**0.5

            # Compute Max wave speed in x, y
            lambda_x = abs(u) + c
            lambda_y = abs(v) + c

            # Extract cell vertices (ordering as explained in get_cell_areas)
            xA, yA = Xg[i, j], Yg[i, j]
            xB, yB = Xg[i, j+1], Yg[i, j+1]
            xC, yC = Xg[i+1, j+1], Yg[i+1, j+1]
            xD, yD = Xg[i+1, j], Yg[i+1, j]

            # Averaged dx, dy
            # Vectors
            ADx = xD - xA
            ADy = yD - yA
            BCx = xC - xB
            BCy = yC - yB

            ABx = xB - xA
            ABy = yB - yA
            DCx = xC - xD
            DCy = yC - yD

            # Edge lengths
            lenAD = np.hypot(ADx, ADy)
            lenBC = np.hypot(BCx, BCy)
            lenAB = np.hypot(ABx, ABy)
            lenDC = np.hypot(DCx, DCy)

            # compute averaged face length in x and y
            dx = 0.5*(lenAD + lenBC)
            dy = 0.5*(lenAB + lenDC)

            # Compute recommended timestep for cell
            dt_cell = cfl / ( (lambda_x / dx) + (lambda_y / dy) )

            dt_min[i, j] = dt_cell
                
    return np.min(dt_min)


