from fvm_solver.models.fluid_config import FluidConfiguration
from fvm_solver.mesh import Mesh
import numpy as np
from numba import jit, prange


def compute_flux_residual(U: np.ndarray, F: np.ndarray, G: np.ndarray, mesh: Mesh, fluid_cfg: FluidConfiguration):
    return _compute_flux_residual(U, F, G, mesh.cell_face_normals, fluid_cfg.gamma)


@jit(nopython=True, fastmath=True, inline="always")
def compute_gamma(U: np.ndarray, i: int, j: int, face_vector: np.ndarray, gamma: float, eta: float, kappa_4: float=1/254):
    # Extract conservative variables for current cell
    rho   = U[i, j, 0]
    rho_u = U[i, j, 1]
    rho_v = U[i, j, 2]
    rho_E = U[i, j, 3]

    # Compute velocity components & fuse
    u = rho_u / rho 
    v = rho_v / rho 
    vel = np.array([u, v])

    # Compute pressure and speed of sound
    p = (gamma - 1) * (rho_E - 0.5 * rho * (u**2 + v**2))
    c = np.sqrt(gamma * p / rho)

    # Compute magnitude of normal vector
    normal_mag = np.linalg.norm(face_vector)

    # Compute the dot product \v * ds\
    vel = np.ascontiguousarray(np.array([u, v]))
    face_vector = np.ascontiguousarray(face_vector)
    v_dot_normal = np.abs(np.dot(vel, face_vector))

    # Compute gamma_face 
    gamma_face = np.maximum(0.5 * kappa_4 * (v_dot_normal + c * normal_mag) - eta, 0)

    return gamma_face


@jit(nopython=True, fastmath=True, inline="always")
def compute_eta(U: np.ndarray, i: int, j: int, di: int, dj: int, normal: np.ndarray, gamma: float, kappa_2: float = 0.5) -> float:
    # Extract conservative variables for current cell
    rho  = U[i, j, 0]
    rhou = U[i, j, 1]
    rhov = U[i, j, 2]
    rhoE = U[i, j, 3]

    # Compute primitive variables
    u = rhou / rho
    v = rhov / rho
    vel = np.array([u, v])

    # Compute pressure & speed of sound
    p = (gamma - 1) * (rhoE - 0.5 * rho * (u**2 + v**2))
    c = np.sqrt(gamma * p / rho)

   # Compute max(nu_i, nu_{i+1})
    v0 = compute_nu(U, i, j, di, dj, gamma)
    v1 = compute_nu(U, i+di, j+dj, di, dj, gamma)
    max_v = np.maximum(v0, v1)

    # Compute dot product v*dS
    vel = np.ascontiguousarray(np.array([u, v]))
    normal = np.ascontiguousarray(normal)
    v_dot_normal = np.abs(np.dot(vel, normal))

    # Compute eta
    eta_face = 0.5 * kappa_2 * (v_dot_normal + c * np.linalg.norm(normal)) * max_v

    return eta_face


@jit(nopython=True, fastmath=True, inline="always")
def compute_nu(U: np.ndarray, i: int, j: int, di: int, dj: int, gamma: float) -> float:
    # Get search direction. We can take absolute value since we always take a predecessor and successor
    di, dj = abs(di), abs(dj)

    # Compute pressure at (i, j)
    p0 = (gamma - 1) * (U[i, j, 3] - 0.5 * U[i, j, 0] * ((U[i, j, 1] / U[i, j, 0])**2 + (U[i, j, 2] / U[i, j, 0])**2))

    if 0 <= i + di < U.shape[0] and 0 <= j + dj < U.shape[1]:
        p1 = (gamma - 1) * (U[i+di, j+dj, 3] - 0.5 * U[i+di, j+dj, 0] * ((U[i+di, j+dj, 1] / U[i+di, j+dj, 0])**2 + (U[i+di, j+dj, 2] / U[i+di, j+dj, 0])**2))
    else:
        p1 = p0  

    if 0 <= i - di < U.shape[0] and 0 <= j - dj < U.shape[1]:
        p2 = (gamma - 1) * (U[i-di, j-dj, 3] - 0.5 * U[i-di, j-dj, 0] * ((U[i-di, j-dj, 1] / U[i-di, j-dj, 0])**2 + (U[i-di, j-dj, 2] / U[i-di, j-dj, 0])**2))
    else:
        p2 = p0

    # Compute ν_i for the face
    nu_face = np.abs((p1 - 2 * p0 + p2) / (p1 + 2 * p0 + p2 + 1e-10))  # avoidd div by zero

    return nu_face


@jit(nopython=True, parallel=True)
def _compute_flux_residual(U: np.ndarray, F: np.ndarray, G: np.ndarray, cell_face_normals: np.ndarray, gamma: float):

    normals_east = cell_face_normals[:,:,0,:]
    normals_west = cell_face_normals[:,:,1,:]
    normals_north = cell_face_normals[:,:,2,:]
    normals_south = cell_face_normals[:,:,3,:]
    res = np.zeros_like(U)

    for i in prange(1, U.shape[0]-1):
        for j in range(1, U.shape[1]-1):

            # Compute eta 
            eta_e = compute_eta(U, i, j,  1,  0, normals_east[i-1, j-1], gamma)
            eta_w = compute_eta(U, i, j, -1,  0, normals_west[i-1, j-1], gamma)
            eta_n = compute_eta(U, i, j,  0,  1, normals_north[i-1, j-1], gamma)
            eta_s = compute_eta(U, i, j,  0, -1, normals_south[i-1, j-1], gamma)

            # Compute coefficienf for artifical dissipation for each face with correct indices
            gamma_e = compute_gamma(U, i, j, normals_east[i-1,j-1], gamma, eta_e)
            gamma_w = compute_gamma(U, i, j, normals_west[i-1,j-1], gamma, eta_w)
            gamma_n = compute_gamma(U, i, j, normals_north[i-1,j-1], gamma, eta_n)
            gamma_s = compute_gamma(U, i, j, normals_south[i-1,j-1], gamma, eta_s)

            # Artificial dissipation
            D_e = gamma_e * (U[i+1,   j] - U[  i,   j]) + eta_e * (U[i+1,   j] - U[  i,   j])
            D_w = gamma_w * (U[  i,   j] - U[i-1,   j]) + eta_w * (U[  i,   j] - U[i-1,   j])
            D_n = gamma_n * (U[  i, j+1] - U[  i,   j]) + eta_n * (U[  i, j+1] - U[  i,   j])
            D_s = gamma_s * (U[  i,   j] - U[  i, j-1]) + eta_s * (U[  i,   j] - U[  i, j-1])

            # Eastern face
            F_e   = 0.5 * (F[i, j, :] + F[i+1, j, :])  # average x-flux
            G_e   = 0.5 * (G[i, j, :] + G[i+1, j, :])  # average y-flux
            Phi_e = F_e * normals_east[i-1, j-1, 0] + G_e * normals_east[i-1, j-1, 1] - D_e

            # Western face
            F_w = 0.5 * (F[i-1, j, :] + F[i, j, :])
            G_w = 0.5 * (G[i-1, j, :] + G[i, j, :])
            Phi_w = F_w * normals_west[i-1, j-1, 0] + G_w * normals_west[i-1, j-1, 1] + D_w

            # Nothern face
            F_n = 0.5 * (F[i, j+1, :] + F[i, j, :])
            G_n = 0.5 * (G[i, j+1, :] + G[i, j, :])
            Phi_n= F_n * normals_north[i-1, j-1, 0] + G_n * normals_north[i-1, j-1, 1] - D_n

            # Southern face
            F_s = 0.5 * (F[i, j-1, :] + F[i, j, :])
            G_s = 0.5 * (G[i, j-1, :] + G[i, j, :])
            Phi_s= F_s * normals_south[i-1, j-1, 0] + G_s * normals_south[i-1, j-1, 1] + D_s

            # Compute net flux
            net_flux = Phi_e + Phi_n + Phi_w + Phi_s
            res[i,j] = net_flux 

    return res
