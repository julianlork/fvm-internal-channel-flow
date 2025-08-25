from fvm_solver.models.mesh_config import MeshConfig
from fvm_solver.models.fluid_config import FluidConfiguration
from fvm_solver.models.solver_config import SolverConfiguration
from fvm_solver.mesh import Mesh
from fvm_solver.solution import Solution
import numpy as np


def apply_boundary_conditions(U: np.ndarray, mesh: Mesh, fluid_cfg: FluidConfiguration,) -> Solution:
    U = apply_inlet_boundary_condition(U, fluid_cfg)
    U = apply_outlet_boundary_condition(U, fluid_cfg)
    U = apply_top_wall_boundary_condition(U, mesh)
    U = apply_bottom_wall_boundary_condition(U, mesh)
    return U
    

def apply_inlet_boundary_condition(U: np.ndarray, fluid_cfg: FluidConfiguration) -> np.ndarray:
    
    # Extract variables from context
    gamma = fluid_cfg.gamma
    T_atm = fluid_cfg.T_atm
    p_atm = fluid_cfg.p_atm
    M_inf = fluid_cfg.M_inf
    R     = fluid_cfg.R

    """Compute Averaged inlet Mach-Number from physical layer behind inlet"""
    # Extract conservative variables from first interior column (vertical physical layer behin ghost-cell inlet)
    rho  = U[1, :, 0]   
    rhou = U[1, :, 1]
    rhov = U[1, :, 2]
    rhoE = U[1, :, 3]

    # Compute primitive variables
    u = rhou / rho
    v = rhov / rho

    # Compute pressure
    p = (gamma - 1) * (rhoE - 0.5 * rho * (u**2 + v**2))

    # Compute speed of sound from pressure
    c = np.sqrt(gamma * p / rho)
    #if np.sum(np.isnan(c)) > 0:
    #    print("t")

    # Compute inlet Mach-number from physical cell properties
    M_inlet = np.sqrt(u**2 + v**2) / c
    M_inlet = np.mean(M_inlet)  # Average over the row

    """Compute primitive and conserved variables"""
    # Compute stagnation temperature and pressure using freestream values
    T_stag = T_atm * (1 + (gamma - 1) / 2 * M_inf**2)
    p_stag = p_atm * (1 + (gamma - 1) / 2 * M_inf**2)**(gamma / (gamma - 1))

    # Compute static inlet temperature & pressure from stagnation temperature & pressure and smoothed inlet Mach-number
    T_inlet = T_stag / (1 + (gamma - 1) / 2 * M_inlet**2)
    p_inlet = p_stag / (1 + (gamma - 1) / 2 * M_inlet**2)**(gamma / (gamma - 1))

    # Compute primitive 1variables
    rho_inlet = p_inlet / (R * T_inlet)
    c_inlet = np.sqrt(gamma * R * T_inlet)   
    u_inlet = M_inlet * c_inlet  
    v_inlet = 0.0
    E_inlet = p_inlet / ((gamma - 1) * rho_inlet) + 0.5 * (u_inlet**2 + v_inlet**2)   # E_internal + E_kinetic

    # Compute & Assign conserved variables
    U[0, :, 0] = rho_inlet
    U[0, :, 1] = rho_inlet * u_inlet
    U[0, :, 2] = rho_inlet * v_inlet
    U[0, :, 3] = rho_inlet * E_inlet

    return U


def apply_outlet_boundary_condition(U: np.ndarray, fluid_cfg: FluidConfiguration) -> np.ndarray:

    # Extract variables from context
    gamma = fluid_cfg.gamma
    p_atm = fluid_cfg.p_atm

    # Extract conservative variables from previous vertical physical layer
    rho_interior   = U[-2, :, 0]
    rho_u_interior = U[-2, :, 1]
    rho_v_interior = U[-2, :, 2]

    # Convert to primitive for interior cell
    rho_prim = rho_interior
    u_prim   = rho_u_interior / rho_interior
    v_prim   = rho_v_interior / rho_interior

    # Compute new primitive variables for outlet cells
    rho_outlet = rho_prim   # Extrapolate -> Copy from previous cell
    u_outlet = u_prim       # Extrapolate -> Copy from previous cell
    v_outlet = v_prim       # Extrapolate -> Copy from previous cell
    E_outlet = p_atm / ((gamma - 1) * rho_prim) + 0.5 * (u_prim**2 + v_prim**2)  # Fix atmospheric pressure in internal energy

    # Compute and assign conservative variables
    U[-1, :, 0] = rho_outlet
    U[-1:, :, 1] = rho_outlet * u_outlet
    U[-1:, :, 2] = rho_outlet * v_outlet
    U[-1:, :, 3] = rho_outlet * E_outlet

    return U


def apply_top_wall_boundary_condition(U: np.ndarray, mesh: Mesh) -> np.ndarray:
    # Extract variables from context
    num_cells_x = mesh.vertices_x.shape[0] - 1
    face_vectors_south = mesh.cell_face_normals[:,:,2,:] * (-1)
    
    for i in range(num_cells_x):  # iterate over x-dimension
        # Extract interior conservative variables
        rho_int  = U[i+1, -2, 0]
        rhou_int = U[i+1, -2, 1]
        rhov_int = U[i+1, -2, 2]
        rhoE_int = U[i+1, -2, 3]

        # Convert to primitive variables
        u_int = rhou_int / rho_int
        v_int = rhov_int / rho_int
        E_int = rhoE_int / rho_int

        # Compute internal thermodynamic energy
        e_int = E_int - 0.5*(u_int**2 + v_int**2)

        # Get face normal and normalize
        nx, ny = face_vectors_south[i, -1] / np.linalg.norm(face_vectors_south[i, -1])

        # Compute normal velocity
        u_n = u_int * nx + v_int * ny

        # Compute corrected velocity by removing the normal component & corrected total energy
        u_wall = u_int - u_n * nx
        v_wall = v_int - u_n * ny
        E_wall = e_int + 0.5 * (u_wall**2 + v_wall**2)

        # Apply boundary conditions in ghost cells
        U[i+1, -1, 0] = rho_int
        U[i+1, -1, 1] = rho_int * u_int
        U[i+1, -1, 2] = rho_int * v_wall
        U[i+1, -1, 3] = rho_int * E_wall

    return U


def apply_bottom_wall_boundary_condition(U: np.ndarray, mesh: Mesh) -> np.ndarray:
    # Extract variables from context
    num_cells_x = mesh.vertices_x.shape[0] - 1
    face_vectors_north = mesh.cell_face_normals[:,:,3,:] * (-1)

    for i in range(num_cells_x): # iterate over x-dimension
        # Extract conservative variables
        rho_int  = U[i+1, 1, 0]
        rhou_int = U[i+1, 1, 1]
        rhov_int = U[i+1, 1, 2]
        rhoE_int = U[i+1, 1, 3]

        # Convert to primitive variables
        u_int = rhou_int / rho_int
        v_int = rhov_int / rho_int
        E_int = rhoE_int / rho_int

        # Compute internal thermodynamic energy
        e_int = E_int - 0.5*(u_int**2 + v_int**2)

        # Get northern face normal for ghost cell top face
        nx, ny = face_vectors_north[i, 0] / np.linalg.norm(face_vectors_north[i, 0])
        
        # Compute normal velocity
        u_n = u_int * nx + v_int * ny

        # Compute corrected velocity by removing the normal component & corrected total energy
        u_wall = u_int - u_n * nx
        v_wall = v_int - u_n * ny
        E_wall = e_int + 0.5 * (u_wall**2 + v_wall**2)

        # Compute and assign conservative variables
        U[i+1, 0, 0] = rho_int
        U[i+1, 0, 1] = rho_int*u_wall
        U[i+1, 0, 2] = rho_int*v_wall
        U[i+1, 0, 3] = rho_int*E_wall

    return U