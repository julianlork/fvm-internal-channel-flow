from fvm_solver.models.mesh_config import MeshConfig, BOTTOM_BUMP
from fvm_solver.models.fluid_config import FluidConfiguration
from fvm_solver.models.solver_config import SolverConfiguration
from fvm_solver.mesh import Mesh
from fvm_solver.solution import Solution
from fvm_solver.solver.core import FVMSolver
from fvm_visualizer.mach import render_mach_number
from fvm_visualizer.mass_flow import render_mass_flow
import matplotlib.pyplot as plt


def run_shock_example() -> None:

    mesh_cfg = MeshConfig()
    fluid_cfg = FluidConfiguration()
    solver_cfg = SolverConfiguration()
    
    mesh = Mesh.create_from_config(mesh_cfg)
    solution = Solution.create_with_initial_flow(mesh_cfg, fluid_cfg)
    
    solver = FVMSolver(solver_cfg, mesh_cfg, fluid_cfg)
    solver.solve(solution, mesh)

    fig, ax = plt.subplots(figsize=(7, 3))
    render_mach_number(solution, mesh, fluid_cfg, ax=ax, title="")
    fig.savefig('./examples/shock_formation.png')


def run_mass_flow_study() -> None:
    # --- Mesh Constants & Configuration ---
    eps = 0.4                               # Bump Geometry Constant 
    Nx, Ny = 128, 64                        # Number of grid lines
    L_x = 3.0                               # Domain length in m
    L_y = 1.0                               # Domain heigth in m

    mesh_cfg = MeshConfig(
        num_grid_lines_x=Nx,
        num_grid_lines_y=Ny,
        domain_size_x=L_x,
        domain_size_y=L_y,
        bottom_shape=BOTTOM_BUMP(eps=eps),
    )

    # --- Solver Constants & Configuration ---
    num_iter = [20_000, 20_000]    
    cfl = 2.0                               # Courant-Friedrichs-Lewy Condition
    rk4_coeff = [1/4, 1/3, 1/2]             # Runge-Kutta Coefficients


    # --- Fluid Constants ---
    mach_inf_inlet = [0.05, 0.1] # Freestream Mach Numbers
    gamma = 1.4                             # Ratio of Specific Heats 
    R = 287.0                               # Specific Gas Constant of Air
    p_atm = 101300.0                        # Atmospheric Pressure
    T_atm = 288.0                           # Atmospheric Temperature

    # --- Plotting Setup ---
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 3))
    ax = ax.flatten()

    # --- Mach Study Loop ---
    for i, (m_inf, n_iter) in enumerate(zip(mach_inf_inlet, num_iter)):
        print(f"Running Mach-Study with M-Infinity: {m_inf}")

        fluid_cfg = FluidConfiguration(
            M_inf=m_inf,
            gamma=gamma,
            R=R,
            p_atm=p_atm,
            T_atm=T_atm,
        )

        solver_cfg = SolverConfiguration(
            iteration_limit=n_iter,
            cfl=cfl,
            rk4_coefficients=rk4_coeff,
        )

        mesh = Mesh.create_from_config(mesh_cfg)
        solution = Solution.create_with_initial_flow(mesh_cfg, fluid_cfg)
        solver = FVMSolver(solver_cfg, mesh_cfg, fluid_cfg)
        mass_in, mass_out, _ = solver.solve(solution, mesh, track_mass_flow_and_residual=True)

        render_mass_flow(mass_in, mass_out, ax=ax[i], title="$M_{\infty}$="+f"{m_inf}")

    fig.tight_layout()
    fig.savefig('./examples/mach_study_mass_flow.png', dpi=400)

def run_mach_study() -> None:
    
    # --- Mesh Constants & Configuration ---
    eps = 0.4                               # Bump Geometry Constant 
    Nx, Ny = 128, 64                        # Number of grid lines
    L_x = 3.0                               # Domain length in m
    L_y = 1.0                               # Domain heigth in m

    mesh_cfg = MeshConfig(
        num_grid_lines_x=Nx,
        num_grid_lines_y=Ny,
        domain_size_x=L_x,
        domain_size_y=L_y,
        bottom_shape=BOTTOM_BUMP(eps=eps),
    )

    # --- Solver Constants & Configuration ---
    num_iter = [50_000, 20_000, 20_000, 20_000]    
    cfl = 2.0                               # Courant-Friedrichs-Lewy Condition
    rk4_coeff = [1/4, 1/3, 1/2]             # Runge-Kutta Coefficients


    # --- Fluid Constants ---
    mach_inf_inlet = [0.01, 0.05, 0.1, 0.7] # Freestream Mach Numbers
    gamma = 1.4                             # Ratio of Specific Heats 
    R = 287.0                               # Specific Gas Constant of Air
    p_atm = 101300.0                        # Atmospheric Pressure
    T_atm = 288.0                           # Atmospheric Temperature

    # --- Plotting Setup ---
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))
    ax = ax.flatten()

    # --- Mach Study Loop ---
    for i, (m_inf, n_iter) in enumerate(zip(mach_inf_inlet, num_iter)):
        print(f"Running Mach-Study with M-Infinity: {m_inf}")

        fluid_cfg = FluidConfiguration(
            M_inf=m_inf,
            gamma=gamma,
            R=R,
            p_atm=p_atm,
            T_atm=T_atm,
        )

        solver_cfg = SolverConfiguration(
            iteration_limit=n_iter,
            cfl=cfl,
            rk4_coefficients=rk4_coeff,
        )

        mesh = Mesh.create_from_config(mesh_cfg)
        solution = Solution.create_with_initial_flow(mesh_cfg, fluid_cfg)
        solver = FVMSolver(solver_cfg, mesh_cfg, fluid_cfg)
        solver.solve(solution, mesh)

        render_mach_number(solution, mesh, fluid_cfg, ax=ax[i], title="$M_{\infty}$="+f"{m_inf}")

    fig.tight_layout()
    fig.savefig('./examples/mach_study_mach_number.png', dpi=400)


if __name__ == '__main__':
    run_shock_example()
    #run_mach_study()
    #run_mass_flow_study()