from fvm_solver.models.mesh_config import MeshConfig
from fvm_solver.models.fluid_config import FluidConfiguration
from fvm_solver.models.solver_config import SolverConfiguration
from fvm_solver.solution import Solution
from fvm_solver.mesh import Mesh
from fvm_solver.solver.time_integration import get_rk4_step
from fvm_solver.solver.time_step import get_time_step
from fvm_solver.solver.mass_flow import compute_mass_flow_rate
from tqdm import tqdm
from typing import Union, Tuple
import numpy as np


class FVMSolver():

    def __init__(self, solver_cfg: SolverConfiguration, mesh_cfg: MeshConfig, fluid_cfg: FluidConfiguration):
        self.solver_cfg = solver_cfg
        self.fluid_cfg = fluid_cfg
        self.mesh_cfg = mesh_cfg

    def solve(self, init_solution: Solution, mesh: Mesh, track_mass_flow_and_residual: bool = False) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if track_mass_flow_and_residual:
            return self._solve_with_mass_flow(init_solution, mesh)
        else:
            return self._solve(init_solution, mesh)
    
    def _solve(self, init_solution: Solution, mesh: Mesh) -> None:
        solution = init_solution

        for step in tqdm(range(self.solver_cfg.iteration_limit)):
            dt = self._get_time_step(solution, mesh)
            solution, residual = self._get_rk4_step(solution, mesh, dt)

    def _solve_with_mass_flow(self, init_solution: Solution, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray]:
        solution = init_solution

        residuals = np.zeros((self.solver_cfg.iteration_limit))
        mass_in = np.zeros(self.solver_cfg.iteration_limit)
        mass_out = np.zeros(self.solver_cfg.iteration_limit)

        for step in tqdm(range(self.solver_cfg.iteration_limit)):
            dt = self._get_time_step(solution, mesh)
            solution, residual = self._get_rk4_step(solution, mesh, dt)
            residuals[step] = residual
            mass_in[step] = compute_mass_flow_rate(solution, mesh, 'inlet')
            mass_out[step] = compute_mass_flow_rate(solution, mesh, 'outlet')

        return mass_in, mass_out, residuals
            

    def _get_time_step(self, solution: Solution, mesh: Mesh) -> float:
        return get_time_step(
            U=solution.U,
            Xg=mesh.vertices_x,
            Yg=mesh.vertices_y,
            gamma=self.fluid_cfg.gamma,
            cfl=self.solver_cfg.cfl,
        )
    
    def _get_rk4_step(self, solution: Solution, mesh: Mesh, dt: float) -> Solution:
        return get_rk4_step(
            solution=solution,
            mesh=mesh,
            fluid_cfg=self.fluid_cfg,
            solver_cfg=self.solver_cfg,
            dt=dt,
        )


            