from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Self, Callable
from fvm_solver.models.mesh_config import MeshConfig
import numpy as np


class Mesh():
    def __init__(
            self,
            vertices_x: np.ndarray,
            vertices_y: np.ndarray,
            cell_areas: np.ndarray,
            cell_face_normals: np.ndarray,)  -> None:
        self.vertices_x = vertices_x # Shape (Nx+1, Ny+1)
        self.vertices_y = vertices_y  # Shape (Nx+1, Ny+1)
        self.cell_areas = cell_areas  # Shape (Nx, Ny, 1)
        self.cell_face_normals = cell_face_normals  # Shape (Nx, Ny, 4, 2) face stored as (E, W, N, S)

    @staticmethod
    def create_from_config(config: MeshConfig) -> Self:
        Xg, Yg = get_grid_points(
            Nx=config.num_grid_lines_x,
            Ny=config.num_grid_lines_y,
            Lx=config.domain_size_x,
            Yb=config.bottom_shape,
            Yt=config.top_shape,
        )

        face_normals = get_cell_face_normals(
            Nx=config.num_grid_lines_x,
            Ny=config.num_grid_lines_y,
            Xg=Xg,
            Yg=Yg,
        )

        cell_areas = get_cell_areas(
            Nx=config.num_grid_lines_x,
            Ny=config.num_grid_lines_y,
            Xg=Xg,
            Yg=Yg,
        )

        return Mesh(
            vertices_x=Xg,
            vertices_y=Yg,
            cell_areas=cell_areas,
            cell_face_normals=face_normals,
        )


def get_grid_points(
        Nx: int,
        Ny: int,
        Lx: float,
        Yb: callable,
        Yt: callable,) -> tuple[np.ndarray, np.ndarray]:
    # Generate x-coordinates
    dx = Lx / (Nx - 1)
    x = np.arange(0, Nx) * dx

    # Generate y-coordinates for top and bottom walls
    y_bottom = Yb(x)
    y_top    = Yt(x)
    dy       = (y_top - y_bottom) / (Ny - 1)

    # Allocate meshgrid
    Xg = np.zeros((Nx, Ny))
    Yg = np.zeros((Nx, Ny))

    for i in range(Nx):
        Yg[i, :] = y_bottom[i] + np.arange(0, Ny) * dy[i]
        Xg[i, :] = x[i]

    return Xg, Yg


def get_cell_face_normals(
        Nx: int,
        Ny: int, 
        Xg: np.ndarray,
        Yg: np.ndarray,) -> np.ndarray:
    
    face_normals = np.zeros((Nx - 1, Ny - 1, 4, 2))
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            xA, yA = Xg[  i,   j], Yg[  i,   j]
            xB, yB = Xg[  i, j+1], Yg[  i, j+1]
            xC, yC = Xg[i+1, j+1], Yg[i+1, j+1]
            xD, yD = Xg[i+1,   j], Yg[i+1,   j]

            face_normals[i, j, 0, :] = [-(yD - yC), (xD - xC)]  # east
            face_normals[i, j, 1, :] = [-(yB - yA), (xB - xA)]  # west
            face_normals[i, j, 2, :] = [-(yC - yB), (xC - xB)]  # north
            face_normals[i, j, 3, :] = [-(yA - yD), (xA - xD)]  # south

    return face_normals


def get_cell_areas(
        Nx: int,
        Ny: int,
        Xg: np.ndarray,
        Yg: np.ndarray,) -> np.ndarray:
    
    cell_areas = np.zeros((Nx - 1, Ny - 1))
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            xA, yA = Xg[  i,   j], Yg[  i,   j]
            xB, yB = Xg[  i, j+1], Yg[  i, j+1]
            xC, yC = Xg[i+1, j+1], Yg[i+1, j+1]
            xD, yD = Xg[i+1,   j], Yg[i+1,   j]
            cell_areas[i, j] = 0.5 * abs((xC - xA) * (yD - yB) - (xD - xB) * (yC - yA))

    return cell_areas
