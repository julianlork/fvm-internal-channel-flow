from matplotlib import pyplot as plt
from fvm_solver.mesh import Mesh



def plot_grid(mesh: Mesh) -> None:
    # short variables for readability
    Xg, Yg = mesh.vertices_x, mesh.vertices_y
    Nx, Ny = mesh.vertices_x.shape

    fig, ax = plt.subplots(figsize=(15, 10))
    for i in range(Nx): ax.plot(Xg[i, :], Yg[i, :], color='tab:blue', lw=0.4)  # Vertical lines (streamwise)
    for j in range(Ny): ax.plot(Xg[:, j], Yg[:, j], color='tab:blue', lw=0.4)  # Horizontal lines (wall-normal)
    ax.set_aspect('equal')