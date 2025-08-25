from dataclasses import dataclass, field
from typing import Callable, Union
import numpy as np


ArrayLike = Union[float, np.ndarray]


def BOTTOM_BUMP(eps: float) -> Callable[[ArrayLike], ArrayLike]:
    def f(x: ArrayLike) -> ArrayLike:
        xa = np.asarray(x)
        core = eps * (xa - 1.0) * (1.0 - (xa - 1.0) / 1.0)
        mask = (xa >= 1.0) & (xa <= 2.0)
        out = np.where(mask, core, 0.0)
        return out if isinstance(x, np.ndarray) else float(out)
    return f


def CONST_TOP(value: float = 1.0) -> Callable[[ArrayLike], ArrayLike]:
    def f(x: ArrayLike) -> ArrayLike:
        xa = np.asarray(x)
        out = np.full_like(xa, value, dtype=float)
        return out if isinstance(x, np.ndarray) else float(out)
    return f


@dataclass
class MeshConfig:
    num_grid_lines_x: int = 128
    num_grid_lines_y: int = 64
    domain_size_x: float = 3.0
    domain_size_y: float = 1.0
    bottom_shape: Callable[[ArrayLike], ArrayLike] = field(default_factory=lambda: BOTTOM_BUMP(0.4))
    top_shape: Callable[[ArrayLike], ArrayLike] = field(default_factory=lambda: CONST_TOP(1.0))


    @property
    def num_cells_x(self) -> int:
        return self.num_grid_lines_x - 1

    @property
    def num_cells_y(self) -> int:
        return self.num_grid_lines_y - 1

