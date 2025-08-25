from dataclasses import dataclass, field
from typing import List


@dataclass
class SolverConfiguration:
    iteration_limit:  int         = 20_000
    cfl:              float       = 2.0
    rk4_coefficients: List[float] = field(default_factory=lambda: [1/4, 1/3, 1/2])
