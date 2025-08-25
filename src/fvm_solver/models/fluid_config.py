from dataclasses import dataclass


@dataclass
class FluidConfiguration:
    M_inf: float = 0.7
    gamma: float = 1.4
    R:     float = 287.05       # Specific gas constant for air
    p_atm: float = 101300.0     # Atmoshpheric pressure (Pascal)
    T_atm: float = 288.0        # Atmospheric temperature (Kelvin)
    
    @property
    def rho_atm(self) -> float:
        return self.p_atm / (self.R * self.T_atm)  # computed from constants by ideal gas law
    