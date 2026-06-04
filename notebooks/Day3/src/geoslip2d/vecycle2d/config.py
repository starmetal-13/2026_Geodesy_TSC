"""Configuration objects for the VECycle2D Python port.

Step 1 of the port only covers configuration and geometry construction.
The numerical Green's-function kernels are intentionally not wired in yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class GeometryConfig:
    """User-facing 2D wedge/interface geometry parameters.

    Distances are in km, following the MATLAB implementation.
    """

    H_elastic_right: float = 30.0
    H_elastic_left: float = 40.0
    L_slab: float = 500.0
    W: float = 200.0
    faultdip_trench: float = 5.0
    faultdip_bottom: float = 35.0
    x_trench: float = 0.0
    x_bottom: float = 300.0
    wedge_bot: float = 80.0
    wedge_top_x: float = 300.0
    pL: float = 8.0


@dataclass(slots=True)
class ForwardConfig:
    """Forward-cycle parameters used after Green's functions are built."""

    tR_scale: float = 10.0
    teq: float = 500.0
    T: float = 550.0
    Ldepth: float = 40.0
    backslip_rate: float = -1.0


@dataclass(slots=True)
class IOConfig:
    """I/O controls for later port phases."""

    output_file: str | Path = "VECycle2D_greens.mat"
    save_output: bool = True
    save_v73: bool = False


@dataclass(slots=True)
class PlotConfig:
    """Plot controls for later port phases."""

    geometry: bool = True
    forward_cycle: bool = True


@dataclass(slots=True)
class InternalConstants:
    """Hardwired constants retained from the verified MATLAB code.

    These are not intended to be user-facing configuration variables.
    """

    shift: float = 1.0e5
    mu_1: float = 1.0
    mu_2: float = 1.0
    nu: float = 0.49
    rigidity: float = 30.0

    @property
    def rhog(self) -> float:
        return 3.0 * (1.0 * 30.0 / self.rigidity * 1.0e-3)


@dataclass(slots=True)
class Config:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    forward: ForwardConfig = field(default_factory=ForwardConfig)
    io: IOConfig = field(default_factory=IOConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    constants: InternalConstants = field(default_factory=InternalConstants)


def default_config(model: str = "viscoelastic_cycle") -> Config:
    """Return the default VECycle2D configuration.

    Parameters
    ----------
    model
        Currently only ``"viscoelastic_cycle"`` is supported. The argument is
        retained to mirror the MATLAB ``geo2dslip_default_config`` wrapper.
    """

    if model.lower() not in {"viscoelastic_cycle", "vecycle2d", "ve_cycle"}:
        raise ValueError(f"Unsupported model type: {model!r}")
    return Config()
