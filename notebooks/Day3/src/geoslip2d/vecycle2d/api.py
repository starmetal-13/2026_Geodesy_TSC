"""Small command-line entry points for VECycle2D."""

from __future__ import annotations

from .config import default_config
from .build_greens import build_greens
from .forward import ForwardConfig, forward_cycle


def run_default_forward():
    """Build default Green's functions and compute default forward velocities."""
    cfg = default_config()
    greens = build_greens(cfg, keep_internals=False, progress=True)
    fwd = ForwardConfig()
    return greens, forward_cycle(greens, fwd)
