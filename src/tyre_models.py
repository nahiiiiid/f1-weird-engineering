
from dataclasses import dataclass
from typing import Dict

@dataclass
class TyreParams:
    base: float  # seconds added baseline (can be 0.0)
    a: float     # linear coefficient per lap in stint
    b: float     # quadratic coefficient per lap^2 in stint

DEFAULT_TYRE_PARAMS: Dict[str, TyreParams] = {
    "Soft":   TyreParams(base=0.00, a=0.035, b=0.0006),
    "Medium": TyreParams(base=0.05, a=0.030, b=0.0005),
    "Hard":   TyreParams(base=0.10, a=0.025, b=0.0004),
    "Inter":  TyreParams(base=1.20, a=0.060, b=0.0010),
    "Wet":    TyreParams(base=2.00, a=0.070, b=0.0012),
}

def stint_adjustment_seconds(stint_lap: int, compound: str, params: Dict[str, TyreParams] = None) -> float:
    """
    Computes additive lap-time adjustment (in seconds) at a given lap number within a stint.
    """
    if params is None:
        params = DEFAULT_TYRE_PARAMS
    comp = params.get(compound, params["Medium"])
    n = max(0, int(stint_lap))
    return comp.base + comp.a * n + comp.b * (n ** 2)
