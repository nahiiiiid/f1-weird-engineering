
from typing import List, Tuple

# Each window: (start_lap, end_lap, type) where type in {"SC","VSC"}
# Multipliers below are intentionally conservative defaults.
SC_LAPTIME_MULT = 0.82
VSC_LAPTIME_MULT = 0.90
SC_PITDELTA_MULT = 0.45
VSC_PITDELTA_MULT = 0.60

def laptime_multiplier(lap: int, windows: List[Tuple[int,int,str]]) -> float:
    for s,e,t in windows or []:
        if s <= lap <= e:
            if t == "SC":
                return SC_LAPTIME_MULT
            if t == "VSC":
                return VSC_LAPTIME_MULT
    return 1.0

def pitdelta_multiplier(lap: int, windows: List[Tuple[int,int,str]]) -> float:
    for s,e,t in windows or []:
        if s <= lap <= e:
            if t == "SC":
                return SC_PITDELTA_MULT
            if t == "VSC":
                return VSC_PITDELTA_MULT
    return 1.0
