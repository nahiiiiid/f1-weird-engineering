
from dataclasses import dataclass

@dataclass
class TrafficConfig:
    enable: bool = True
    penalty_per_lap_sec: float = 0.25  # extra time due to dirty air/traffic
    max_laps_after_rejoin: int = 5     # how many laps penalty lasts when rejoin is worse than baseline
    trigger_places_delta: int = 1      # apply penalty if simulated rank at rejoin is worse by >= this many places

def compute_traffic_penalty(rejoin_rank_delta: int, lap_in_window: int, cfg: TrafficConfig) -> float:
    """
    Returns additive seconds for current lap due to traffic, given how much worse the rejoin rank is.
    Simple model: fixed penalty per lap for the first K laps after a pit if rank worsened.
    """
    if not cfg.enable:
        return 0.0
    if rejoin_rank_delta >= cfg.trigger_places_delta and lap_in_window < cfg.max_laps_after_rejoin:
        # Could scale by rejoin_rank_delta if desired
        return cfg.penalty_per_lap_sec
    return 0.0
