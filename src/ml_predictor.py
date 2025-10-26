
from dataclasses import dataclass
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

_FEATURES = ["lap", "stint_lap", "pit_count_so_far", "is_sc", "is_vsc"]  # keep it simple

@dataclass
class MLConfig:
    enable: bool = False
    n_estimators: int = 200
    max_depth: Optional[int] = None
    random_state: int = 42

class LapTimeML:
    def __init__(self, cfg: MLConfig):
        self.cfg = cfg
        self.model = None

    def _make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        for col in _FEATURES:
            if col not in X.columns:
                X[col] = 0
        return X[_FEATURES]

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        train_df columns expected:
          - lap (global lap number)
          - stint_lap
          - pit_count_so_far
          - is_sc (0/1)
          - is_vsc (0/1)
          - y: lap_time_seconds
        """
        if not self.cfg.enable:
            return
        X = self._make_features(train_df)
        y = train_df["y"].astype(float).values
        self.model = RandomForestRegressor(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            random_state=self.cfg.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.cfg.enable or self.model is None:
            return df["y"].values if "y" in df.columns else np.zeros(len(df))
        X = self._make_features(df)
        return self.model.predict(X)
