from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split


FEATURES = ["distance", "congestion", "cart_load", "picker_speed"]
TARGET = "travel_seconds"


@dataclass
class TrainingReport:
    model_path: str
    r2: float
    mae: float
    rmse: float
    evaluation_method: str
    cv_r2_mean: float
    cv_mae_mean: float
    cv_rmse_mean: float


class TravelTimeEstimator:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None
        self.evaluation: Optional[Dict[str, float | str]] = None
        if os.path.exists(model_path):
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict) and "model" in loaded:
                self.model = loaded["model"]
                eval_payload = loaded.get("evaluation")
                if isinstance(eval_payload, dict):
                    self.evaluation = eval_payload
            else:
                # Backward compatibility for older artifacts that stored only the model object.
                self.model = loaded

    def _to_matrix(self, samples: Iterable[Dict[str, float]]) -> np.ndarray:
        rows = []
        for s in samples:
            rows.append([float(s.get(f, 0.0)) for f in FEATURES])
        return np.array(rows, dtype=np.float64)

    def train(self, samples: List[Dict[str, float]]) -> TrainingReport:
        if len(samples) < 20:
            raise ValueError("At least 20 samples are required to train the travel time model.")
        x = self._to_matrix(samples)
        y = np.array([float(s[TARGET]) for s in samples], dtype=np.float64)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        r2 = float(r2_score(y_test, pred))
        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_validate(
            model,
            x,
            y,
            cv=cv,
            scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
            n_jobs=-1,
        )
        cv_r2_mean = float(np.mean(cv_scores["test_r2"]))
        cv_mae_mean = float(-np.mean(cv_scores["test_neg_mean_absolute_error"]))
        cv_rmse_mean = float(-np.mean(cv_scores["test_neg_root_mean_squared_error"]))
        evaluation_method = "5-fold cross-validation (R2, MAE, RMSE) + holdout split"
        self.evaluation = {
            "evaluation_method": evaluation_method,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "cv_r2_mean": cv_r2_mean,
            "cv_mae_mean": cv_mae_mean,
            "cv_rmse_mean": cv_rmse_mean,
        }

        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump({"model": model, "evaluation": self.evaluation}, self.model_path)
        self.model = model
        return TrainingReport(
            model_path=self.model_path,
            r2=r2,
            mae=mae,
            rmse=rmse,
            evaluation_method=evaluation_method,
            cv_r2_mean=cv_r2_mean,
            cv_mae_mean=cv_mae_mean,
            cv_rmse_mean=cv_rmse_mean,
        )

    def _deterministic_baseline(
        self,
        distance: float,
        congestion: float | None,
        picker_speed: float,
        *,
        turn_penalty_seconds: float = 0.0,
        stop_count: int = 0,
    ) -> float:
        # Keep the baseline honest: distance / speed is primary, and only
        # explicitly supplied runtime factors should perturb it.
        base_seconds = max(distance / max(picker_speed, 0.1), 0.0)
        congestion_multiplier = 1.0 + max(congestion, 0.0) if congestion is not None else 1.0
        stop_penalty = max(stop_count, 0) * 1.5
        return max((base_seconds * congestion_multiplier) + max(turn_penalty_seconds, 0.0) + stop_penalty, 0.0)

    def predict_seconds_hybrid(
        self,
        distance: float,
        congestion: float | None,
        cart_load: float,
        picker_speed: float,
        *,
        turn_penalty_seconds: float = 0.0,
        stop_count: int = 0,
    ) -> float:
        baseline = self._deterministic_baseline(
            distance,
            congestion,
            picker_speed,
            turn_penalty_seconds=turn_penalty_seconds,
            stop_count=stop_count,
        )
        if self.model is None:
            return baseline

        # ML is intentionally best-effort. Without live telemetry, the model
        # should act as a bounded correction, not a wholesale replacement for
        # the deterministic route-time estimate.
        effective_congestion = max(congestion, 0.0) if congestion is not None else 0.0
        x = np.array([[distance, effective_congestion, cart_load, picker_speed]], dtype=np.float64)
        model_seconds = float(self.model.predict(x)[0])
        max_adjustment = max(baseline * 0.35, 5.0)
        correction = max(min(model_seconds - baseline, max_adjustment), -max_adjustment)
        correction_weight = 0.35 if congestion is not None else 0.2
        return max(baseline + (correction_weight * correction), 0.0)

    def predict_seconds(self, distance: float, congestion: float | None, cart_load: float, picker_speed: float) -> float:
        return self.predict_seconds_hybrid(
            distance=distance,
            congestion=congestion,
            cart_load=cart_load,
            picker_speed=picker_speed,
        )

    def get_evaluation(self) -> Optional[Dict[str, float | str]]:
        return self.evaluation
