"""Train, cross-validate, benchmark, and persist student performance ML models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.preprocess import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    load_dataset,
)

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def get_candidate_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Return dictionary of candidate regression models for benchmarking."""
    models: dict[str, BaseEstimator] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.05, random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=5, random_state=random_state
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=random_state,
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    return models


def _aggregate_feature_importance(pipeline: Pipeline) -> dict[str, float]:
    """Aggregate transformed feature importances/coefficients back to original 19 columns."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        raw_scores = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        raw_scores = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return {}

    grouped: dict[str, float] = {feat: 0.0 for feat in ALL_FEATURES}
    for name, score in zip(transformed_names, raw_scores):
        if name.startswith("num__"):
            orig = name.removeprefix("num__")
            if orig in grouped:
                grouped[orig] += float(score)
        elif name.startswith("cat__"):
            rest = name.removeprefix("cat__")
            for cat_col in CATEGORICAL_FEATURES:
                if rest.startswith(f"{cat_col}_"):
                    grouped[cat_col] += float(score)
                    break

    total = sum(grouped.values())
    if total > 0:
        grouped = {k: round((v / total) * 100.0, 2) for k, v in grouped.items()}
    return dict(sorted(grouped.items(), key=lambda item: item[1], reverse=True))


def _build_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics and correlations for the dashboard."""
    missing_counts = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if int(df[col].isna().sum()) > 0
    }
    numeric_cols = NUMERIC_FEATURES + [TARGET_COLUMN]
    corr_matrix = df[numeric_cols].corr(numeric_only=True).round(3)
    target_corrs = (
        corr_matrix[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN])
        .sort_values(ascending=False)
        .to_dict()
    )
    score_series = df[TARGET_COLUMN]
    return {
        "row_count": int(len(df)),
        "feature_count": len(ALL_FEATURES),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "missing_counts": missing_counts,
        "target_stats": {
            "min": float(score_series.min()),
            "max": float(score_series.max()),
            "mean": round(float(score_series.mean()), 2),
            "median": round(float(score_series.median()), 2),
            "std": round(float(score_series.std()), 2),
            "q25": round(float(score_series.quantile(0.25)), 2),
            "q75": round(float(score_series.quantile(0.75)), 2),
        },
        "target_correlations": {k: float(v) for k, v in target_corrs.items()},
        "correlation_matrix": corr_matrix.to_dict(),
    }


def train_and_evaluate(
    csv_path: Path | None = None,
    models_dir: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run full training, CV, test evaluation, and artifact serialization."""
    project_root = Path(__file__).resolve().parent.parent
    out_dir = models_dir or (project_root / "models")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path)
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    candidates = get_candidate_models(random_state=random_state)
    fitted_pipelines: dict[str, Pipeline] = {}
    leaderboard: list[dict[str, Any]] = []
    predictions_by_model: dict[str, list[float]] = {}
    feature_importance_by_model: dict[str, dict[str, float]] = {}

    for model_name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
            n_jobs=-1,
        )

        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - start_time

        y_pred = pipeline.predict(X_test)
        test_r2 = float(r2_score(y_test, y_pred))
        test_rmse = float(root_mean_squared_error(y_test, y_pred))
        test_mae = float(mean_absolute_error(y_test, y_pred))

        fitted_pipelines[model_name] = pipeline
        predictions_by_model[model_name] = [round(float(v), 3) for v in y_pred]
        feature_importance_by_model[model_name] = _aggregate_feature_importance(pipeline)

        leaderboard.append(
            {
                "model": model_name,
                "test_r2": round(test_r2, 4),
                "test_rmse": round(test_rmse, 4),
                "test_mae": round(test_mae, 4),
                "cv_r2_mean": round(float(cv_results["test_r2"].mean()), 4),
                "cv_r2_std": round(float(cv_results["test_r2"].std()), 4),
                "cv_rmse_mean": round(
                    float(-cv_results["test_neg_root_mean_squared_error"].mean()), 4
                ),
                "cv_mae_mean": round(
                    float(-cv_results["test_neg_mean_absolute_error"].mean()), 4
                ),
                "fit_time_sec": round(fit_time, 3),
            }
        )

    leaderboard.sort(key=lambda row: row["test_rmse"])
    best_model_name = leaderboard[0]["model"]
    best_pipeline = fitted_pipelines[best_model_name]

    joblib.dump(best_pipeline, out_dir / "best_model.joblib")
    joblib.dump(fitted_pipelines, out_dir / "all_models.joblib")

    report: dict[str, Any] = {
        "best_model": best_model_name,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "leaderboard": leaderboard,
        "feature_importance": feature_importance_by_model,
        "test_actual": [round(float(v), 3) for v in y_test.tolist()],
        "test_predictions": predictions_by_model,
        "dataset_summary": _build_dataset_summary(df),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


if __name__ == "__main__":
    summary = train_and_evaluate()
    print(f"Best model: {summary['best_model']}")
    for row in summary["leaderboard"]:
        print(
            f"  {row['model']:20s} | R2={row['test_r2']:.4f} | "
            f"RMSE={row['test_rmse']:.4f} | MAE={row['test_mae']:.4f} | "
            f"CV_R2={row['cv_r2_mean']:.4f}±{row['cv_r2_std']:.4f}"
        )
