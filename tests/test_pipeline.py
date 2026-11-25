"""Unit and integration tests for preprocessing, training, and prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.preprocess import (
    ALL_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    classify_risk_tier,
    load_dataset,
    resolve_dataset_path,
)
from src.predict import predict_batch, predict_single
from src.train import train_and_evaluate


def test_dataset_loads_and_validates() -> None:
    path = resolve_dataset_path()
    df = load_dataset(path)
    assert len(df) > 6000
    for col in ALL_FEATURES + [TARGET_COLUMN]:
        assert col in df.columns
    assert df[TARGET_COLUMN].isna().sum() == 0


def test_preprocessor_handles_missing_categoricals() -> None:
    df = load_dataset().head(20).copy()
    df.loc[0, "Teacher_Quality"] = None
    df.loc[1, "Parental_Education_Level"] = None
    df.loc[2, "Distance_from_Home"] = None

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df[ALL_FEATURES])
    assert transformed.shape[0] == 20
    assert not pd.isna(transformed).any()


def test_risk_tiers_cover_score_ranges() -> None:
    assert classify_risk_tier(58.0)["tier"] == "High Risk"
    assert classify_risk_tier(65.5)["tier"] == "Moderate Risk"
    assert classify_risk_tier(69.0)["tier"] == "On Track"
    assert classify_risk_tier(76.0)["tier"] == "High Performer"


def test_train_and_predict_end_to_end(tmp_path: Path) -> None:
    report = train_and_evaluate(models_dir=tmp_path)
    assert (tmp_path / "best_model.joblib").exists()
    assert (tmp_path / "all_models.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert len(report["leaderboard"]) >= 6
    best_row = report["leaderboard"][0]
    assert best_row["test_r2"] > 0.5

    df = load_dataset()
    sample = df[ALL_FEATURES].iloc[0].to_dict()
    import joblib

    pipeline = joblib.load(tmp_path / "best_model.joblib")
    single_out = predict_single(sample, pipeline)
    assert 0.0 <= single_out["predicted_score"] <= 100.0
    assert single_out["risk"]["tier"] in {
        "High Risk",
        "Moderate Risk",
        "On Track",
        "High Performer",
    }
    assert len(single_out["recommendations"]) >= 1

    batch_out = predict_batch(df[ALL_FEATURES].head(5), pipeline)
    assert len(batch_out) == 5
    assert "Predicted_Exam_Score" in batch_out.columns
    assert "Risk_Tier" in batch_out.columns
