"""Inference helpers, cohort comparison, and personalized study recommendations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocess import ALL_FEATURES, classify_risk_tier, clean_dataframe


def load_artifacts(
    models_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load saved best pipeline, all candidate pipelines, and metrics JSON."""
    root = Path(__file__).resolve().parent.parent
    dir_path = models_dir or (root / "models")
    best_model = joblib.load(dir_path / "best_model.joblib")
    all_models = joblib.load(dir_path / "all_models.joblib")
    with open(dir_path / "metrics.json", encoding="utf-8") as fh:
        metrics = json.load(fh)
    return best_model, all_models, metrics


def generate_recommendations(student: dict[str, Any], predicted_score: float) -> list[str]:
    """Generate actionable, domain-grounded recommendations from student inputs."""
    tips: list[str] = []

    attendance = float(student.get("Attendance", 80))
    hours = float(student.get("Hours_Studied", 20))
    sleep = float(student.get("Sleep_Hours", 7))
    prev = float(student.get("Previous_Scores", 75))
    tutoring = float(student.get("Tutoring_Sessions", 1))
    physical = float(student.get("Physical_Activity", 3))

    if attendance < 80:
        tips.append(
            f"Attendance is {attendance:.0f}%. Raising class attendance above 85% is one of the strongest predictors of higher exam scores."
        )
    if hours < 18:
        tips.append(
            f"Weekly study time is {hours:.0f} hrs. Adding 4–6 structured study hours per week can meaningfully lift projected performance."
        )
    if sleep < 6:
        tips.append(
            f"Sleep averages {sleep:.0f} hrs/night. Aiming for 7–8 hours helps memory consolidation and exam recall."
        )
    if prev < 65 and tutoring < 2:
        tips.append(
            "Previous scores indicate foundational gaps; adding 1–2 monthly tutoring or peer-study sessions is recommended."
        )
    if student.get("Access_to_Resources") == "Low" or student.get("Internet_Access") == "No":
        tips.append(
            "Limited study resources detected; leverage school library materials, offline notes, or campus computer labs."
        )
    if student.get("Motivation_Level") == "Low":
        tips.append(
            "Set weekly milestone targets and break study blocks into 45-minute focused sessions to build momentum."
        )
    if physical < 2:
        tips.append(
            "Include 2–3 hours of weekly physical activity to reduce academic stress and improve focus."
        )

    if not tips:
        if predicted_score >= 72:
            tips.append(
                "Current study habits and attendance are well balanced—maintain consistency and practice timed mock papers."
            )
        else:
            tips.append(
                "Focus on targeted practice tests in weaker subjects and review past exam mistakes weekly."
            )

    return tips


def predict_single(
    student: dict[str, Any],
    pipeline: Any,
    cohort_mean: float = 67.24,
) -> dict[str, Any]:
    """Predict Exam_Score and risk tier for a single student record."""
    row_df = pd.DataFrame([{feat: student.get(feat) for feat in ALL_FEATURES}])
    cleaned = clean_dataframe(row_df, require_target=False)
    raw_pred = float(pipeline.predict(cleaned)[0])
    bounded_pred = max(0.0, min(100.0, raw_pred))
    risk = classify_risk_tier(bounded_pred)

    return {
        "predicted_score": round(bounded_pred, 2),
        "raw_prediction": round(raw_pred, 2),
        "delta_vs_cohort": round(bounded_pred - cohort_mean, 2),
        "risk": risk,
        "recommendations": generate_recommendations(student, bounded_pred),
    }


def predict_batch(df: pd.DataFrame, pipeline: Any) -> pd.DataFrame:
    """Run batch predictions on a DataFrame containing the 19 student features."""
    cleaned = clean_dataframe(df, require_target=False)
    preds = pipeline.predict(cleaned[ALL_FEATURES])
    result = df.copy()
    result["Predicted_Exam_Score"] = [round(max(0.0, min(100.0, float(v))), 2) for v in preds]
    result["Risk_Tier"] = [
        classify_risk_tier(score)["tier"] for score in result["Predicted_Exam_Score"]
    ]
    return result
