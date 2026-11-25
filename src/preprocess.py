"""Dataset loading, schema validation, preprocessing pipeline, and risk tier mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Exam_Score"

NUMERIC_FEATURES: list[str] = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]

CATEGORICAL_FEATURES: list[str] = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["No", "Yes"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Internet_Access": ["No", "Yes"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "School_Type": ["Public", "Private"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Learning_Disabilities": ["No", "Yes"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
    "Gender": ["Female", "Male"],
}

NUMERIC_BOUNDS: dict[str, tuple[int, int, int]] = {
    # (min, max, default)
    "Hours_Studied": (1, 44, 20),
    "Attendance": (60, 100, 80),
    "Sleep_Hours": (4, 10, 7),
    "Previous_Scores": (50, 100, 75),
    "Tutoring_Sessions": (0, 8, 1),
    "Physical_Activity": (0, 6, 3),
}


def resolve_dataset_path(project_root: Path | None = None) -> Path:
    """Locate StudentPerformanceFactors.csv in data/ or the project root."""
    root = project_root or Path(__file__).resolve().parent.parent
    candidates = [
        root / "data" / "StudentPerformanceFactors.csv",
        root / "StudentPerformanceFactors.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find StudentPerformanceFactors.csv in 'data/' or project root."
    )


def validate_feature_columns(df: pd.DataFrame, require_target: bool = False) -> None:
    """Ensure required feature columns (and optionally target) exist."""
    expected = set(ALL_FEATURES)
    if require_target:
        expected.add(TARGET_COLUMN)
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def clean_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Strip string columns, coerce numeric features, and validate schema."""
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    validate_feature_columns(cleaned, require_target=require_target)

    for col in CATEGORICAL_FEATURES:
        cleaned[col] = cleaned[col].where(cleaned[col].notna(), None)
        cleaned[col] = cleaned[col].apply(
            lambda val: val.strip() if isinstance(val, str) and val.strip() else None
        )

    for col in NUMERIC_FEATURES:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if require_target:
        cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    return cleaned


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    """Load and clean the Kaggle Student Performance Factors dataset."""
    path = csv_path or resolve_dataset_path()
    df = pd.read_csv(path)
    return clean_dataframe(df, require_target=True)


def build_preprocessor() -> ColumnTransformer:
    """Build leakage-free preprocessing pipeline for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def classify_risk_tier(score: float) -> dict[str, Any]:
    """Map predicted Exam_Score to an actionable academic risk tier."""
    if score < 64.0:
        return {
            "tier": "High Risk",
            "color": "#ef4444",
            "summary": "Immediate academic support recommended.",
        }
    if score < 67.5:
        return {
            "tier": "Moderate Risk",
            "color": "#f59e0b",
            "summary": "Borderline performance; targeted study improvements advised.",
        }
    if score < 72.0:
        return {
            "tier": "On Track",
            "color": "#3b82f6",
            "summary": "Solid expected performance with room to reach top quartile.",
        }
    return {
        "tier": "High Performer",
        "color": "#10b981",
        "summary": "Strong academic standing projected.",
    }
