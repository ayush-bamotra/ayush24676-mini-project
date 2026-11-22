"""Dataset generation, training, and inference for the placement predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "cgpa",
    "internships",
    "projects",
    "coding_score",
    "communication_score",
    "aptitude_score",
    "backlogs",
]
TARGET = "placed"
RANDOM_STATE = 42


def generate_dataset(rows: int = 800, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Create reproducible, synthetic student records for demonstration only."""
    if rows < 20:
        raise ValueError("rows must be at least 20 to support a train/test split")

    rng = np.random.default_rng(random_state)
    cgpa = np.clip(rng.normal(7.2, 1.0, rows), 4.0, 10.0)
    internships = rng.choice([0, 1, 2, 3], rows, p=[0.35, 0.38, 0.20, 0.07])
    projects = rng.integers(0, 7, rows)
    coding = np.clip(rng.normal(64, 17, rows), 0, 100)
    communication = np.clip(rng.normal(68, 15, rows), 0, 100)
    aptitude = np.clip(rng.normal(62, 18, rows), 0, 100)
    backlogs = rng.choice([0, 1, 2, 3], rows, p=[0.72, 0.17, 0.08, 0.03])

    linear_score = (
        0.75 * (cgpa - 7.0)
        + 0.38 * internships
        + 0.12 * (projects - 2)
        + 0.025 * (coding - 60)
        + 0.012 * (communication - 65)
        + 0.020 * (aptitude - 60)
        - 0.65 * backlogs
        - 0.10
    )
    probability = 1 / (1 + np.exp(-linear_score))
    placed = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "cgpa": np.round(cgpa, 2),
            "internships": internships,
            "projects": projects,
            "coding_score": np.round(coding, 1),
            "communication_score": np.round(communication, 1),
            "aptitude_score": np.round(aptitude, 1),
            "backlogs": backlogs,
            TARGET: placed,
        }
    )


def build_pipeline() -> Pipeline:
    """Return a fresh, fitted-ready classification pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def train_model(
    data_path: Path,
    model_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    """Train, evaluate, and save a model; generate demo data if it is missing."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        generate_dataset().to_csv(data_path, index=False)

    data = pd.read_csv(data_path)
    required = set(FEATURES + [TARGET])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    if data.empty or set(data[TARGET].unique()) != {0, 1}:
        raise ValueError("Dataset target must contain both placement classes encoded as 0 and 1")
    if data[TARGET].value_counts().min() < 2:
        raise ValueError("Dataset must contain at least two rows from each placement class")

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET].astype(int),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data[TARGET],
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "rows": int(len(data)),
        "features": FEATURES,
        "note": "Evaluation uses synthetic demonstration data, not real placement outcomes.",
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(model_path: Path) -> Pipeline:
    """Load a previously trained pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    return joblib.load(model_path)


def predict_placement(model: Pipeline, student: dict[str, float | int]) -> tuple[int, float]:
    """Return predicted class and probability of placement for one student."""
    missing = set(FEATURES).difference(student)
    if missing:
        raise ValueError(f"Student record is missing features: {', '.join(sorted(missing))}")
    row = pd.DataFrame([{feature: student[feature] for feature in FEATURES}])
    probability = float(model.predict_proba(row)[0, 1])
    return int(probability >= 0.5), probability
