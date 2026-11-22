"""Train the demo placement model and save its evaluation metrics."""

from pathlib import Path
import json

from placement_predictor.model import train_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    metrics = train_model(
        ROOT / "data" / "student_placement.csv",
        ROOT / "artifacts" / "placement_model.joblib",
        ROOT / "artifacts" / "metrics.json",
    )
    print(json.dumps({key: metrics[key] for key in ("accuracy", "roc_auc", "rows")}, indent=2))


if __name__ == "__main__":
    main()
