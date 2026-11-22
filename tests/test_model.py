import pytest

from placement_predictor.model import (
    FEATURES,
    generate_dataset,
    predict_placement,
    train_model,
)


def test_generated_dataset_is_reproducible_and_has_both_classes():
    first = generate_dataset(rows=120)
    second = generate_dataset(rows=120)

    assert first.equals(second)
    assert set(FEATURES + ["placed"]).issubset(first.columns)
    assert set(first["placed"].unique()) == {0, 1}


def test_training_and_prediction_round_trip(tmp_path):
    data_path = tmp_path / "data" / "students.csv"
    model_path = tmp_path / "artifacts" / "model.joblib"
    metrics_path = tmp_path / "artifacts" / "metrics.json"

    metrics = train_model(data_path, model_path, metrics_path)

    assert data_path.exists()
    assert model_path.exists()
    assert metrics_path.exists()
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_prediction_rejects_missing_features():
    with pytest.raises(ValueError, match="missing features"):
        predict_placement(None, {"cgpa": 8.0})
