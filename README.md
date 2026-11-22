# Student Placement Predictor

A small, end-to-end machine-learning project that trains a binary classifier and
provides an interactive Streamlit interface for experimenting with student profiles.

> **Important:** The included data is generated synthetically for demonstration.
> It does not represent actual students or placement outcomes. Predictions are
> educational estimates only and must not be used to make hiring, admissions,
> financial, or other consequential decisions.

## Features

- Reproducible synthetic dataset with 800 example records.
- Scikit-learn preprocessing and logistic-regression pipeline.
- Stratified train/test evaluation with accuracy, ROC-AUC, and a classification report.
- Guided Streamlit profile form with an interactive estimate and model evaluation cards.
- Coordinated light and dark themes with accessible, native Streamlit controls.
- Focused unit tests for data generation, training, and input validation.

## Project layout

```text
ML_Project/
├── app.py
├── .streamlit/
│   └── config.toml                 # Light/dark themes and visual defaults
├── data/
│   └── student_placement.csv       # generated on first training run if absent
├── artifacts/                      # trained model and metrics; generated locally
├── placement_predictor/
│   └── model.py
├── scripts/
│   └── train_model.py
├── tests/
│   └── test_model.py
└── requirements.txt
```

## Setup

Use Python 3.10 or newer. From this directory, create and activate a virtual
environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and run

Train the model and create the demo dataset (the dataset is created if missing):

```powershell
python -m scripts.train_model
```

Launch the web application:

```powershell
python -m streamlit run .\app.py
```

The app also trains the model automatically on first launch if the model artifact
does not exist. Remove `artifacts\placement_model.joblib` to explicitly retrain;
the model artifact and evaluation metrics are local generated files.

## Test

```powershell
python -m pytest
```

## Input fields

CGPA, internship count, project count, coding, communication and aptitude
assessment scores, and backlog count. These features are used only to illustrate
a simple supervised-learning workflow; the model is not validated for real
placement prediction.
