"""Interactive Streamlit interface for the student placement predictor."""

from pathlib import Path
import json

import streamlit as st

from placement_predictor.model import load_model, predict_placement, train_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "placement_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
DATA_PATH = ROOT / "data" / "student_placement.csv"

st.set_page_config(
    page_title="Student placement predictor",
    page_icon=":material/school:",
    layout="wide",
)


@st.cache_resource
def get_model(model_path: str):
    return load_model(Path(model_path))


@st.cache_data
def get_metrics(metrics_path: str) -> dict:
    return json.loads(Path(metrics_path).read_text(encoding="utf-8"))


st.badge("Machine Learning • Deep Learning • Innovation", icon=":material/auto_awesome:", color="orange")
st.title("Student placement predictor", icon=":material/school:")
st.markdown(
    "Explore how your academic profile, skills, and experience shape a "
    "placement estimate from a machine-learning model."
)
st.caption("A simple, private-to-this-session way to explore your profile. No sign-in required.")
st.warning(
    "This demo uses synthetic data, not real student outcomes. Treat the estimate as "
    "educational only—not as a guarantee or a basis for hiring, admissions, or other decisions.",
    icon=":material/info:",
)

if not MODEL_PATH.exists():
    with st.spinner("Preparing the demonstration model..."):
        train_model(DATA_PATH, MODEL_PATH, METRICS_PATH)

model = get_model(str(MODEL_PATH))

profile_column, result_column = st.columns([1.2, 0.8], gap="large")

with profile_column:
    with st.container(border=True):
        st.subheader("Build your profile", icon=":material/person_edit:")
        st.caption("Adjust the fields, then select **Estimate my outlook**.")

        with st.form("student_profile", border=False):
            st.markdown("**Academic background**")
            academic_left, academic_right = st.columns(2)
            with academic_left:
                cgpa = st.slider(
                    "CGPA (out of 10)",
                    min_value=4.0,
                    max_value=10.0,
                    value=7.2,
                    step=0.1,
                    help="Your current cumulative grade point average.",
                )
                internships = st.number_input(
                    "Internships",
                    min_value=0,
                    max_value=3,
                    value=1,
                    help="Number of internships completed.",
                )
            with academic_right:
                backlogs = st.number_input(
                    "Backlogs",
                    min_value=0,
                    max_value=3,
                    value=0,
                    help="Number of current or past backlogs.",
                )
                projects = st.number_input(
                    "Projects",
                    min_value=0,
                    max_value=6,
                    value=2,
                    help="Number of projects you have completed.",
                )

            st.markdown("**Assessment scores**")
            coding_score = st.slider("Coding", 0, 100, 65, help="Coding assessment score.")
            communication_score = st.slider(
                "Communication", 0, 100, 68, help="Communication assessment score."
            )
            aptitude_score = st.slider(
                "Aptitude", 0, 100, 62, help="Aptitude assessment score."
            )

            submitted = st.form_submit_button(
                "Estimate my outlook",
                type="primary",
                icon=":material/auto_awesome:",
                width="stretch",
            )

if submitted:
    student = {
        "cgpa": cgpa,
        "internships": internships,
        "projects": projects,
        "coding_score": coding_score,
        "communication_score": communication_score,
        "aptitude_score": aptitude_score,
        "backlogs": backlogs,
    }
    prediction, probability = predict_placement(model, student)
    st.session_state.placement_result = {
        "prediction": prediction,
        "probability": probability,
        "student": student,
    }

with result_column:
    with st.container(border=True):
        st.subheader("Your outlook", icon=":material/insights:")
        result = st.session_state.get("placement_result")

        if result is None:
            st.info(
                "Your estimate will appear here. Complete your profile and select "
                "**Estimate my outlook** to get started.",
                icon=":material/arrow_back:",
            )
            st.metric("Placement probability", "—", border=True)
            st.progress(0.0, text="Waiting for your profile")
        else:
            probability = result["probability"]
            prediction = result["prediction"]

            st.metric(
                "Estimated placement probability",
                f"{probability:.0%}",
                border=True,
                help="The model's estimated probability for the placed class; not a real-world likelihood.",
            )
            st.progress(probability, text="Model estimate")

            if prediction:
                st.success(
                    "Good news! Based on the provided information, the student is likely to get placed.",
                    icon=":material/check_circle:",
                )
            else:
                st.info(
                    "Based on the provided information, the student may need additional preparation to improve placement chances.",
                    icon=":material/info:",
                )

            st.caption(
                "This output reflects patterns in generated sample data. "
                "It does not measure a student's ability or future potential."
            )

        with st.expander("How to read this estimate", icon=":material/help:"):
            st.write(
                "The model compares the profile with patterns learned from synthetic "
                "examples. The percentage is a model output, not a promise, a calibrated "
                "real-world probability, or a recommendation."
            )

st.subheader("About this model", icon=":material/analytics:")
st.caption("Evaluation metrics use a held-out split of synthetic demonstration data.")

if METRICS_PATH.exists():
    metrics = get_metrics(str(METRICS_PATH))
    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric(
    "Prediction Accuracy",
    f"{metrics['accuracy']:.1%}",
    border=True
)

metric_middle.metric(
    "Model Reliability",
    f"{metrics['roc_auc']:.3f}",
    border=True
)

metric_right.metric(
    "Training Samples",
    f"{metrics['rows']:,}",
    border=True
)
