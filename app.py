"""
Smart Recruitment Assistant - Streamlit Deployment App
Matches ITI_Project.ipynb pipeline exactly:
  - preprocessor.pkl (ColumnTransformer: median-impute+scale numeric,
    most_frequent-impute+OneHotEncode(drop='first') categorical)
  - 4 separate raw classifiers trained on the preprocessed array:
    logistic_model.pkl, random_forest_model.pkl,
    decision_tree_model.pkl, knn_model.pkl

Run locally with:  streamlit run app.py
Deploy free at:     https://share.streamlit.io
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Smart Recruitment Assistant", page_icon="🧑‍💼", layout="centered")

# ---------------------------------------------------------------------------
# REQUIRED FILES
# ---------------------------------------------------------------------------
PREPROCESSOR_FILE = "preprocessor.pkl"
CITY_MAPPING_FILE = "city_mapping.csv"   # optional but recommended

MODEL_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "KNN": "knn_model.pkl",
}

# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_preprocessor():
    if os.path.exists(PREPROCESSOR_FILE):
        return joblib.load(PREPROCESSOR_FILE)
    return None

@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_data
def load_city_mapping():
    if os.path.exists(CITY_MAPPING_FILE):
        return pd.read_csv(CITY_MAPPING_FILE)
    return None

preprocessor = load_preprocessor()
models = load_models()
city_mapping = load_city_mapping()

st.title("🧑‍💼 Smart Recruitment Assistant")
st.write("Predict whether a candidate is likely to be looking for a job change, using your choice of model.")

# ---------------------------------------------------------------------------
# FILE CHECKS
# ---------------------------------------------------------------------------
if preprocessor is None:
    st.error(
        "`preprocessor.pkl` not found. This file is required — it turns raw form "
        "inputs into the exact feature format the models were trained on. "
        "Save it in your notebook with `joblib.dump(preprocessor, 'preprocessor.pkl')` "
        "and add it to this app's folder."
    )
    st.stop()

if not models:
    st.error("No model files found. Add at least one of: " + ", ".join(MODEL_FILES.values()))
    st.stop()

missing_models = [name for name in MODEL_FILES if name not in models]
if missing_models:
    st.info(f"Not available yet (file missing): {', '.join(missing_models)}")

# ---------------------------------------------------------------------------
# MODEL SELECTOR
# ---------------------------------------------------------------------------
model_choice = st.selectbox("Choose a model", list(models.keys()))
selected_model = models[model_choice]

st.divider()
st.subheader("Candidate Information")

# ---------------------------------------------------------------------------
# CITY -> city_development_index
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if city_mapping is not None:
        city_options = sorted(city_mapping["city"].dropna().unique().tolist())
        selected_city = st.selectbox("City", city_options)
        city_development_index = float(
            city_mapping.loc[city_mapping["city"] == selected_city, "city_development_index"].iloc[0]
        )
        st.caption(f"City Development Index: {city_development_index:.3f}")
    else:
        selected_city = st.text_input("City (as it appeared in training data, e.g. 'city_103')", value="city_103")
        city_development_index = st.slider("City Development Index", 0.0, 1.0, 0.75, 0.01)

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    relevent_experience = st.selectbox(
        "Relevant Experience",
        ["Has relevent experience", "No relevent experience"]
    )
    enrolled_university = st.selectbox(
        "Enrolled University",
        ["no_enrollment", "Full time course", "Part time course"]
    )

with col2:
    education_level = st.selectbox(
        "Education Level",
        ["Primary School", "High School", "Graduate", "Masters", "Phd"]
    )
    major_discipline = st.selectbox(
        "Major Discipline",
        ["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"]
    )
    company_size = st.selectbox(
        "Company Size",
        ["<10", "10/49", "50-99", "100-500", "500-999",
         "1000-4999", "5000-9999", "10000+"]
    )
    company_type = st.selectbox(
        "Company Type",
        ["Pvt Ltd", "Funded Startup", "Early Stage Startup",
         "Public Sector", "NGO", "Other"]
    )

st.divider()
col3, col4 = st.columns(2)
with col3:
    experience_raw = st.slider("Years of Experience", 0, 21, 5, help="21 represents '>20 years'")
    training_hours = st.number_input("Training Hours Completed", min_value=0, max_value=400, value=50)
with col4:
    last_new_job_raw = st.selectbox(
        "Years Since Last Job Change",
        [("Never", 0), ("Less than 1 year", 0.5), ("1 year", 1), ("2 years", 2),
         ("3 years", 3), ("4 years", 4), ("More than 4 years", 5)],
        format_func=lambda x: x[0]
    )[1]

# ---------------------------------------------------------------------------
# BUILD RAW INPUT ROW - matches notebook feature engineering exactly
# ---------------------------------------------------------------------------
training_hours_per_experience = training_hours / (experience_raw + 1)

input_df = pd.DataFrame([{
    "city": selected_city,
    "city_development_index": city_development_index,
    "gender": gender,
    "relevent_experience": relevent_experience,
    "enrolled_university": enrolled_university,
    "education_level": education_level,
    "major_discipline": major_discipline,
    "experience": experience_raw,
    "company_size": company_size,
    "company_type": company_type,
    "last_new_job": last_new_job_raw,
    "training_hours": training_hours,
    "training_hours_per_experience": training_hours_per_experience,
}])

st.divider()

if st.button("🔍 Predict", type="primary"):
    try:
        input_processed = preprocessor.transform(input_df)
        prediction = selected_model.predict(input_processed)[0]
        try:
            probability = selected_model.predict_proba(input_processed)[0][1]
        except AttributeError:
            probability = None

        if prediction == 1:
            st.error("⚠️ Prediction: Candidate is LIKELY looking for a job change")
        else:
            st.success("✅ Prediction: Candidate is UNLIKELY to be looking for a job change")

        if probability is not None:
            st.metric("Probability of looking for a change", f"{probability:.1%}")
            st.progress(float(probability))

        st.caption(f"Prediction made using: **{model_choice}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.caption(
            "This usually means the input categories don't match what the "
            "preprocessor was fitted on, or an unseen 'city' value was used "
            "without city_mapping.csv."
        )

# ---------------------------------------------------------------------------
# MODEL COMPARISON (loads all_models_comparison.csv from the notebook if present)
# ---------------------------------------------------------------------------
with st.expander("📊 Model Performance Comparison"):
    if os.path.exists("all_models_comparison.csv"):
        comp_df = pd.read_csv("all_models_comparison.csv", index_col=0)
        st.dataframe(comp_df, use_container_width=True)
        st.bar_chart(comp_df)
    else:
        st.info(
            "Add `all_models_comparison.csv` (saved at the end of your notebook) "
            "to this folder to show the full metrics table here."
        )

# ---------------------------------------------------------------------------
# TOP 10 CANDIDATE RANKING (bonus feature from the notebook)
# ---------------------------------------------------------------------------
with st.expander("🏆 Top 10 Candidate Ranking"):
    if os.path.exists("top10_candidate_ranking.csv"):
        top10_df = pd.read_csv("top10_candidate_ranking.csv", index_col=0)
        st.dataframe(top10_df, use_container_width=True)
    else:
        st.info(
            "Add `top10_candidate_ranking.csv` (saved at the end of your notebook) "
            "to this folder to show the ranked candidates here."
        )