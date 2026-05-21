import streamlit as st
import joblib
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#ff4b4b;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:20px;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.1);
}

.result{
    padding:20px;
    border-radius:10px;
    font-size:24px;
    font-weight:bold;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
model = joblib.load("model/heart_model.pkl")

# ---------------- Header ----------------
st.markdown('<p class="title">❤️ Heart Disease Prediction System</p>',
            unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">AI-powered heart disease prediction using Machine Learning</p>',
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2382/2382461.png",
    width=120
)

st.sidebar.header("About Project")

st.sidebar.info("""
This project predicts the possibility of heart disease using patient medical information.

Model: Random Forest  
Dataset: UCI Heart Disease Dataset
""")

# ---------------- Input Form ----------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🩺 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age",20,100,50)

    trestbps = st.slider(
        "Resting Blood Pressure",
        80,
        200,
        120
    )

    chol = st.slider(
        "Cholesterol",
        100,
        600,
        200
    )

    thalach = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    oldpeak = st.slider(
        "Old Peak",
        0.0,
        10.0,
        1.0
    )


with col2:

    sex = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    sex = 1 if sex=="Male" else 0

    dataset = st.selectbox(
        "Dataset",
        [0,1,2,3]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal",
            "Asymptomatic"
        ]
    )

    cp_map={
        "Typical Angina":0,
        "Atypical Angina":1,
        "Non-anginal":2,
        "Asymptomatic":3
    }

    cp=cp_map[cp]

    fbs=st.selectbox(
        "Fasting Blood Sugar",
        ["No","Yes"]
    )

    fbs=1 if fbs=="Yes" else 0

    restecg=st.selectbox(
        "Rest ECG",
        [0,1,2]
    )

    exang=st.selectbox(
        "Exercise Angina",
        ["No","Yes"]
    )

    exang=1 if exang=="Yes" else 0

    slope=st.selectbox(
        "Slope",
        [0,1,2]
    )

    ca=st.slider(
        "Major Vessels",
        0,
        4,
        0
    )

    thal=st.selectbox(
        "Thal",
        [0,1,2]
    )

st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Prediction ----------------

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    features=np.array([

        age,
        sex,
        dataset,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal

    ]).reshape(1,-1)

    prediction=model.predict(features)

    st.write("")
    st.subheader("Prediction Result")

    if prediction[0]==0:

        st.markdown(
            """
            <div class="result"
            style="background:#d4edda;color:#155724;">
            ✅ Low Risk of Heart Disease
            </div>
            """,
            unsafe_allow_html=True
        )

        st.balloons()

    else:

        st.markdown(
            """
            <div class="result"
            style="background:#f8d7da;color:#721c24;">
            ⚠ High Risk of Heart Disease
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning(
            "Please consult a medical professional."
        )