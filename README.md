# 🫀 HeartGuard: Heart Disease Prediction System with Automated CI/CD Pipeline

![Heart Disease Prediction](https://img.shields.io/badge/ML-Heart%20Disease%20Prediction-red)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-ff4b4b)

**HeartGuard** is a production-ready, resume-grade Machine Learning web application built to predict the likelihood of heart disease in patients and deliver interactive, real-time predictions through a clean clinical UI.

Instead of exposing a raw model output, HeartGuard provides an intuitive patient input form with sliders and dropdowns, returning confidence-backed predictions powered by a **Random Forest Classifier** trained on real-world cardiac data — fully containerized and backed by an automated CI/CD pipeline.

---

## 🚀 Key Features

### 🔍 Interactive Prediction Hub:
- Patient detail form with sliders, dropdowns, and input fields for clinical features (age, sex, chest pain type, cholesterol, blood pressure, ECG results, and more).
- Real-time **Positive / Negative** heart disease prediction with confidence scores.
- Custom CSS-styled UI with a clean clinical aesthetic.

### ⚙️ Machine Learning Engine:
- **Random Forest Classifier** (100 estimators) trained on the UCI Heart Disease dataset (`heart.csv`).
- Automated **Label Encoding** for categorical features (e.g., sex).
- **Missing value imputation** using column means for robust inference.
- Model persisted via **Joblib** (`.pkl`) for instant load on startup.

### 📊 Exploratory Data Analysis (EDA):
- Jupyter Notebook (`notebooks/EDA.ipynb`) with full dataset inspection, null-value analysis, feature distribution plots, correlation heatmaps, and summary statistics.

### 🐳 Production Containerization:
- A fully optimized **Docker image** pre-trained and ready for instant startup.
- Serves the Streamlit app on port `8501` on any machine without environment setup.

### 🛡️ CI/CD Integration:
- **GitHub Actions** workflow automatically installs dependencies, runs `train.py`, and verifies project health on every push or pull request.

---

## 🛠️ Technology Stack

| Layer | Tools |
|---|---|
| **Frontend & UI** | Streamlit, Custom CSS |
| **Machine Learning** | Scikit-Learn (RandomForestClassifier), NumPy, Joblib |
| **Data Processing** | Pandas, Scikit-Learn (LabelEncoder, train_test_split) |
| **EDA & Visualization** | Jupyter Notebook, Matplotlib / Seaborn |
| **DevOps & Containerization** | Docker, GitHub Actions (YAML Workflows) |

---

## 📐 Architecture & Workflow

```
Dataset (heart.csv)
       ↓
Data Preprocessing
(Label Encoding → Missing Value Imputation → Boolean Conversion)
       ↓
Exploratory Data Analysis
(EDA.ipynb — distributions, correlations, statistics)
       ↓
Random Forest Classifier
(train_test_split → model.fit → model.predict)
       ↓
Model Persistence
(joblib.dump → model/heart_model.pkl)
       ↓
Streamlit Web Application
(Patient Form → Prediction → Confidence Score)
       ↓
GitHub Repository
       ↓
GitHub Actions CI/CD
(Checkout → Install Deps → Run train.py → Verify)
       ↓
Docker Containerization
(Build Image → Run Container → Serve on :8501)
       ↓
Streamlit Cloud Deployment
```

---

## 📖 Model Details

HeartGuard uses a **Random Forest Classifier** from Scikit-Learn. The prediction for a patient record is determined by majority vote across 100 decision trees:

$$\hat{y} = \text{MajorityVote}\left(\{T_1(x), T_2(x), \ldots, T_{100}(x)\}\right)$$

Where each tree $T_i$ is trained on a bootstrap sample of the dataset with a random feature subset at each split.

**Preprocessing pipeline applied before training and inference:**

| Step | Method |
|---|---|
| Categorical Encoding | `LabelEncoder` on `sex` column |
| Missing Values | `df.fillna(df.mean(numeric_only=True))` |
| Boolean Normalization | `.replace({True: 1, False: 0})` |
| Train/Test Split | `train_test_split()` (default 80/20) |

**Target variable:** `num` — predicts presence of heart disease (binary classification).

**Achieved Accuracy:** `~59%` on the test split.

> ⚠️ This model is intended for educational and demonstration purposes only. It is not a substitute for clinical diagnosis.

---

## 🏃 Local Setup & Installation

### Prerequisites
- Python 3.11+
- Git

### Step-by-Step Installation

**Clone the Repository:**
```bash
git clone https://github.com/yourusername/Heart-Disease-Prediction-DevOps.git
cd Heart-Disease-Prediction-DevOps
```

**Create a Virtual Environment:**
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Run the Training Pipeline:** Train the Random Forest model on `heart.csv`:
```bash
python train.py
```
Output: `Model saved successfully` → `model/heart_model.pkl`

**Run Streamlit:** Launch the web application locally:
```bash
streamlit run app.py
```

Access the app in your browser at **http://localhost:8501**.

---

## 🐳 Docker Deployment

To build and run the containerized application:

**Build the Docker Image:**
```bash
docker build -t heart-app .
```

**Run the Container:**
```bash
docker run -p 8501:8501 heart-app
```

> ⚠️ **Note:** Ensure **Docker Desktop is running** before executing these commands. The `dockerDesktopLinuxEngine not found` error occurs when Docker Desktop is not active.

The application will train the model and serve on port `8501`.

---

## 🛡️ CI/CD Pipeline

Every `git push` or pull request triggers the following **GitHub Actions** workflow (`.github/workflows/ci.yml`):

```yaml
name: CI Pipeline

on:
  push:
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run app test
        run: python train.py
```

**What happens on every push:**
1. GitHub detects the code change.
2. Spins up a fresh Ubuntu virtual machine.
3. Installs all project dependencies.
4. Executes `train.py` to verify end-to-end training works correctly.
5. Reports pipeline success or failure.

---

## ☁️ Streamlit Cloud Deployment

The application is deployed on **Streamlit Community Cloud**:

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account.
3. Select the repository: `Heart-Disease-Prediction-DevOps`
4. Branch: `main` | Main file: `app.py`
5. Click **Deploy**.

🔗 **Live App:** *(Add your Streamlit deployment URL here)*

---

## 📁 Project Structure

```
Heart-Disease-Prediction-DevOps/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
│
├── data/
│   └── heart.csv               # UCI Heart Disease dataset
│
├── model/
│   └── heart_model.pkl         # Trained Random Forest model (auto-generated)
│
├── notebooks/
│   └── EDA.ipynb               # Exploratory Data Analysis notebook
│
├── app.py                      # Streamlit web application
├── train.py                    # Model training script
├── Dockerfile                  # Docker container configuration
├── requirements.txt            # Python dependencies
├── README.md
└── .gitignore
```

---

## 📦 Requirements

```
joblib==1.5.3
numpy==2.4.6
pandas==3.0.3
scikit-learn==1.8.0
streamlit==1.57.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## ✅ Objectives Achieved

| Objective | Status |
|---|---|
| Develop ML model (Random Forest) | ✅ Achieved |
| EDA and data preprocessing | ✅ Achieved |
| Select and evaluate best model | ✅ Achieved |
| Interactive Streamlit UI | ✅ Achieved |
| Docker containerization | ✅ Achieved |
| GitHub Actions CI/CD pipeline | ✅ Achieved |
| Automated testing & deployment | ✅ Achieved |
| Streamlit Cloud deployment | ✅ Achieved |

---

## ⚠️ Disclaimer

This application is built for **educational and portfolio demonstration purposes only**. The predictions made by this model should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.
