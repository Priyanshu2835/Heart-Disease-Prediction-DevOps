import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("Loading dataset...")

# Load data
df = pd.read_csv("data/heart.csv")

print(df.head())

# Remove ID if exists
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

# Convert bool → int
df = df.replace({True: 1, False: 0})

# Fill missing values safely
for col in df.columns:

    # Check for text columns
    if (
        df[col].dtype == "object"
        or str(df[col].dtype).startswith("string")
    ):

        # fill with most frequent value
        df[col] = df[col].fillna(df[col].mode()[0])

    else:
        # numeric column
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean())

# Encode categorical columns
label_encoders = {}

for col in df.columns:

    if (
        df[col].dtype == "object"
        or str(df[col].dtype).startswith("string")
    ):

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

        label_encoders[col] = le

print("\nData types after conversion:")
print(df.dtypes)

# Features & target
X = df.drop("num", axis=1)
y = df["num"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)

# Create model folder if absent
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/heart_model.pkl")

print("Model saved successfully!")