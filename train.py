import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")

# Load dataset
df = pd.read_csv(
    "C:/Users/Priyanshu Kumar/Desktop/Heart-Disease-Prediction-DevOps/data/heart.csv"
)

print(df.head())

# Remove ID column
df = df.drop("id", axis=1)

# Convert ALL categorical columns
for col in df.columns:
    if df[col].dtype == "object" or str(df[col].dtype) == "string" or str(df[col].dtype) == "str":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Convert True/False if present
df = df.replace({True: 1, False: 0})

# Fill missing values
df = df.fillna(df.mean())

print("\nData types after conversion:")
print(df.dtypes)

# Features and target
X = df.drop("num", axis=1)
y = df["num"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
print("Training model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

# Save model
joblib.dump(model, "model/heart_model.pkl")

print("Model saved successfully!")