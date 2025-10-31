import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


# Step 1: Load Dataset

data_path = "data/train.csv"
df = pd.read_csv(data_path)

print("✅ Data loaded successfully!")
print("Shape:", df.shape)
print(df.head())


# Step 2: Data Preprocessing

# Fill missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Drop irrelevant columns
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Encode categorical columns
label_enc = LabelEncoder()
df["Sex"] = label_enc.fit_transform(df["Sex"])
df["Embarked"] = label_enc.fit_transform(df["Embarked"])


# Step 3: Split Data

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 4: Model Training

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Step 5: Evaluation

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n🎯 Model Evaluation:")
print(f"Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Step 6: Save Model

os.makedirs("models", exist_ok=True)
with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n💾 Model saved successfully: models/random_forest_model.pkl")


# Step 7: Save Predictions

os.makedirs("results", exist_ok=True)
predictions = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
predictions.to_csv("results/predictions.csv", index=False)
print("📁 Predictions saved in results/predictions.csv")

print("\n✅ Titanic Survival Prediction Complete!")
