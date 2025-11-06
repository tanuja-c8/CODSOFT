import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Step 1: Load Dataset
print("🔹 Loading the Iris dataset from data/iris.csv ...")
data_path = "data/iris.csv"
df = pd.read_csv(data_path)

print("✅ Dataset loaded successfully!")
print(df.head())

# Step 2: Encode Target Column if Needed
# Check if species is text (like "setosa", "versicolor", "virginica")
if df['species'].dtype == 'object':
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    print("🔹 Encoded target column (species) successfully.")

# Step 3: Split Features and Target
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
print("🔹 Training Random Forest Classifier...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete!")

# Step 5: Evaluate Model
y_pred = model.predict(X_test)
print("\n📊 Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 6: Save Model
os.makedirs("models", exist_ok=True)
with open("models/iris_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("💾 Model saved to models/iris_model.pkl")

# Step 7: Save Predictions
os.makedirs("results", exist_ok=True)
pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
pred_df.to_csv("results/predictions.csv", index=False)
print("📁 Predictions saved to results/predictions.csv")

print("\n🎉 Iris flower classification completed successfully!")
