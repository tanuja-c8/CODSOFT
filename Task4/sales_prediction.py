import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# STEP 1: Load Dataset
# Update the file name based on your dataset
data = pd.read_csv("advertising.csv")
print("\n✅ Dataset Loaded Successfully")
print(data.head())

# STEP 2: Data Preprocessing
X = data[['TV', 'Radio', 'Newspaper']]  # independent variables
y = data['Sales']  # target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# STEP 3: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully")

# STEP 4: Predict
y_pred = model.predict(X_test)

# STEP 5: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 MODEL EVALUATION RESULTS")
print(f"Mean Squared Error  : {mse:.2f}")
print(f"R² Score             : {r2 * 100:.2f}%")

# STEP 6: Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# STEP 7: Display Coefficients
print("\n📌 Model Coefficients (feature importance):")
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
