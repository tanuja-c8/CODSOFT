import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.data_preprocessing import load_data, preprocess

def train_and_save(model_path="models/random_forest_model.pkl", perf_path="results/model_performance.txt", importance_path="results/feature_importance.png"):
    df = load_data("data/cleaned_movies.csv")
    df = preprocess(df)
    X = df.drop(columns=["rating"])
    y = df["rating"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    # Save performance
    with open(perf_path, "w") as f:
        f.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}\n")
    # Feature importances plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        feat_imp = model.feature_importances_
        feat_names = X.columns.tolist()
        order = np.argsort(feat_imp)[::-1]
        plt.figure(figsize=(8,5))
        plt.bar([feat_names[i] for i in order], feat_imp[order])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(importance_path)
        plt.close()
    except Exception as e:
        print("Could not save feature importance plot:", e)
    print("Training complete. Model saved to", model_path)

if __name__ == "__main__":
    train_and_save()
