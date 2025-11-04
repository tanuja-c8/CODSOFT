import pickle
import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess, load_data

def load_model(path="models/random_forest_model.pkl"):
    with open(path,"rb") as f:
        return pickle.load(f)

def predict_example(model):
    # Example input matching preprocessing assumptions:
    example = pd.DataFrame([
        {"title":"Example Movie","year":2024,"genre":"Action","duration":110,"director":"A. Kumar","budget_musd":30,"box_office_musd":80,"votes":50000,"rating":0}
    ])
    # append the example so get_dummies uses consistent columns.
    df = load_data("data/cleaned_movies.csv")
    df2 = pd.concat([df, example], ignore_index=True)
    df2 = preprocess(df2)
    X_example = df2.drop(columns=["rating"]).tail(1)
    pred = model.predict(X_example)[0]
    print(f"Predicted rating: {pred:.2f}")
    return pred

if __name__ == "__main__":
    model = load_model()
    predict_example(model)
