import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # Basic preprocessing:
    df = df.copy()
    # One-hot encode genre (keep top genres and 'Other')
    top_genres = df['genre'].value_counts().nlargest(10).index.tolist()
    df['genre'] = df['genre'].where(df['genre'].isin(top_genres), 'Other')
    df = pd.get_dummies(df, columns=['genre'], drop_first=True)
    # Director frequency encoding
    director_counts = df['director'].value_counts().to_dict()
    df['director_popularity'] = df['director'].map(director_counts)
    # Fill NA numeric with median
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())
    # Drop non-numeric identifiers
    identifiers = ['title', 'director']
    for col in identifiers:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

if __name__ == "__main__":
    df = load_data("data/cleaned_movies.csv")
    print("Loaded", len(df), "rows")
    dfp = preprocess(df)
    print("Columns after preprocessing:", dfp.columns.tolist())
