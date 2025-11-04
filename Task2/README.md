🎬 Movie Rating Prediction

📖 Introduction
The **Movie Rating Prediction** project aims to analyze historical movie data and build a **machine learning model** that predicts movie ratings given by users or critics.  
By studying patterns from IMDb-style data such as genres, duration, director, and votes, the model helps understand what factors most influence a movie’s success.  
This project demonstrates practical knowledge of **data analysis, feature engineering, model training, and evaluation**.

📊 Dataset
- **Source:** IMDb or publicly available movie dataset  
- **Files Used:**
  - `IMDb_Movies.csv` — raw dataset  
  - `cleaned_movies.csv` — cleaned and processed dataset  
- **Key Attributes:**
  - `Title`, `Genre`, `Director`, `Runtime`, `Votes`, `Year`, `Rating`  
- The dataset is preprocessed to handle missing values, encode categorical data, and scale numerical features for optimal model performance.

🧰 Tools and Libraries Used
- **Programming Language:** Python  
- **Libraries:**
  - `pandas` — Data manipulation  
  - `numpy` — Numerical computations  
  - `matplotlib` & `seaborn` — Data visualization  
  - `scikit-learn` — Machine learning model building  
- **Environment:** Jupyter Notebook / VS Code  

⚙️ Project Workflow
1. **Data Collection:**  
   Load the IMDb dataset containing various movie attributes.
2. **Data Preprocessing:**  
   - Handle missing or inconsistent data  
   - Encode categorical variables (e.g., genre, director)  
   - Scale numerical features such as votes or runtime  
3. **Exploratory Data Analysis (EDA):**  
   - Visualize relationships between features and movie ratings  
   - Identify patterns and correlations  
4. **Model Development:**  
   - Train and evaluate regression models  
   - Selected algorithm: **Random Forest Regressor**  
   - Hyperparameter tuning for better accuracy  
5. **Model Evaluation:**  
   - Metrics used: R² Score, Mean Absolute Error (MAE)  
   - Save trained model as `random_forest_model.pkl`
6. **Prediction:**  
   - Predict ratings for new/unseen movies  
   - Evaluate and compare predictions with actual ratings  
7. **Result Visualization:**  
   - Display feature importance in `feature_importance.png`  
   - Store performance metrics in `model_performance.txt`

🔍 Key Insights
- Features like **number of votes, duration, and genre** significantly affect the final movie rating.  
- The **Random Forest model** achieved a strong performance with:
  - **R² Score:** ~0.95  
  - **MAE:** ~0.11  
- Higher IMDb votes and longer runtime correlate positively with better ratings.  
- Ensemble models outperform linear models for this dataset.

🧾 Conclusion
This project successfully demonstrates how data-driven insights can be used to predict movie ratings.  
By applying machine learning techniques, we can understand what makes a movie successful and estimate how audiences might respond.  
The workflow can be extended to include **sentiment analysis of reviews** or **box office prediction** for future enhancement.
📬 Contact
**Project:** Movie Rating Prediction  
**Email:** tanuja6305@gmail.com

> ⭐ *This project demonstrated my skills in data analysis, machine learning, and model evaluation.*


