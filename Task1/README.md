🚢 Titanic Survival Prediction
📘 Introduction

The Titanic Survival Prediction project aims to predict whether a passenger survived or not based on various features such as age, gender, ticket class, fare, and cabin.
This is one of the most popular beginner-level machine learning projects and helps understand data preprocessing, feature engineering, model training, and evaluation.

📂 Dataset

The dataset used for this project is the Titanic dataset, available from Kaggle’s Titanic Competition
.
It contains information about individual passengers such as:

PassengerId – Unique ID of each passenger

Pclass – Ticket class (1st, 2nd, or 3rd)

Name – Passenger’s name

Sex – Gender

Age – Age in years

SibSp – Number of siblings/spouses aboard

Parch – Number of parents/children aboard

Ticket – Ticket number

Fare – Passenger fare

Cabin – Cabin number

Embarked – Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Survived – Target variable (0 = No, 1 = Yes)

📁 File Location: data/train.csv (ensure it exists in your project folder)

🧰 Tools and Libraries Used

Python 3.x

pandas – for data cleaning and manipulation

numpy – for numerical operations

matplotlib / seaborn – for visualization

scikit-learn – for machine learning model building and evaluation

pickle – for saving the trained model

⚙️ Project Workflow

Data Loading: Load Titanic dataset from the data folder.

Data Cleaning: Handle missing values, encode categorical features, and drop irrelevant columns.

Exploratory Data Analysis (EDA): Visualize relationships and patterns using matplotlib and seaborn.

Feature Engineering: Select useful features like Pclass, Sex, Age, Fare, Embarked, etc.

Model Training: Train a Random Forest Classifier to predict survival.

Evaluation: Evaluate model performance using accuracy, confusion matrix, and classification report.

Model Saving: Save trained model as models/random_forest_model.pkl.

Prediction Results: Save predictions to results/predictions.csv.

🧠 Key Insights

Gender and class are the strongest indicators of survival — women and higher-class passengers had a higher survival rate.

Passengers with higher fares and embarked from Cherbourg (C) also had better chances.

Missing values (especially in Age and Cabin) needed proper handling for model stability.

The Random Forest model achieved ~80% accuracy, making it a reliable baseline.

✅ Conclusion

This project demonstrates the process of building a supervised classification model using real-world data.
The Titanic dataset is a benchmark example for data preprocessing, feature selection, and predictive modeling — essential steps in any data science workflow.