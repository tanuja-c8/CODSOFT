📊 Sales Prediction using Python 

✅ Project Overview

This project focuses on predicting product sales based on advertising expenditure across different marketing channels. Using Machine Learning regression techniques, the model forecasts how advertising budgets on **TV, Radio, and Newspaper** impact product sales.

Sales prediction is essential in businesses to help:

* Estimate future sales
* Allocate advertising budget effectively
* Improve decision-making with data-driven insights

📂 Dataset

You can use any dataset with the following columns:

Dataset should contain the following columns:

* TV – Advertising budget spent on TV (in thousands)
* Radio – Advertising budget spent on Radio (in thousands)
* Newspaper – Advertising budget spent on Newspaper (in thousands)
* Sales – Sales generated (in thousands of units)

Example dataset used: **advertising.csv****
🧠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

🚀 Project Workflow

1. Import the dataset
2. Preprocess the data (feature selection)
3. Split the dataset into train & test sets
4. Train the model using **Linear Regression**
5. Predict sales based on advertisement budget
6. Evaluate the model using:

   * Mean Squared Error (MSE)
   * R² Score
7. Visualize Actual vs Predicted Sales


🧾 Code Snippet (Main Logic)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


📊 Output Results

* Displays dataset preview
* Prints model performance metrics
* Shows graph of Actual vs Predicted Sale

📈 Visualization

Actual Sales  vs  Predicted Sales


This helps understand how well the model performs on unseen data.



▶️ How to Run the Project

1. Install required libraries:

pip install pandas numpy scikit-learn matplotlib

2. Run the script:

python sales_prediction.py

🏁 Conclusion

The model successfully predicts sales using Linear Regression. Businesses can utilize this to **optimize media budgets** and increase revenue through targeted advertising.


🔗 GitHub Repository

https://github.com/tanuja-c8/CODSOFT

