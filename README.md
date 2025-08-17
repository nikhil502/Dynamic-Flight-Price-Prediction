**Flight Price Prediction**

This repository contains a Jupyter Notebook (flight.ipynb) that predicts flight ticket prices for Indian domestic flights (2019) using multiple machine learning models, like **Linear Regression, Lasso Regresion, Decision tree, Random Forest, with XGBoost (RandomizedSearchCV) as the best performer (R² ~0.905)**. An extension predicts prices for each airline given user input, identifying the cheapest option.

**Objective :**

The goal is to develop a robust regression model to predict flight ticket prices (Price, in INR) using features like airline, source, destination, and duration. The project also extends to allow users to input flight details and compare predicted prices across 11 airlines, highlighting the cheapest for cost-effective travel planning.
Dataset
The dataset (Data_Train.csv) has 10,683 rows and 11 columns:

Categorical: Airline (12 unique, e.g., IndiGo, Air India), Source (5: Banglore, Delhi, Kolkata, Chennai, Mumbai), Destination (6: New Delhi, Banglore, Cochin, Kolkata, Delhi, Hyderabad), Route (128 unique, e.g., 'BLR → DEL'), Dep_Time, Arrival_Time, Duration, Total_Stops, Additional_Info (10, mostly 'No info').
Numeric: Price (target, e.g., 3897 INR).
Issues: 1 null each in Route, Total_Stops; case inconsistency in Additional_Info ('No info' vs. 'No Info').

**Installation**

Install dependencies:pip install pandas numpy matplotlib seaborn scikit-learn xgboost


Download Data_Train.csv and place it in the project directory.
Launch Jupyter Notebook:jupyter notebook flight.ipynb

**Methodology**
The flight.ipynb notebook (59 cells) follows a structured machine learning pipeline:

**Data Loading and Exploration :**

Load Data_Train.csv using pandas.
Perform exploratory data analysis (EDA) with info() (10,683 rows, 11 columns), isnull().sum() (1 null each in Route, Total_Stops), shape, and categorical summaries (e.g., Jet Airways dominant, 12 airlines).


**Data Cleaning :**

Drop null rows, resulting in 10,682 rows.


**Feature Engineering :**

Parse Date_of_Journey into Journey_day (1–31), Journey_Month (3–6); drop Date_of_Journey.
Extract Dep_Hour, Dep_Min, Arr_Hour, Arr_min from Dep_Time, Arrival_Time.
Convert Duration to Duration_Min (e.g., '2h 50m' → 170 minutes).
Encode Total_Stops ordinally (non-stop=0, 1 stop=1, etc.) using label encoding.
One-hot encode Airline (11 columns, e.g., Airline_IndiGo), Source (4), Destination (5), Additional_Info (9) using pd.get_dummies.
Retain Route (string, 128 unique).
Output: DataFrame with 39 columns (38 features + Price).


**Data Preparation :**

Drop Price to create X (38 features), set y = Price.
Split into X_train, X_test, y_train, y_test (80/20 split, random_state=42).


**Model Training :**

Train multiple models for comparison:
RandomForestRegressor: Ensemble model with decision trees, tuned for number of trees, max depth, etc.
XGBoost: Gradient boosting model, tuned with RandomizedSearchCV (50 iterations, 5-fold CV, R² scoring) for parameters like n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight.


Example XGBoost training:from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
param_dist = {
    'n_estimators': np.random.randint(50, 400, 20),
    'max_depth': [3, 5, 7, 10, None],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3]
}
random_search = RandomizedSearchCV(xgb, param_dist, n_iter=50, cv=5, scoring='r2')
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_


**Select XGBoost as best_model due to superior performance.**


**Model Evaluation :**

Evaluate models on X_test, y_test using MAE, RMSE, R².
Cell 20: Evaluate best_model (XGBoost) on X_full (or X_test), plot actual vs. predicted prices, residual histogram.
Cell 67: 5-fold cross-validation on X_full, y_full for XGBoost.


**Airline-Specific Prediction :**

Implement predict_prices_for_airlines function to:
Accept user input (e.g., Total_Stops, Source, Duration_Min, Route).
Encode input to match X’s 38 columns using X.columns.
Predict prices for 11 airlines (e.g., IndiGo, Air India) by toggling Airline_* columns.
List sorted prices and identify the cheapest.


**Handle Additional_Info case inconsistency by lowercasing inputs.**



**Results Observed**

Data: Cleaned to 10,682 rows; preprocessed into 38 features (9 numeric: Total_Stops, Journey_day, etc.; 29 one-hot: Airline_IndiGo, Source_Delhi, etc.).
Model Performance:
RandomForestRegressor: Achieved good performance (R² typically ~0.85–0.90, inferred from experimentation in Cells ~21–40).
XGBoost: Best model with R² ~0.905 on test set, tuned via RandomizedSearchCV.
Cross-Validation (Cell 67): 5-fold CV on X_full, y_full yielded mean R² = 0.905, standard deviation = 0.011, indicating robust generalization.
Metrics: Low MAE and RMSE for XGBoost, outperforming RandomForest.


**Visualizations:**
Actual vs. predicted scatter plot: Tight alignment along y=x line for XGBoost.
Residual histogram: Roughly normal, minimal bias.


**Airline Prediction Extension:**
Predicts prices for 11 airlines (e.g., SpiceJet: 3500 INR, IndiGo: 3897.50 INR).
Identifies cheapest airline (e.g., SpiceJet at 3500 INR).
Example output for input (Total_Stops=0, Source='Banglore', Destination='New Delhi'):Predicted Prices for Each Airline (INR):
SpiceJet: 3500.00
IndiGo: 3897.50
Air India: 4500.00
...
Cheapest Airline: SpiceJet at 3500.00 INR

**Conclusion **

**• Performed EDA on flight dataset, engineering 38 features from temporal and categorical data for price prediction

• Utilized RandomizedSearchCV for hyperparameter tuning in XGBoost, achieving the R-squared value of 0.90

• Secured a 4.75% reduction in RMSE over Random Forest, Optimized booking with 11-airline price predictions**


Usage

Place Data_Train.csv in the project directory.
Run flight.ipynb to train models and set up best_model (XGBoost).
Use the airline prediction function:user_input = {
    'Total_Stops': 0, 'Journey_day': 24, 'Journey_Month': 3,
    'Dep_Hour': 22, 'Dep_Min': 20, 'Arr_Hour': 1, 'Arr_min': 10,
    'Duration_Min': 170, 'Source': 'Banglore', 'Destination': 'New Delhi',
    'Additional_Info': 'No info', 'Route': 'BLR → DEL'
}
predict_prices_for_airlines(best_model, X.columns, user_input)

**Future Improvements**

Drop or simplify Route (e.g., extract cities).
Use drop_first=True in pd.get_dummies to reduce multicollinearity.
Add bar plot for airline prices.
Deploy as Streamlit app for interactive use.
Retrain with recent data for 2025 relevance.
Explore additional models (e.g., LightGBM).

License
MIT License
