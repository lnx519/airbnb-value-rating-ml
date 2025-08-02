# Predicting Airbnb Value Ratings with Machine Learning

This project explores a supervised machine learning workflow to predict the **value rating** of Airbnb listings based on key listing and host features. The dataset includes information such as accommodation size, minimum stay, availability, and booking policies. We use both linear and ensemble models, as well as hyperparameter tuning, to identify the best predictive approach.

---

## 🧠 Objectives
- Load and clean Airbnb listing data using Pandas
- Identify and define a regression ML problem
- Perform exploratory data analysis (EDA)
- Train and evaluate Linear Regression and Random Forest models
- Use GridSearchCV for model optimization
- Interpret results and feature importance

---

## 🔧 Methodology
1. **Data Selection & Preprocessing**
   - Selected relevant numerical and categorical features
   - Handled missing values and converted binary features
   - Split into training and test sets (70/30)

2. **Modeling**
   - Trained baseline `LinearRegression` model
   - Trained `RandomForestRegressor` model
   - Performed grid search over `n_estimators` and `max_depth` using 3-fold CV

3. **Evaluation Metrics**
   - Root Mean Squared Error (RMSE)
   - R² Score

---

## 📊 Results & Key Findings
- **Best RMSE:** ~0.50 using tuned Random Forest
- **R² Score:** ~0.04 — predictions are close but explain little variance
- Feature importance highlights `accommodates`, `minimum_nights`, and `host_is_superhost` as top contributors

---

## 📌 Next Steps
- Explore additional features (e.g., text reviews, price)
- Handle score imbalance using stratified sampling or reweighting
- Try gradient boosting or neural nets for improved performance

---

## 👩‍💻 Author
Ningxin Li & Tessa Volpe  
BU Principles of Machine Learning – Final Lab  
August 2025
