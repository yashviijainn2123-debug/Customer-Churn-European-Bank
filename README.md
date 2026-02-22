# Customer Churn Prediction Project

## Problem Statement
The objective of this project is to analyze customer data and build predictive models to identify customers who are likely to leave the bank (churn).

---

## Dataset Description
The dataset includes customer information such as:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Exited (Target Variable)

---

## Exploratory Data Analysis
- Checked missing values
- Performed feature engineering (Age Group, Engagement Level)
- Visualized churn distribution
- Analyzed churn by age, engagement, and number of products
- Generated correlation matrix

---

## Models Used
1. Logistic Regression
2. Random Forest Classifier

---

## Model Performance
- Logistic Regression Accuracy: ~83%
- Random Forest Accuracy: ~86%
- ROC-AUC Score: ~0.79

Random Forest performed better than Logistic Regression.

---

## Conclusion
- Approximately 20% of customers have churned.
- Inactive customers show significantly higher churn rates.
- Customers with fewer products tend to churn more.
- Engagement and Age are strong churn indicators.

---

## Recommendations
- Target inactive customers with retention campaigns.
- Offer loyalty benefits to older customers.
- Provide bundled product discounts to increase product usage.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

---

## Author
Yashvi Jain
