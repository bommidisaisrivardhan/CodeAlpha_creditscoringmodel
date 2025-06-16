# CodeAlpha_creditscoringmodel

## 📌 Project Title: Credit Scoring Model

### 🧠 Objective:
To predict an individual's **creditworthiness** based on historical financial data using **classification algorithms**.

---

## 🔍 Description:

This project is part of the **CodeAlpha internship**. It focuses on building a credit scoring system using machine learning. The model uses key financial indicators to predict the credit score category of a customer.

### 💼 Problem Statement:
Banks and financial institutions use credit scores to evaluate the risk associated with lending to a customer. Automating this prediction process helps streamline decisions and reduce defaults.

---

## 📂 Dataset:

The dataset includes the following attributes:
- Age
- Annual Income
- Monthly Inhand Salary
- Number of Bank Accounts
- Credit Cards
- Interest Rate
- Outstanding Debt
- Credit Utilization Ratio
- Loan Delay History
- Credit History Age
- And many more financial behavior indicators

Target Variable: `Credit_Score`  
(Encoded: 0 = Poor, 1 = Standard, 2 = Good)

---

## 🧪 Approach:

- Data Cleaning and Preprocessing
  - Handled missing values
  - Encoded categorical variables
  - Converted credit history age to months
- Feature Selection
- Model: **Logistic Regression**
- Model Evaluation Metrics:
  - Confusion Matrix
  - Classification Report
  - ROC AUC Score

---

## 📈 Results:

- Accuracy: ~56%
- ROC AUC Score: ~0.63
- Observed class imbalance (more 'Good' than 'Poor' credits)

---

## 🛠️ Technologies Used:

- Python 🐍
- Pandas
- scikit-learn
- Jupyter Notebook / VS Code
- Git & GitHub

---

