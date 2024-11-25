# Credit Score Prediction
This projet is a Midterm project of [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by [Alexey Grigorev](https://github.com/alexeygrigorev)

## Overview

The extensive dataset of customer banking and credit-related details collected over the years.
The goal is to develop a Classification ML model capable of categorizing individuals into predefined credit categories, thereby minimizing manual effort and improving efficiency.
This project provides a machine learning-based app for predicting a person's credit score based on their financial and credit-related information. 
It uses a Flask web application to serve the trained model and Docker for containerization.
The goal is to help a global finance company automate the classification of individuals into predefined credit score brackets, reducing manual effort and increasing efficiency.

Dataset contains **28 columns** and **100K rows**:
Target feature is "credit_score" : **"Good," "Poor," "Standard"**.

| **No** | **Feature**                | **Description**                                                                 |
|--------|----------------------------|---------------------------------------------------------------------------------|
| 1      | `id`                       | Unique identifier for each record.                                             |
| 2      | `customer_id`              | Unique identifier for each customer.                                           |
| 3      | `month`                    | Month of the transaction or record.                                            |
| 4      | `name`                     | Customer’s name.                                                               |
| 5      | `age`                      | The customer’s age.                                                            |
| 6      | `ssn`                      | Customer’s social security number.                                             |
| 7      | `occupation`               | The customer’s occupation.                                                     |
| 8      | `annual_income`            | The customer’s annual income.                                                  |
| 9      | `monthly_inhand_salary`    | The customer’s monthly take-home salary.                                       |
| 10     | `num_bank_accounts`        | Total number of bank accounts owned by the customer.                           |
| 11     | `num_credit_card`          | Total number of credit cards held by the customer.                             |
| 12     | `interest_rate`            | The interest rate applied to loans or credits.                                 |
| 13     | `num_of_loan`              | Number of loans the customer has taken.                                        |
| 14     | `type_of_loan`             | Categories of loans obtained by the customer.                                  |
| 15     | `delay_from_due_date`      | The delay in payment relative to the due date.                                 |
| 16     | `num_of_delayed_payment`   | Total instances of late payments made by the customer.                         |
| 17     | `changed_credit_limit`     | Adjustments made to the customer’s credit limit.                               |
| 18     | `num_credit_inquiries`     | Number of inquiries made regarding the customer's credit.                      |
| 19     | `credit_mix`               | The variety of credit types the customer uses (e.g., loans, credit cards).     |
| 20     | `outstanding_debt`         | Total amount of debt the customer currently owes.                              |
| 21     | `credit_utilization_ratio` | Proportion of credit used compared to the total credit limit.                  |
| 22     | `credit_history_age`       | Duration of the customer’s credit history.                                     |
| 23     | `payment_of_min_amount`    | Indicates if the customer pays the minimum required amount each month.         |
| 24     | `total_emi_per_month`      | Total Equated Monthly Installment (EMI) paid by the customer.                  |
| 25     | `amount_invested_monthly`  | Monthly investment amount made by the customer.                                |
| 26     | `payment_behaviour`        | Customer’s payment habits and tendencies.                                      |
| 27     | `monthly_balance`          | The remaining balance in the customer’s account at the end of each month.      |
| 28     | `credit_score`             | The customer’s credit score (target variable: "Good," "Poor," "Standard").     |


## Download the latest version
The dataset from Kaggle:
 ```
path = kagglehub.dataset_download("parisrohan/credit-score-classification")
 ```

## Problem Statement

Managing credit risk is a critical task for financial institutions. Over the years, this company has gathered vast amounts of customer data, including banking and credit information.
The task is to create an intelligent system capable of:
- Classifying individuals into credit score classes.
- Reducing the manual workload for the company's analysts.
- Delivering accurate predictions for better risk management.

## Files description

  ### The dataset cleaning, analysis and model selection were conducted using Jupyter Notebook. You can find the corresponding notebook file in this repository:
  - notebook_cleaning and preprocessing.ipynb
  - notebook_EDA_model_selection.ipynb
  ### Cleaned dataset and dataset for testing final model:
  -  cleaned_df.csv
  -  df_test.csv
  ### Trained final model with best parameters and pickled files:
  -  train.py
  -  final_model.pkl
  -  dict_vectorizer.pkl
  -  label_encoder.pkl
  ### Files for running the project locally:
  -  predict.py
  -  predict-test.py
  -  requirements.txt

## Running using Waitress as WSGI server
Please use predict.py and predict-test.py for testing model prediction.
1. Create and activate .venv:
 ```
   python -m venv venv
venv\Scripts\activate
 ```
3. Install dependencies:
 ```
pip install -r requirements.txt
 ```
4. Run predict.py:
```
waitress-serve --listen=0.0.0.0:9696 predict:app
```
5. Run predict-test.py and use the command line:
```
python predict-test.py
```

