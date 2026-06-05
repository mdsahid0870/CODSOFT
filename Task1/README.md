# Task 1: Titanic Survival Prediction

## Objective
The objective of this project is to build a machine learning model that predicts whether a passenger on the Titanic survived or not based on their individual passenger data.

## Overview
This is a classic beginner project provided by the CodSoft Data Science Internship. The dataset contains information about individual passengers, such as their age, gender, ticket class, fare, cabin, and survival status.

## Steps Performed
1. **Data Loading:** Fetched the Titanic dataset.
2. **Data Preprocessing:**
   - Handled missing values (filled missing ages and fares with medians, and embarked locations with the mode).
   - Dropped irrelevant columns such as `name`, `ticket`, `cabin`, etc., that do not directly contribute to the survival probability in this baseline model.
   - Encoded categorical features such as `sex` (male/female) and `embarked` locations using one-hot encoding.
3. **Data Splitting:** Split the dataset into training and testing sets.
4. **Model Training:** Scaled the features and trained a **Random Forest Classifier** to predict the survival status.
5. **Evaluation:** Evaluated the model using Accuracy score and a detailed Classification Report to understand the prediction capabilities for both surviving and non-surviving passengers.

## How to Run
Open `Titanic_Survival_Prediction.ipynb` in Jupyter Notebook and execute the cells to see the preprocessing steps and the model's performance.
