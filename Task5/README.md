# Task 5: Credit Card Fraud Detection

## Objective
The objective of this project is to build a machine learning model to identify fraudulent credit card transactions. 

## Overview
This is an advanced task from the CodSoft Data Science Internship. Credit card fraud detection datasets are highly imbalanced, meaning there are very few fraudulent transactions compared to genuine ones. This requires special techniques to train an effective model.

## Steps Performed
1. **Data Loading:** Loaded the standard Credit Card Fraud Detection dataset.
2. **Data Preprocessing & Normalization:** Normalized the `Amount` and `Time` features using standard scaling, as required by the internship instructions.
3. **Handling Class Imbalance:** Handled the significant class imbalance issue by using **SMOTE (Synthetic Minority Over-sampling Technique)** to oversample the minority (fraudulent) class during the training phase.
4. **Model Training:** Split the dataset into training and testing sets, and trained a **Random Forest Classifier** algorithm to classify transactions as fraudulent or genuine.
5. **Evaluation:** As explicitly requested in the instructions, the model's performance was evaluated using multiple critical metrics:
   - **Precision**
   - **Recall**
   - **F1-Score**
   - **Accuracy**

## How to Run
Open `Credit_Card_Fraud_Detection.ipynb` in Jupyter Notebook and execute the cells. 
*(Note: If the dataset fails to download automatically via the API due to its large size, you may need to manually download `creditcard.csv` from Kaggle and place it in this folder).*
