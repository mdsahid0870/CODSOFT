# Task 3: Iris Flower Classification

## Objective
The objective of this project is to train a machine learning model that can learn from Iris flower measurements (sepal length, sepal width, petal length, petal width) and accurately classify the Iris flowers into their respective species: *setosa*, *versicolor*, and *virginica*.

## Overview
This task is part of the CodSoft Data Science Internship. The Iris dataset is widely used for introductory classification tasks. 

## Steps Performed
1. **Data Loading:** Loaded the built-in Iris dataset using `scikit-learn`.
2. **Data Exploration:** Converted the dataset into a Pandas DataFrame for easier manipulation and exploration.
3. **Data Splitting:** Separated the features (measurements) from the target variable (species) and split the data into training and testing sets.
4. **Model Training:** 
   - Standardized the features to ensure all measurements contribute equally to the model.
   - Trained a **Logistic Regression** model to classify the species.
5. **Evaluation:** Evaluated the model's accuracy on the test data and printed a Classification Report to verify precision and recall across all three species.

## How to Run
Open `Iris_Flower_Classification.ipynb` in Jupyter Notebook and execute the cells to reproduce the classification model and view the evaluation metrics.
