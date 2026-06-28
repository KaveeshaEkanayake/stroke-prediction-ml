# Stroke Prediction System

## Project Overview

A machine learning system that predicts stroke risk in patients using medical and demographic data.
Built as part of my AI/ML learning journey to demonstrate end-to-end machine learning skills.

## Dataset

* Source: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
* 5110 patient records with 12 features
* Binary classification: stroke (1) or no stroke (0)
* Class imbalance: only 4.9% positive stroke cases

## Project Pipeline

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Handling Class Imbalance with SMOTE
4. Model Training and Evaluation
5. Model Optimization
6. Feature Importance Analysis

## Key Challenges

* Severe class imbalance (4.9% stroke cases) solved using SMOTE
* 201 missing BMI values handled using median imputation
* Categorical encoding for text features using Label Encoding and One Hot Encoding

## Models Trained

|Model|Accuracy|Precision|Recall|ROC-AUC|
|-|-|-|-|-|
|Logistic Regression|84%|17%|56%|78%|
|Decision Tree|88%|8%|14%|-|
|Random Forest|92%|19%|16%|-|

## Best Model: Logistic Regression

Selected based on highest recall (56%)  catching the most actual stroke cases.
In healthcare, missing a real stroke is far more dangerous than a false alarm.

## Key Findings

* Age is the strongest predictor of stroke risk
* Smoking history significantly increases stroke risk
* Work type and marital status show meaningful correlation with stroke risk
* Average glucose level (diabetes indicator) influences stroke prediction

## Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib, Seaborn

## How to Run

1. Clone this repository
2. Download the dataset from Kaggle (link above)
3. Install dependencies: pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
4. Open stroke\_prediction.ipynb in Jupyter Notebook
5. Run all cells from top to bottom

## Files

* stroke\_prediction.ipynb  main notebook with full code and analysis
* stroke\_prediction\_model.pkl  saved trained model
* scaler.pkl  saved scaler for preprocessing new data



