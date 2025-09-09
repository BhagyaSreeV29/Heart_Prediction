Heart Disease Prediction 🫀
Overview

Predicts the likelihood of heart disease using Logistic Regression and K-Nearest Neighbors (KNN) based on patient medical data.

Dataset

Source: UCI Heart Disease Dataset

Target: 0 → No Heart Disease, 1 → Heart Disease Present

Features: Age, Sex, Chest pain type, BP, Cholesterol, FBS over 120, EKG results, Max HR, Exercise angina, ST depression, Slope of ST, Number of vessels fluro, Thallium

Workflow

Data preprocessing: encoding & normalization

Exploratory Data Analysis

Model training: Logistic Regression & KNN

Evaluation: Accuracy, Precision, Recall, F1-score

Installation
git clone https://github.com/BhagyaSreeV29/Heart_Prediction.git
cd Heart_Prediction

# Using requirements.txt
pip install -r requirements.txt

# Or using Poetry
poetry install


Run the notebook:

jupyter notebook heartpred.ipynb

Results
Model	Accuracy	Precision	Recall	F1-score
KNN	0.78	0.78/0.79	0.86/0.68	0.82/0.73
Logistic Regression	0.88	0.87/0.90	0.93/0.82	0.90/0.86
