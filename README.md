Heart Disease Prediction 🫀
📌 Overview

This project predicts the likelihood of heart disease using Logistic Regression and K-Nearest Neighbors (KNN).
The dataset includes medical attributes such as age, cholesterol, blood pressure, chest pain type, and more.

The objective is to demonstrate how machine learning can support early detection and preventive healthcare.

📊 Dataset

Source: UCI Heart Disease Dataset

Target variable:

0 → No Heart Disease

1 → Heart Disease Present

Features used:

Age – Age of the patient

Sex – Gender (1 = male, 0 = female)

Chest pain type – Type of chest pain

BP – Resting blood pressure

Cholesterol – Serum cholesterol

FBS over 120 – Fasting blood sugar > 120 mg/dl

EKG results – Resting electrocardiographic results

Max HR – Maximum heart rate achieved

Exercise angina – Exercise induced angina

ST depression – Depression induced by exercise

Slope of ST – Slope of the ST segment

Number of vessels fluro – Number of major vessels colored by fluoroscopy

Thallium – Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)

⚙️ Project Workflow

Data Preprocessing

Encoded categorical features

Normalized numeric features

Train-test split

Exploratory Data Analysis (EDA)

Correlation heatmap

Feature-target relationships

Models Used

Logistic Regression

K-Nearest Neighbors (KNN)

Evaluation Metrics

Accuracy Score

Precision, Recall, F1-score

Confusion Matrix

🛠️ Tech Stack

Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Dependency Management: Poetry

🚀 Installation
Option 1: Using requirements.txt
git clone https://github.com/BhagyaSreeV29/Heart_Prediction.git
cd Heart_Prediction
pip install -r requirements.txt

Option 2: Using Poetry
git clone https://github.com/BhagyaSreeV29/Heart_Prediction.git
cd Heart_Prediction
poetry install

▶️ Running the Notebook
jupyter notebook heartpred.ipynb

📈 Results
Model	Accuracy	Precision (0 / 1)	Recall (0 / 1)	F1-score (0 / 1)
K-Nearest Neighbors	0.78	0.78 / 0.79	0.86 / 0.68	0.82 / 0.73
Logistic Regression	0.88	0.87 / 0.90	0.93 / 0.82	0.90 / 0.86
