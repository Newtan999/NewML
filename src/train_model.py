import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Load data
df = pd.read_csv("../data/treated/df_final.csv")

# Features and target
X = df.drop(["heartdisease", "max_hr_predicted", "oldpeak"], axis=1)
y = df["heartdisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=16
    )

clf = LogisticRegression(max_iter=1500)

# Train
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")

# Save model
joblib.dump(clf, "../models/logistic_model.pkl")