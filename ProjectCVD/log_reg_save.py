import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

heart_failure = pd.read_csv('data/HeartFailureDataset.csv')
heart_failure_features = heart_failure.copy().drop(["id", "cardio"], axis=1)
heart_failure_target = heart_failure["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    heart_failure_features,
    heart_failure_target,
    test_size=0.3,
    stratify=heart_failure_target,
    random_state=42,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier.fit(X_train_scaled, y_train)
lr_predictions = lr_classifier.predict(X_test_scaled)

# сохраняем scaler и model через joblib
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(lr_classifier, 'models/heart_disease_model.pkl')

