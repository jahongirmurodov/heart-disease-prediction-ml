import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


df = pd.read_csv('data/HeartFailureDataset.csv')

df['bmi'] = df['weight'] / ((df['height']/100)**2) # BMI - ИМТ
df['age_years'] = df['age'] / 365 # Перевод возраста в года



X = df.drop(columns=['cardio', 'id'])
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Пробуем модель Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print('--' * 45)
print('--- Метрики точности ---')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'ROC-AUC: {roc_auc:.4f}')

print('--' * 45)
print('Модель Random Forest демонстрирует показатели выше среднего на данном датасете.')
print('Несмотря на то, что она лучше случайного классификатора, значения Accuracy, Recall и ROC-AUC недостаточно высоки для медицинского применения.')
print('Основная проблема заключается в том, что модель пропускает значительную долю пациентов с заболеванием,')
print('Модель лучше чем KNN K=5 или K=9, но было решение выбрать именно Logistic Regression')
print('так как модель более лучше подходит для медицинских датасетов.')