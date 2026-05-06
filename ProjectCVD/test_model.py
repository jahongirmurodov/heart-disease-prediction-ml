import joblib
import numpy as np
model = joblib.load("models/heart_disease_model.pkl")
scaler = joblib.load("models/scaler.pkl")

user_data = np.array([[22584, 2, 178, 95, 130, 90, 3, 3, 0, 0, 1]])

# масштабирование
user_data_scaled = scaler.transform(user_data)

prediction = model.predict(user_data_scaled)

if prediction[0] == 0:
    print("Пациент здоров")
else:
    print("Обнаружен риск сердечно-сосудистого заболевания")