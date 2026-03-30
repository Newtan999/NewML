import pandas as pd
import joblib

model = joblib.load("../models/logistic_model.pkl")

new_data = pd.DataFrame([{
    "age": 60,
    "restingbp": 130,
    "cholesterol": 250,
    "maxhr": 140,
    "hr_ratio": 0.85,
    "sex": 1,
    "fastingbs": 1,
    "exerciseangina": 0,
    "elderly": 1,
    "restingecg_normal": 1,
    "restingecg_st": 0,
    "st_slope_flat": 1,
    "st_slope_up": 0,
    "oldpeak_risk_low": 0,
    "oldpeak_risk_moderated": 1,
    "chestpaintype_ata": 0,
    "chestpaintype_nap": 1,
    "chestpaintype_ta": 0
}])

prediction = model.predict(new_data)
                    
print(f"Prediction: ", prediction)