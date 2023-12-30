import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

class Model:
    def __init__(self) -> None:
        model_path = 'models'

        self.model = load(f'{model_path}/logre.joblib')
        self.encoder = load(f'{model_path}/encoder.joblib')
        self.scaler = load(f'{model_path}/scaler.joblib')
    
    def preprocess(self, features):
        features['thal'].replace({'fixed defect':'fixed_defect' , 'reversable defect': 'reversable_defect' }, inplace =True)
        features['cp'].replace({'typical angina':'typical_angina', 'atypical angina': 'atypical_angina' }, inplace =True)

        features['sex'] = (features['sex'] == 'Male')*1
        features['fbs'] = (features['fbs'])*1
        features['exang'] = (features['exang'])*1

        features.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 
                    'cholesterol', 'fasting_blood_sugar',
                    'max_heart_rate_achieved', 'exercise_induced_angina', 
                    'st_depression', 'st_slope_type', 'num_major_vessels', 
                    'thalassemia_type']
        
        X_test = self.encoder.transform(features)
        self.scaler.clip = False
        X_test = self.scaler.transform(X_test)

        return X_test
    
    def serving_pipeline(self, features):
        prepocessed_features = self.preprocess(features)
        prediction = self.model.predict(prepocessed_features)
        return prediction
