import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from category_encoders import OneHotEncoder


# Read Data
data = pd.read_csv('heart_disease_uci.csv')

# Preprocessing
data.dropna(inplace = True)

data['thal'].replace({'fixed defect':'fixed_defect' , 'reversable defect': 'reversable_defect' }, inplace =True)
data['cp'].replace({'typical angina':'typical_angina', 'atypical angina': 'atypical_angina' }, inplace =True)

data_tmp = data[['age','sex','cp', 'trestbps', 'chol', 'fbs',  'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].copy()
data_tmp['target'] = ((data['num'] > 0)*1).copy()
data_tmp['sex'] = (data['sex'] == 'Male')*1
data_tmp['fbs'] = (data['fbs'])*1
data_tmp['exang'] = (data['exang'])*1

data_tmp.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 
              'cholesterol', 'fasting_blood_sugar',
              'max_heart_rate_achieved', 'exercise_induced_angina', 
              'st_depression', 'st_slope_type', 'num_major_vessels', 
              'thalassemia_type', 'target']

data = data_tmp

# Split data into train set and test set
X = data.drop('target', axis = 1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_test: {y_test.shape}')

# Preprocessing
encoder = OneHotEncoder(cols=['chest_pain_type', 'fasting_blood_sugar', 'exercise_induced_angina', 'st_slope_type', 'thalassemia_type'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

columns_name = X_train.columns

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
dctree = DecisionTreeClassifier(criterion='entropy')
logre = LogisticRegression(solver='sag', penalty='l2')

knn.fit(X_train, y_train)
dctree.fit(X_train, y_train)
logre.fit(X_train, y_train)

folder_path = 'models'

dump(knn, f'{folder_path}/knn.joblib')
dump(dctree, f'{folder_path}/dctree.joblib')
dump(logre, f'{folder_path}/logre.joblib')
dump(encoder, f'{folder_path}/encoder.joblib')
dump(scaler, f'{folder_path}/scaler.joblib')

clf = load(f'{folder_path}/logre.joblib')

print(classification_report(y_test, clf.predict(X_test)))

print(clf.intercept_)
coeffecients = pd.DataFrame(clf.coef_.ravel(), columns_name)
coeffecients.columns = ['Coeffecient']
coeffecients.sort_values(by=['Coeffecient'], inplace=True, ascending=False)
print(coeffecients)

