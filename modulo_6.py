from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')

X = pd.read_csv('X_opening.csv')
y = X['worldwide_gross']
X = X.drop('worldwide_gross', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y)

scaler = StandardScaler()
scaler.fit(X_train)
print('*'*80)
print(scaler.mean_)
print(scaler.scale_)
print('*'*80)
print(X.values)
print('*'*80)
print(scaler.transform(X_train))
print('*'*80)
X_scaled_train, X_scaled_test = (scaler.transform(X_train), scaler.transform(X_test))

model = Lasso()
model_scaled = Lasso()

model.fit(X_train, y_train)
model_scaled.fit(X_scaled_train, y_train)

print(model.score(X_test, y_test))
print(model_scaled.score(X_scaled_test, y_test))