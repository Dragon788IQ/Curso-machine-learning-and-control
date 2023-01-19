from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
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

model_scaled = make_pipeline(StandardScaler(), Lasso())
model_scaled.fit(X_train, y_train)

print(model_scaled.score(X_test,y_test))
