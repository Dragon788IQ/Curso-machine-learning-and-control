import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')

#Desicion Tree
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=2)

from sklearn.model_selection import train_test_split
X = pd.read_csv("X_opening.csv")
y = X['worldwide_gross']
X = X.drop('worldwide_gross', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)

model.fit(X_train, y_train)
import graphviz
from sklearn.tree import export_graphviz

#treedot = export_graphviz(model, out_file=None, feature_names=X.columns)


#Modelo con Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

forest = RandomForestRegressor(200)
results = cross_validate(forest, X, y, cv=5, scoring='r2', return_train_score=True)

test_scores = results['test_score']
train_scores = results['train_score']
print(np.mean(train_scores))
print(np.mean(test_scores))

#Modelos con Gradient Boosted Trees
print('*'*80)
from sklearn.ensemble import GradientBoostingRegressor
ensemble = GradientBoostingRegressor()
results = cross_validate(ensemble, X, y, cv=5, scoring='r2', return_train_score=True)

test_scores = results['test_score']
train_scores = results['train_score']
print(np.mean(train_scores))
print(np.mean(test_scores))
