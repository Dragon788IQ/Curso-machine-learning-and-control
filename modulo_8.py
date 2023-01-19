from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

#Controlar la aletoriedad de la seleccion de nuestros datos
X = pd.read_csv('X_opening.csv')
y = X['worldwide_gross']
X = X.drop('worldwide_gross', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)

X2_train, X2_test, y2_train, y2_test = train_test_split(X,y, random_state=1)

print(pd.DataFrame.equals(X_train, X2_train))

print(Lasso().fit(X_train, y_train).score(X_test, y_test))

scores = cross_val_score(Lasso(), X, y, cv=5, scoring='r2')
print(scores)
print(np.mean(scores))
print('*'*80)

results = cross_validate(Lasso(), X, y, cv=5, return_train_score=True)
test_score = results['test_score']
train_score = results['train_score']

print(f'test score= {test_score}, mean = {np.mean(test_score)}')
print(' ')
print(f'train score = {train_score}, mean = {np.mean(train_score)}')
print(' ')

#Probando otro modelo a ver si logramos mejorar la diferencia entre el score de train and test
#Usaremos una validacion de curvas para ver el comportamiento de los datos
#print(cross_validate(KNeighborsRegressor(), X, y, cv=5))

n = np.arange(2,50,2)
train_scores, test_scores = validation_curve(KNeighborsRegressor(), X, y, param_name='n_neighbors', param_range=n, cv=5)
print(np.mean(train_scores, axis=1))
plt.plot(np.mean(test_scores, axis=1), label='test_score')
plt.plot(np.mean(train_scores, axis=1), label='train_score')
plt.legend()
plt.grid()
plt.xticks(np.arange(24),n)
plt.show()

from sklearn.model_selection import learning_curve

lcurve = learning_curve(KNeighborsRegressor(n_neighbors=6), X, y, cv=5)
samples, train, test = lcurve[0], lcurve[1], lcurve[2]
plt.plot(samples[1:], np.mean(train, axis=1)[1:])
plt.plot(samples[1:], np.mean(test, axis=1)[1:])
plt.show()

#Como solucionar el overfitting y el underfitting?
#
#Viarianza Alta:
#   Conseguir más ejemplos
#   Reducir cantidad de features
#   Aumentar coeficiente de regularización
#Bias Alto:
#   Más features
#   Modelo más complejo