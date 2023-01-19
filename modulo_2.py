import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

#Funcion para generar histogramas
def histograma(datos, titulo, x_title, y_title):
    #plt.grid()
    plt.hist(datos, edgecolor = 'black')
    plt.title(titulo)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

#Creamos nuestro df
X = pd.read_csv('X_opening.csv')
print(X.columns)
#Ahora separamos en un df diferente nuestra variable objetivo
y = X['worldwide_gross']
X = X.drop('worldwide_gross', axis = 1)
#X = X.drop('opening_gross', axis = 1)


#Separamos los datos que tenemos para entrenar y hacer test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state = 1)

#Miramos la cantidad de datos con la que nos quedamos
print(len(X))
print(len(X_test))
print(len(X_train))

#Inicializamos el modelo
model = Lasso()
model.fit(X_train, y_train)

#Generamos las predicciones
predicted = model.predict(X_test)
print(predicted.shape)

#Mostramos los resultados de la primera iteracion usando el modelo Lasso
histograma([predicted, y_test], 'Primera iteracion', 'worldwide gross', 'Peliculas')

#Mostramos el score de nuestro modelo inicial
print(f'El score del modelo es: {model.score(X_test, y_test)}')

residuals = y_test - predicted

#plt.scatter(y_test, residuals)
#plt.show()

ap_residuals = np.abs(residuals)/y_test
#plt.scatter(y_test, ap_residuals)
#plt.show()

lap_residuals = np.log(ap_residuals)
#plt.scatter(y_test, lap_residuals)
#plt.show()


plt.hist(lap_residuals, bins=100,  density=1, histtype='step', cumulative=True)
plt.title('Evolucion del error')
plt.xlabel('Error')
plt.ylabel('Porcentaje de datos')
plt.show()