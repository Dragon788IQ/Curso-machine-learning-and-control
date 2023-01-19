import pandas as pd 
import numpy as np
def espaciado():
    print("*"*80)

peliculas_dt = pd.read_csv("peliculas.csv", encoding='utf-8')

espaciado()

print(peliculas_dt.head()) #Nos muestra las primeras 5 lineas del df
print(peliculas_dt.shape) #Nos muestra las dimensiones del df

espaciado()
print(peliculas_dt.columns) #Nos muestra los nombres de todas las columnas en una lista

espaciado()
print(peliculas_dt.index)

espaciado()
print(peliculas_dt[peliculas_dt.columns[0]]) #Esta es una forma de acceder a una columna

espaciado()
linea = peliculas_dt.loc[10, :]
print(linea)
print(linea.shape)

espaciado()
print(peliculas_dt.loc[:,'movie_title'].head())

espaciado()
print(peliculas_dt.info()) #Info general de los tipos de datos y Non-Null Count

espaciado()
print(peliculas_dt.dtypes == object) #Nos indicara un True o False en las columnas que tengan el tipo de dato que especifcamos
print('datos enteros')
print(peliculas_dt.dtypes == np.int64) #Nos indicara un True o False en las columnas que tengan el tipo de dato que especifcamos

#Creamos un dt con los booleanos true solo si son datos int o float
espaciado()
print('Datos numericos')
num = (peliculas_dt.dtypes == float) | (peliculas_dt.dtypes == np.int64)
print(num)

#Creando una dt solo con los datos numericos
espaciado()
num_cols = [c for c in num.index if num[c]]
print(num_cols)
peliculas_nums_dt = peliculas_dt[num_cols]
print(peliculas_nums_dt.describe())

#Creando un df solo con los datos object
espaciado()
object_colums = (peliculas_dt.dtypes == object)
object_cols = [c for c in object_colums.index if object_colums[c]]
peliculas_objects_dt = peliculas_dt[object_cols]
print(object_cols)
print(peliculas_objects_dt.head()) 

#Resumen de cols
espaciado()
print(peliculas_dt.columns)
print(num_cols)
print(object_cols)