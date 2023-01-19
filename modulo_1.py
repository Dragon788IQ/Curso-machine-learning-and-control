import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

def espaciado():
    print('*'*80)

def histograma(datos, titulo, x_title, y_title):
    #plt.grid()
    plt.hist(datos, edgecolor = 'black')
    plt.title(titulo)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

peliculas_dt = pd.read_csv('peliculas.csv')
#print(peliculas_dt.head())

num = (peliculas_dt.dtypes == float)|(peliculas_dt.dtypes == np.int64)
#print(num)

num_cols = [c for c in num.index if num[c]]
#print(num_cols)

obj = (peliculas_dt.dtypes == object)
obj_cols = [c for c in obj.index if obj[c]]
#print(obj_cols)

peliculas_nums_dt = peliculas_dt[num_cols]
peliculas_object_dt = peliculas_dt[obj_cols]

#print(peliculas_nums_dt.describe())
#histograma(peliculas_nums_dt['duration'], 'Duracion', 'Minutos', 'No.de peliculas')
#histograma(peliculas_nums_dt['imdb_score'], 'IMDB score', 'Score', 'no.de peliculas')
#histograma(peliculas_nums_dt['budget'], 'Budget', 'Money', 'No.depeliculas') #Esos datos no hace sentido

#mask = (peliculas_nums_dt['budget'] > 1e9) #Aplicamos esta mascara para ver que datos destacan demasiado
#print(peliculas_dt[mask]) #Aqui vemos que no tenemos una conversion a dolares

#Nos damos cuenta que la BDD fue hecha sin diferenciar
#1.- La moneda en la que se ingresa el presupuesto y el ingreso
#La zona en la que se registro el ingreso

espaciado()
finalcials = pd.read_csv("thenumbers.csv")

finalcials = finalcials[['movie_title', 'production_budget', 'worldwide_gross']]
print(finalcials.head())

gross_opening = pd.read_csv('opening_df.csv')
espaciado()
print(finalcials.shape)
print(peliculas_dt.shape)

#Uniremos los titulos de las peliculas al df donde teniamos la info numerica
peliculas_nums_dt = pd.concat([peliculas_nums_dt, peliculas_dt['movie_title']], axis=1)
print(gross_opening.columns)
espaciado()
#Eleminaremos de las ganancias la columna Unnamed: 0 
gross_opening = gross_opening.drop('Unnamed: 0', axis=1)

#Uniremos los datos financieros con los datos del df con toda la info numerica
peliculas_df_v2 = pd.merge(finalcials, peliculas_nums_dt, on ='movie_title', how='left')
print(peliculas_df_v2.columns)
espaciado()

#Ahora uniremos las ganancias con nuestro df v2
peliculas_df_v2 = pd.merge(peliculas_df_v2, gross_opening, on='movie_title', how='left')
print(peliculas_df_v2.columns)
print(peliculas_df_v2.shape)
espaciado()

#Ya que preparamos nuesto df, Procederemos a limpiar los datos NaN o Nulos
notnull_df =peliculas_df_v2.notnull().apply(pd.Series.value_counts) #nos mostrara una tabla con la cuenta de los datos notnull
print(notnull_df)
espaciado()
dif_zero = (peliculas_df_v2 != 0).apply(pd.Series.value_counts) #Nos mostrara una cuenta de los datos diferentes de zero
print(dif_zero)
espaciado()

#Crearemos un df de datos disponibles
available = ((peliculas_df_v2 != 0) & (peliculas_df_v2.notnull()))
print(available.all(axis=1).value_counts())

#No podemos meter a nuestro algoritmo datos notnull de nuestra variable, Eliminaremos esas lineas
mask = available["worldwide_gross"]
peliculas_df_v2 = peliculas_df_v2[mask]
espaciado()
#Ahora dropearemos las columnas repetidas o innecesarias
peliculas_df_v2 = peliculas_df_v2.drop('movie_title', axis=1)
peliculas_df_v2 = peliculas_df_v2.drop('duration', axis=1)
peliculas_df_v2 = peliculas_df_v2.drop('gross', axis=1)
print(peliculas_df_v2.head())
espaciado()

#Nos damos cuenta que en screens aun tenemos datos NaN
peliculas_df_v2 = peliculas_df_v2[available.screens]
print(peliculas_df_v2.head())
print(len(peliculas_df_v2))

#Como estrategia para manejar los datos no disponibles de las columnas no objetivo
#Podemos replazar los datos por la media de la caracteristicas mencionada
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
values = imputer.fit_transform(peliculas_df_v2)
X = pd.DataFrame(values)
X.columns = peliculas_df_v2.columns
X.index = peliculas_df_v2.index
print(X.head())

#Guardaremos nuestro avance en un CSV
X.to_csv('X_opening.csv', index = False)

#pd.Series(X.index).apply(lambda x: np.inv_map.loc[x])
#X.to_csv('X.csv',index=False)