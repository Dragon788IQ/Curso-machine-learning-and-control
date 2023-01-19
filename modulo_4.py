import warnings
warnings.simplefilter("ignore")
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

X = pd.read_csv('X_opening.csv')
y = X['worldwide_gross']
X = X.drop('worldwide_gross', axis=1)


model = Lasso()
X_train, X_test, y_train, y_test = train_test_split(X,y)

print(len(X_train)/len(X))

print(model.fit(X_train,y_train))
print(model.score(X_test, y_test))

print(model.coef_)

var = np.floor(np.log10(np.abs(model.coef_)))

plt.rcParams['figure.figsize'] = [12,8]
plt.plot(var)
plt.xticks(np.arange(9), list(X.columns))
plt.grid()
plt.show()

clase = pd.cut(X['production_budget'], 8).cat.codes.rename('class')
Z = pd.concat([X,clase], axis=1)
sns.pairplot(Z, hue='class')
plt.show()
