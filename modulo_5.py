from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.simplefilter("ignore")

X = pd.read_csv("X_opening.csv")
y = X['worldwide_gross']
X = X.drop('worldwide_gross', axis=1)
#X = X.drop('opening_gross', axis=1)


selector = SelectKBest(mutual_info_regression, k=4)
selector.fit(X,y)

scores = selector.scores_
plt.rcParams['figure.figsize'] = [12,8]
plt.plot(scores)
plt.xticks(np.arange(9), list(X.columns))
#plt.show()

X2 = X[['production_budget', 'title_year','duration.1','cast_total_facebook_likes', 'imdb_score']]
X3 = X[['production_budget', 'cast_total_facebook_likes', 'imdb_score']]

X_train, X_test, y_train, y_test = train_test_split(X,y)

cols2 = ['production_budget', 'title_year','duration.1','cast_total_facebook_likes', 'imdb_score']
X2_train, X2_test, y2_train, y2_test = X_train[cols2], X_test[cols2], y_train, y_test

cols3 = ['production_budget', 'cast_total_facebook_likes', 'imdb_score']
X3_train, X3_test, y3_train, y3_test = X_train[cols3], X_test[cols3], y_train, y_test

model1 = Lasso()
model2 = Lasso()
model3 = Lasso()

model1.fit(X_train, y_train)
model2.fit(X2_train, y2_train)
model3.fit(X3_train, y3_train)

print(model1.score(X_test, y_test))
print(model2.score(X2_test, y2_test))
print(model3.score(X3_test, y3_test))