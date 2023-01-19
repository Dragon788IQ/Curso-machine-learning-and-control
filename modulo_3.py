import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X = pd.read_csv('X_opening.csv')
X = X.drop('worldwide_gross', axis=1)

sns.heatmap(X.corr(), annot = True)
plt.show()