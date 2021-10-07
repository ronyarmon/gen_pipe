#load data
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
target = 'PRICE'
boston = load_boston()
feature_names = boston['feature_names']
headers = list(feature_names) + [target]
X, y = boston['data'], boston['target']
data = pd.DataFrame(X, columns=feature_names)
data[target] = y
data.to_csv('boston_data.csv', index=False)

