import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

data = pd.read_csv('./data/boston_data.csv')
print(data.head())

#split target and features
X, y = data.iloc[:, :-1], data.iloc[:,-1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.01)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          early_stopping_rounds=20)

optimal_tree = model.best_ntree_limit
print('optimal_tree:', optimal_tree)

preds = model.predict(X_test)
preds_optimal_tree = model.predict(X_test, ntree_limit=optimal_tree)

# Compute rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: {r}".format(r=rmse))
rmse = np.sqrt(mean_squared_error(y_test, preds_optimal_tree))
print("optimal tree RMSE: {r}".format(r=rmse))
