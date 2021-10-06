import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

data = pd.read_csv('./data/boston_data.csv')
print(data.head())

#split target and features
X, y = data.iloc[:, :-1], data.iloc[:,-1]

#Convert the dataset into Dmatrix
data_dmatrix = xgb.DMatrix(data=X,label=y)

#train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# instantiate an XGBoost regressor object: xg_reg
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

#fit and predict
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)

#Compute rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: {r}".format(r=rmse))
