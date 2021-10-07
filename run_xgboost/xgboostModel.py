import sys
modules_path = './modules'
sys.path.append(modules_path)
#!/usr/bin/env python
# coding: utf-8
from libraries import *
from ml.utils import *

data = pd.read_csv('./data/boston_data.csv')
features = list(data.iloc[:, :-1].columns)
print(data.head())

#split target and features
X, y = data.iloc[:, :-1], data.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.01)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          early_stopping_rounds=20)

features_importances = get_feature_importance(model, features)
print('features importance:')
print(features_importances)

# save in text format
model.save_model("model_sklearn.txt")

# load the model
model2 = xgb.XGBRegressor()
model2.load_model("model_sklearn.txt")

optimal_tree = model2.best_ntree_limit
print('optimal_tree:', optimal_tree)

preds = model2.predict(X_test)
preds_optimal_tree = model2.predict(X_test, ntree_limit=optimal_tree)

# Compute rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: {r}".format(r=rmse))
rmse = np.sqrt(mean_squared_error(y_test, preds_optimal_tree))
print("optimal tree RMSE: {r}".format(r=rmse))

# Cross validation evaluation
scores = cross_val_score(model2, X, y, \
                         cv=10, scoring='neg_mean_squared_error')
rmse = np.sqrt(-scores.mean())
print("Mean cv score: %.2f" % rmse)


# from sklearn.model_selection import GridSearchCV
#
# clf = xgb.XGBRegressor()
# parameters = {
#     "eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
#     "max_depth": [ 3, 4, 5, 6, 8, 10, 12, 15],
#     "min_child_weight" : [ 1, 3, 5, 7 ],
#     "gamma":[ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
#     "colsample_bytree": [ 0.3, 0.4, 0.5 , 0.7 ]
# }
#
# grid = GridSearchCV(clf,
#                     parameters, n_jobs=4,
#                     scoring='neg_mean_squared_error',
#                     cv=3)
# grid.fit(X_train, y_train)