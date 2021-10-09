import sys
modules_dir = '/home/rony/gen_pipe/modules'
sys.path.append(modules_dir)
from libraries import *
from ml.utils import *

data = pd.read_excel('./data/synthetic_data.xlsx')
features = list(data.iloc[:, :-1].columns)
print(data.head())

#split target and features
X, y = data.iloc[:, :-1], data.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

hyper_params = {
    "n_estimators": [10, 25, 50, 100, 200, 1000],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.8],
    "min_child_weight": [1, 3, 5, 7, 9]}
hp_names = list(hyper_params.keys())

combinations = []
counter = 0
for hp1 in hyper_params["n_estimators"]:
    for hp2 in hyper_params["max_depth"]:
        for hp3 in hyper_params["learning_rate"]:
            for hp4 in hyper_params["colsample_bytree"]:
                for hp5 in hyper_params["min_child_weight"]:
                    counter+=1
                    hp_values = [hp1, hp2, hp3, hp4, hp5]
                    combination = dict(zip(hp_names, hp_values))
                    combinations.append(combination)

start = time.time()
def run_model(hyper_params_conf):
    print('combination: {co}'.format(co=hyper_params_conf))

    model = xgb.XGBRegressor(**hyper_params_conf)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              early_stopping_rounds=20)

    features_importances = get_feature_importance(model, features)
    print('features importance:')
    print(features_importances)

    # Evaluate mode: rmse
    ## train set
    preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, preds))
    print("Train RMSE: {r}".format(r=train_rmse))

    ## test set
    preds = model.predict(X_test)
    eval_rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("Eval RMSE: {r}".format(r=eval_rmse))

    # Cross validation evaluation
    scores = cross_val_score(model, X, y, \
                             cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-scores.mean())
    print("Mean cv score: %.2f" % cv_rmse)

    return [hyper_params_conf, train_rmse, eval_rmse, cv_rmse, features_importances]

results = []
results_headers = ['hyper_params', 'train_rmse', 'eval_rmse',\
                   'cv_rmse', 'features_importances']

# combinations = combinations[:100]
num_excutors = 4
print('num_excutors:', num_excutors)
def controller():
    executor = ProcessPoolExecutor(num_excutors)
    for result in executor.map(run_model, combinations):
        results_df = pd.read_excel('./results/hyper_params_tuning.xlsx')
        results.append(result)
        results_df = pd.DataFrame(results, columns=results_headers)
        results_df.to_excel('./results/hyper_params_tuning.xlsx', index=False)

print('{n} hyper parameters combinations to test'.format(n=len(combinations)))
controller()
end = time.time()
duration = round(end-start, 2)
print(duration)