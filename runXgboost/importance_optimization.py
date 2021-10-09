import sys
modules_dir = '/modules'
sys.path.append(modules_dir)
from libraries import *
from ml.utils import *
import ast

# # Training data
# data = pd.read_excel('./data/synthetic_data.xlsx')
# features = list(data.iloc[:, :-1].columns)
# print(data.head())
# X, y = data.iloc[:, :-1], data.iloc[:,-1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Tuning results
tuning_results = pd.read_excel('./results/hyper_params_tuning.xlsx')
print(tuning_results.info())
top_models = tuning_results.sort_values(by='eval_rmse')[:100]
hyper_params_sets = [ast.literal_eval(fi) \
                     for fi in list(top_models['hyper_params'])]
importance_features_scores = [ast.literal_eval(fi)\
                for fi in list(top_models['features_importances'])]

scores_dict = importance_features_scores[0]
print(scores_dict)
scores_dict = {k: v for k, v in scores_dict.items() if v > 0}
scores = list(scores_dict.values())
for s in scores: print(s)

mean_score, median_score = np.mean(scores), np.median(scores)

print(mean_score, median_score)

def get_threshold(importance_dict):
    vals = list(importance_dict.values())

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