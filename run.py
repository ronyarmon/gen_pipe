import sys
modules_path = './modules'
sys.path.append(modules_path)

from libraries import *
from vizz.utils import *
from vizz.plots import *
from ml.preProcess import *
from ml.explore import *

# Dataset
target = 'median_house_value'
metadata = ['latitude', 'longitude']
categorical = ['ocean_proximity']
dataset_path = ('./dataset/dataset.csv')
if 'dataset.csv' not in os.listdir('./dataset'):
    url_path = 'https://raw.githubusercontent.com/nyandwi/public_datasets/master/housing.csv'
    url_path = urllib.request.urlretrieve(url_path)[0]
    dataset = pd.read_csv(url_path)
    dataset.to_csv(dataset_path, index=False)
else:
    dataset = pd.read_csv(dataset_path)

exclude_columns = metadata + categorical
feature_names = [c for c in dataset.columns if c not in exclude_columns]

# Models
models = [LinearRegression(), SGDRegressor(), DecisionTreeRegressor(), LinearSVR(), SVR()]


dataset = dataset[feature_names]
print(dataset.head())
# Results paths and directories
# metadata dictionary
rows_count, features_count = len(dataset), len(dataset.columns)-1
metadata_dict = {'instances': rows_count, 'features': features_count}
np.save('metadata.npy', metadata_dict)
metadata_dict1 = np.load('metadata.npy', allow_pickle=True)[()]
from paths import *
print('Run results: results/{n}'.format(n=run_dir_name))

## EDA ##
# Mean, standard deviation and median
desc_stats = dataset.describe().T[['mean', 'std', '50%']]
desc_stats = desc_stats.rename(columns={'50%': 'median'})
desc_stats['column'] = desc_stats.index
print('descriptive statistics')
print(desc_stats)

# Nulls detection/action
nulls_summary = get_nulls_count_percent(dataset)
nulls_summary.to_excel(os.path.join(tables_dir, 'nulls_count_percents.xlsx'), index=False)
print('nulls_summary')
print(nulls_summary)

dataset_columns = list(dataset.columns)
action = 'i'
while action not in ['d', 'f']:
    action = input('nulls action(d=drop,f=fill):')
if action == 'd':
    dataset = dataset.dropna()
elif action == 'f':
    strategy = input('How(m=median, me=mean,mf=most_frequent, c=constant)?')
    if action != 'c':
        imputer = SimpleImputer(strategy=imputer_strategies[strategy])
        metadata_dict['Imputer'] = action
    else:
        fill_value = input('Constan value to fill?')
        imputer = SimpleImputer(strategy=imputer_strategies[strategy], fill_value=fill_value)
        metadata_dict['Imputer'] = action+'_'+fill_action
    dataset = imputer.fit_transform(dataset)

dataset = pd.DataFrame(dataset, columns=dataset_columns)


# Outliers detection
standardized_dataset = dataset.transform(zscore)
outliers_summary = count_df_outliers(standardized_dataset)
outliers_summary.to_excel(os.path.join(tables_dir, 'outliers.xlsx'), index=False)
print('outliers_summary')
print(outliers_summary)
action = input('outliers action(d=drop, i=ignore):')
# add: replace by mean...etc as in fillna
print('{n1} rows before outliers removal'.format(n1=len(dataset)))
print(dataset.head())
if action == 'd':
    dataset = drop_outliers(dataset, target=target)
    print('{n2} rows after outliers removal'.format(n2=len(dataset)))

features = dataset[feature_names]

# Normality detection
print('Plots for Inspection')
# Attribute histograms
features.hist(bins=50, figsize=(20, 15))
save_fig("attribute_histogram_plots", images_dir)

skewed_features = features.skew()
print('features skewness')
print(skewed_features)

# Correlated features
#### Visualizing correlation
# Create correlation matrix
features_corr = features.corr().abs()
print('Features correlated with other features')
masked_heatmap(features_corr)

correlated_features = get_correlated_features(features_corr, threshold=0.50)
print('Top correlated features')
print(correlated_features)

# Train/Test
features = dataset.drop([target], axis=1)
labels = dataset[target]
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
train_features = train_set.drop([target], axis=1)
train_labels = train_set[target]
test_features = test_set.drop([target], axis=1)
test_labels = test_set[target]

# Run models
model_names = [model.__class__.__name__ for model in models]

results = []
for index, model in enumerate(models):
    model_name = model_names[index]
    print('model name:', model_name)
    model.fit(train_features, train_labels)
    predictions = model.predict(train_features)

    ## Evaluation ##
    # rmse: train set
    mse = mean_squared_error(train_labels, predictions)
    rmse_train = round(np.sqrt(mse))

    ## rmse: test set
    predictions = model.predict(test_features)
    mse = mean_squared_error(test_labels, predictions)
    rmse_test = round(np.sqrt(mse))

    # cross validation rmse
    scoring = 'neg_root_mean_squared_error'
    scores = cross_val_score(model, features, labels,  scoring=scoring, cv=5)
    scores = -scores
    cv_rmse = round(scores.mean())
    results.append([model_name, rmse_train, rmse_test, cv_rmse])

results_headers = ['model', 'rmse-train', 'rmse-test', 'cv_rmse']
models_evals = pd.DataFrame(results, columns=results_headers)
print(models_evals)