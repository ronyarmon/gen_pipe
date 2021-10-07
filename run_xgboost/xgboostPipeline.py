import sys
modules_path = '../modules'
sys.path.append(modules_path)

from libraries import *
from vizz.utils import *
from vizz.plots import *
from ml.preProcess import *
from ml.explore import *

# Dataset
dataset_path = ('./data/california_data.csv')
dataset = pd.read_csv(dataset_path)
target = 'PRICE'
metadata = ['latitude', 'longitude']
categorical = ['ocean_proximity']

exclude_columns = metadata + categorical
feature_names = [c for c in dataset.columns if c not in exclude_columns]
dataset = dataset[feature_names]

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
X = dataset.drop([target], axis=1)
y = dataset[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

params = {'n_estimators':100, 'max_depth':3, 'learning_rate': 0.01}
#model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.01)
model = xgb.XGBRegressor(**params)
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


