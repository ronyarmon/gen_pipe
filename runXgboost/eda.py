import sys
modules_dir = '/home/rony/gen_pipe/modules'
sys.path.append(modules_dir)

from vizz.utils import *
from vizz.plots import *
from ml.preProcess import *
from ml.explore import *
from libraries import *

# Dataset
target = 'median_house_value'
print ('target:', target)
metadata = ['latitude', 'longitude']
categorical = ['ocean_proximity']
dataset_path = ('./data/california_data.csv')
dataset = pd.read_csv(dataset_path)

exclude_columns = metadata + categorical
keep_columns = [c for c in dataset.columns if c not in keep_columns]
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

# Mean, standard deviation and median
desc_stats = dataset.describe().T[['mean', 'std', '50%']]
desc_stats = desc_stats.rename(columns={'50%': 'median'})
desc_stats['column'] = desc_stats.index
print('descriptive statistics')
print(desc_stats)
desc_stats.to_excel('./results/desc_stats.xlsx', index=False)

# Nulls detection
nulls_summary = get_nulls_count_percent(dataset)
print('nulls_summary')
print(nulls_summary)
nulls_summary.to_excel('./results/nulls_summary.xlsx', index=False)

# Nulls decision
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
nulls_summary.to_excel('./results/nulls_summary.xlsx', index=False)

# Outliers decision
action = input('outliers action(d=drop, i=ignore):')
print('{n1} rows before outliers removal'.format(n1=len(dataset)))
print(dataset.head())
if action == 'd':
    dataset = drop_outliers(dataset, target=target)
    print('{n2} rows after outliers removal'.format(n2=len(dataset)))

features = dataset[feature_names]

# Features auto-correlation
#### Visualizing correlation
# Create correlation matrix
features_corr = features.corr().abs()
print('Features correlated with other features')
masked_heatmap(features_corr)

correlated_features = get_correlated_features(features_corr, threshold=0.50)
print('Top correlated features')
print(correlated_features)
action = input('Drop correlated features?(y=yes')
if action == 'y':
    print('Update drop_features.txt, each feature in a newline')
    confirm_update = input('drop_features updated?(y=yes')
    if confirm_update == 'y':
        features_to_drop = open('drop_features.txt').read().split('\n')
        dataset = dataset.drop(features_to_drop, axis=1)

print('Cleaned Dataset')
print(dataset.info())
print(dataset.head())