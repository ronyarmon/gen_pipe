#!/usr/bin/env python
# coding: utf-8
from libraries import *

# Count outliers per column and per dataframe
def count_column_outliers(df, column, threshold=3):
    outliers_count = len(df[df[column] >= threshold])
    return outliers_count

def count_df_outliers(df):
    '''
    Count outliers by zscore in a standardized dataframe columns
    :param df: dataframe standardized by the zscore method
    :param threshold: The zscore to apply as outliers threshold
    :return:
    '''
    rows_count = len(df)
    columns = df.columns
    outliers_summary = {column: count_column_outliers(df, column)\
                        for column in columns}
    outliers_summary = pd.DataFrame(list(zip(outliers_summary.keys(), outliers_summary.values())),
                                    columns=['column', 'outliers_count'])
    outliers_count = np.array(outliers_summary['outliers_count'])
    outliers_perc = np.round(100 * outliers_count/ rows_count, 2)
    outliers_summary['%outliers'] = outliers_perc
    return outliers_summary

def get_nulls_count_percent(df, threshold=10):
    '''
    Produce dataframe info as a pandas dataframe that can be then be written to excel
    '''
    df_headers = ['column', 'nulls_count']
    rows_count = len(df)
    null_count = df.isna().sum()
    nulls_summary = pd.DataFrame(list(zip(null_count.index, null_count.values)), \
                                columns=df_headers)
    null_perc = round(100 * null_count / rows_count, 2)
    nulls_summary['%nulls'] = null_perc.values

    return nulls_summary

def concat_column_results(dfs):

    '''
    Collect summary statistics, nulls and outliers percentages for each dataframe column
    :param dfs: Results dataframe
    :return: Extended info dataframe
    '''

    # Results summary
    extended_info = reduce(lambda left, right: pd.merge(left, right,\
                        on=['column'], how='outer'), dfs)
    count_columns = [c for c in list(extended_info.columns) if 'count' in c]
    extended_info = round(extended_info.drop(count_columns, axis=1), 2)
    extended_info = extended_info[['column', 'median', 'mean', 'std',\
                                   '%nulls', '%outliers']]

    return(extended_info)


def get_correlated_features(corr_matrix, threshold=0.90):

    '''
    Identify highly correlated features
    :param corr_matrix: Correlation matrix
    :param threshold: The correlation cutoff
    :return: A list of the correlated features
    source: https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    '''

    print('corr_matrix')
    print(corr_matrix)

    n_cols = len(corr_matrix.columns)
    results = []
    for i in range(n_cols):
        for k in range(i + 1, n_cols):
            val = corr_matrix.iloc[k, i]
            col = corr_matrix.columns[i]
            row = corr_matrix.index[k]
            if abs(val) >= threshold:
                # Prints the correlated feature set and the corr val
                val = round(val, 2)
                print(col, "|", row, "|", val)
                #drop_cols.append(col)
                results.append([col, row, val])

    return pd.DataFrame(results, columns=['featureA', 'featureB', 'correlation'])
