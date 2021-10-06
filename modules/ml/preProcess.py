#!/usr/bin/env python
# coding: utf-8
from libraries import *

def coerce_fill_median(df):

    '''
    Coerce the values of the dataframe to numeric types to allow for the calculation of fillna
    values, than fill the null values of the dataframe using that value (here: median)
    :param df:
    :return:
    '''
    df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.median())
    return df

def check_encode_categorical (df):
    import pandas as pd
    df_cat = df.select_dtypes(exclude=['int'])
    df_num = df.select_dtypes(include=['int'])
    df_cat_dummies = pd.get_dummies(df_cat)
    df1 = pd.concat ([df_num,df_cat_dummies],axis=1)
    return (df1)

target = 'target'
def drop_outliers(X, target=target, threshold = 3):

    '''
    Drop outliers from each column in an input data
    :params:
    X: Dataset
    target: The name of the target column
    threshold: The zscore level defining outliers
    :return:
    The normalized dataset as a combination of standardized features but with the
    normal values for the target
    '''

    # Standardize the data
    standX = X.transform(zscore)
    normal_X = standX[(np.abs(standX) < threshold).all(axis=1)]

    # Indices for target outliers
    stand_y = standX[[target]]
    y_outliers = pd.Series((np.abs(stand_y) >= threshold).any(axis='columns'))
    y_outliers_indices = list(y_outliers[y_outliers].index)

    # Drop rows with target outliers
    source_y = X[[target]]
    normal_y = source_y[~source_y.index.isin(y_outliers_indices)]

    # Replace the target values in X by the normal target values as in the source
    normal_X = normal_X.drop([target], axis=1)
    normal_X[target] = normal_y

    return normal_X