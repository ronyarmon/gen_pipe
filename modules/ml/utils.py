from libraries import *

def get_feature_importance(model, features, result_as = 'dict'):

    '''
    build feature importance table
    :params:
    model: The model used
    features: The features to map to their calculated importance
    '''

    importance_scores = model.feature_importances_
    if result_as == 'dict':
        features_importance = dict(zip(features, importance_scores))
    else:
        features_importance = pd.DataFrame \
            (list(zip(features, importance_scores)),\
             columns=['feature', 'importance'])
        features_importance = features_importance.sort_values\
            (by='importance', ascending=False)

    return features_importance

