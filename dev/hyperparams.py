models_parameters = {
    "n_estimators": [10, 25, 50, 100, 200, 1000],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma":[0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}

def get_params_combinations(parameters):

    '''
    Get a dictionary of lists of hyperparameters and get all values combination
    Status: in prep
    '''

    names_values_combinations = []
    for name, params in parameters.items():
        for val in params:
            names_values_combinations.append((name, val))
