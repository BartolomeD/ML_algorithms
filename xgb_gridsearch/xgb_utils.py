import xgboost as xgb
from sklearn.model_selection import GridSearchCV

''' Function to perform gridsearch parameter tuning on xgboost algorithm.

# Arguments:
    x_train:     ndarray, the trainset features
    y_train:     array, the trainset labels
    params:      dict, initial xgboost parameters
    tune_params: dict, parameters to tune with value grid
    
# Returns:
    gsearch:     dict, optimal values for tune_params parameters
'''

def gridsearch(X, y, params, tune_params):

    model = xgb.XGBClassifier(learning_rate = params['learning_rate'], n_estimators = params['n_estimator']
                              , max_depth = params['max_depth'], min_child_weight = params['min_child_weight']
                              , gamma = params['gamma'], subsample = params['subsample']
                              , colsample_bytree = params['colsample_bytree'], objective = params['objective']
                              , scale_pos_weight = params['scale_pos_weight'], seed = params['seed'])

    gsearch = GridSearchCV(estimator=model, param_grid=tune_params, scoring=params['scoring']
                           , n_jobs=1, iid=False, verbose=1)

    gsearch.fit(X, y)
    print(gsearch.best_params_)

    return gsearch.best_params_

def update(base, new):
    for par in new.keys():
        base[par] = new[par]
    return base
