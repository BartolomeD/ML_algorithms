import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

''' Function to perform stack ensembling on dataset with sklearn models.

# Arguments:
    models:  list, a list of models to be used to create meta-features
    X:       dataframe, the trainset features
    y:       dataframe, the trainset labels
    Xtest:   dataframe, the testset features
    splits:  int, the number of splits for trainset CV
    verbose: bool, true when print outputs are desired
    
# Returns:
    X:       dataframe, new trainset with meta-features
    Xtest:   dataframe, new testset with meta-features
'''

def stack_ensemble(models, x_train, y_train, x_test, splits=5, verbose=True):
        
    # join data and meta features
    def _join(x, preds):
        meta = pd.DataFrame(columns = [col for col in preds.keys()])
        for model, y_hat in preds.items():
            meta[model] = pd.Series(y_hat)
        return pd.concat([x, meta], axis=1)
    
    # init variables
    kf = KFold(n_splits = splits)
    preds_train = {}
    preds_test = {}

    # iterate over all inserted models
    for name, model in [(str(type(model)).split('.')[-1][:-2], model) for model in models]:
        if verbose: print('Getting predictions from {}..'.format(name))
        preds_train[name] = []

        # train and predict
        for train, test in kf.split(x_train):
            model.fit(x_train.iloc[train], y_train[train])
            preds_train[name].extend(model.predict(x_train.iloc[test]))
        preds_test[name] = model.predict(x_test)
        
    # return trainset and testset with meta features
    return _join(x_train, preds_train), _join(x_test, preds_test)
    