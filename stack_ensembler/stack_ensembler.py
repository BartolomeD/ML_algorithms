import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import KFold

''' Function to perform stack ensembling on dataset with sklearn models.

# Arguments:
    models:    list, a list of models to be used to create meta-features
    x_train:   ndarray, the trainset features
    y_train:   array, the trainset labels
    x_test:    ndarray, the testset features
    splits:    int, the number of splits for trainset CV
    proba:     bool, outputs probability predictions when true
    orig_data: bool, original features are kept when true
    verbose:   bool, prints progression when true
    
# Returns:
    x_train:   ndarray, new trainset with meta-features
    x_test:    ndarray, new testset with meta-features
'''

def StackEnsemble(models, x_train, y_train, x_test, splits=5, proba=True, orig_data=True, verbose=True):
    
    # join data and meta features
    def _join(x, preds):
        if not orig_data: 
            x = np.array([]).reshape(len(x), 0)
        for pred in preds.values():
            if proba:
                x = np.concatenate([x, np.array(pred).reshape(-1, len(set(y_train)))], axis=1)
            else:
                x = np.concatenate([x, np.array(pred).reshape(-1, 1)], axis=1)
        return x

    # init variables
    kf = KFold(n_splits = splits)
    preds_train = OrderedDict()
    preds_test = OrderedDict()

    # iterate over all inserted models
    for name, model in [(str(type(model)).split('.')[-1][:-2], model) for model in models]:
        if verbose: print('Getting predictions from {}..'.format(name))

        # train & predict with probability predictions
        if proba:
            preds_train[name] = np.array([]).reshape(0, len(set(y_train)))
            for train, test in kf.split(x_train):
                model.fit(x_train[train], y_train[train])
                preds_train[name] = np.vstack([preds_train[name], np.array(model.predict_proba(x_train[test]))])
            preds_test[name] = model.predict_proba(x_test)     

        # train & predict without probability predictions
        else:
            preds_train[name] = []
            for train, test in kf.split(x_train):
                model.fit(x_train[train], y_train[train])
                preds_train[name].extend(model.predict(x_train[test]))
            preds_test[name] = model.predict(x_test)

    return _join(x_train, preds_train), _join(x_test, preds_test)