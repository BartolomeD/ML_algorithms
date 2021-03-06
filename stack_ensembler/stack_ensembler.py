import numpy as np
from collections import OrderedDict
from sklearn.model_selection import KFold

''' Function to perform stack ensembling on dataset with sklearn models.

# Arguments:
    models:    list, a list of models to be used to create meta-features
    x_train:   ndarray, the trainset features
    y_train:   array, the trainset labels
    x_test:    ndarray, the testset features
    n_folds:   int, the number of splits for trainset CV
    prob:      bool, outputs probability predictions when true
    orig_data: bool, returns original features when true
    verbose:   bool, prints progression when true
    
# Returns:
    x_train:   ndarray, new trainset with meta-features
    x_test:    ndarray, new testset with meta-features
'''

def StackEnsemble(models, x_train, y_train, x_test, n_folds=5, prob=True, orig_data=True, verbose=True):
    
    def _join(x, preds): 
        x = x if orig_data else np.array([]).reshape(len(x), 0) 
        for pred in preds.values():
            x = np.concatenate([x, np.array(pred).reshape(-1, len(set(y_train)) if prob else 1)], axis=1)
        return x
    
    def _predict(x):
        return model.predict_proba(x) if prob else model.predict(x)

    kf = KFold(n_splits = n_folds)
    preds_train = OrderedDict()
    preds_test = OrderedDict()

    for name, model in [(str(type(m)).split('.')[-1][:-2], m) for m in models]:
        if verbose: print('Getting predictions from {}..'.format(name))

        preds_train[name] = []
        for train, test in kf.split(x_train):
            model.fit(x_train[train], y_train[train])
            preds_train[name].append(_predict(x_train[test]))
        preds_test[name] = _predict(x_test)    

    return _join(x_train, preds_train), _join(x_test, preds_test)
