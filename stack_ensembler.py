import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

''' Function to perform stack ensembling on arbitrary dataset with sklearn models.

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

def stackEnsemble(models, X, y, Xtest, splits, verbose):

    # assert correct data-types 
    assert type(models) == list
    assert type(splits) == int
    assert type(verbose) == bool

    # init variables
    kf = KFold(n_splits = splits)
    predsTR = {}
    predsTE = {}

    # iterate over all inserted models
    for n, model in enumerate(models):
        if verbose: print('Using model %d to make predictions..' % (n+1))

        # prepare split for predictions
        predsTR['model'+str(n+1)] = []
        for i, (train, test) in enumerate(kf.split(X)):
            if verbose: print('..on split %d' % (i+1))

            # fit on split and predict
            model.fit(X.iloc[train], y[train])
            predsTR['model'+str(n+1)].append(list(model.predict(X.iloc[test])))

        # predict on testset
        predsTE['model'+str(n+1)] = list(model.predict(Xtest))
    
    # combine trainset predictions in dataframe, join with trainset
    meta_feats = pd.DataFrame(columns = [col for col in predsTR.keys()])
    for model in predsTR.keys():
        meta_feats[model] = np.array([item for lst in predsTR[model] for item in lst])
    X = pd.concat([X, meta_feats], axis=1)

    # combine testset predictions in dataframe, join with testset
    meta_feats = pd.DataFrame(columns = [col for col in predsTE.keys()])
    for model in predsTE.keys():
        meta_feats[model] = np.array(predsTE[model])
    Xtest = pd.concat([Xtest, meta_feats], axis=1)

    # return trainset and testset with metafeatures
    return X, Xtest
    