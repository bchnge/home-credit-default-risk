'''
# utils.py
## contains generic utilities for data cleaning and modeling
'''

import pandas as pd
import os
import random
import numpy as np
from pprint import pprint
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc, make_scorer
from matplotlib import pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import requests
import json


def notify_ifttt(data = '', config_path = '/projects/ifttt/code_complete.json'):   
    with open(config_path, 'r') as f:
        config_jsn = json.load(f)
    ifttt_notification_url = 'https://maker.ifttt.com/trigger/' + config_jsn.get('name') + '/with/key/' + config_jsn.get('ifttt_key')
    payload = {'value1': data}
    requests.post(ifttt_notification_url, data=payload)
    
def get_feature_grouping(X):
    pca = PCA().fit(X.dropna())
    
    num_groups, num_features = pca.components_.shape
    print(str(num_groups) + ' groups...')
    print(str(num_features) + ' features...')
    
    # Calculate factor loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    abs_loadings = np.abs(loadings)
    variable_groupings = abs_loadings.argmax(axis = 1)

    groupings = {}
    for g in range(variable_groupings.min(), variable_groupings.max()):
        groupings[g] = [X.columns[i] for i,x in enumerate(variable_groupings) if x == g]
    return(groupings)

def autoencode_dataframe(df):
    ''' given a dataframe with numerics and objects, convert objects into dummies'''
    objects = df.columns[(df.dtypes == object)]        
    # Convert categorical to dummy
    for o in objects:
        df = pd.concat([df, pd.get_dummies(df[o])], axis = 1)
        del df[o]
    return(df)

def combine_levels(o, covariates, min_rt = 0.05, return_assignment = False):
    ''' given a Series categorical o and covariate df, produce an alternate series p that is reduced '''
    val_cts = o.value_counts()
    val_rts = val_cts / float(len(o))
    # print(val_rts)
    # print('\n')
    vals = list(val_cts.index)
    core_levels = [vals[i] for i,x in enumerate(val_rts) if x >= min_rt]
    reduce_levels = [vals[i] for i,x in enumerate(val_rts) if x < min_rt]
    # print('Assign these levels: ' + '\n')
    # print(reduce_levels)
    # print(' \n to these levels: \n')
    # print(core_levels)
    df = autoencode_dataframe(covariates)
    df['x'] = o
    # Get the average value across all covariates for each group
    distributions = df.groupby(['x']).mean().reset_index()
    labs = distributions.x
    distributions = distributions.drop('x', axis = 1)
    cs = cosine_similarity(distributions)
    # print(cs)
    similarities = pd.DataFrame(cs, columns = labs, index = labs) \
        .loc[reduce_levels, core_levels]
    assignment = similarities.apply(lambda row: row.idxmax(), axis = 1).to_dict()
    # print(assignment)
    z = o.replace(assignment)
    if return_assignment:
        return(z, assignment)
    else:
        return(z)
    
def clean_numeric(x):
    is_missing = pd.isnull(x)
    num_missing = is_missing.sum()
    if num_missing > 1:
        #z = x.fillna(x.mean())
        z = x.fillna(0)
        cleaned = pd.DataFrame({x.name+'_IMP' : z, x.name + '_NAFLAG' : is_missing})
    else:
        cleaned = pd.DataFrame({x.name+'_ORI' : x.fillna(0)})        
    return(cleaned)
        
def enhance_numeric(df):
    ''' given a numeric Series, produce an alternate version with a missing encoding and imputation'''
    numerics = df.columns[df.dtypes != object]
    for n in numerics:
        df = pd.concat([df, clean_numeric(df[n])], axis = 1)
        del df[n]
    return(df)
    
def category_to_numeric(x):
    values = sorted(list(x.astype(str).unique()))
    le = LabelEncoder().fit(values + ['nan'])
    result = le.transform(x.values.astype(str))
    return(result)

def get_design_matrix_lbl(data, y_label = None, features = None, convert_categorical = True):
    if features is None:
        if y_label is None:
            features = list(data.columns)
            data = data[features]
        else:
            features = list(set(data.columns) - set([y_label]))
            data = data[features + [y_label]]
    else:
        if y_label is None:
            data = data[features]
        else:
            data = data[features + [y_label]]

    if convert_categorical:
        # convert categorical column to labeled values
        objects = data.columns[(data.dtypes == object)]
        objects = objects[objects.isin(features)]
   
        for o in objects:
            data.loc[:,o] = category_to_numeric(data[o])

    if y_label is None:
        return(data)
    else:
        X = data.drop(y_label, axis = 1)
        y = data[y_label]
        return(X, y)
    
def get_design_matrix_refined(data, y_label, features, train_test_split = False, train_sample = 0.75):
    objects = data.columns[(data.dtypes == object)]
    objects = objects[objects.isin(features)]

    numerics = data.columns[(data.dtypes != object)]
    numerics = numerics[numerics.isin(features)]
    
    data = data[features + ['TARGET']]
    
    # Convert categorical to dummy
    for o in objects:
        data = pd.concat([data, pd.get_dummies(data[o], prefix = o, prefix_sep = '_')], axis = 1)
        del data[o]                

    # Discretize numerics if needed
    for n in numerics:
        if pd.isnull(data[n]).sum() > 0:
            data[n] = data[n]
            disc = pd.qcut(data[n], q = 100, duplicates = 'drop')
            disc_dummies = pd.get_dummies(disc, prefix = n, prefix_sep = '_', dummy_na= True)
            data = pd.concat([data, disc_dummies], axis = 1)
            del data[n]
        
    train_sample_idx = np.array([random.random() <= train_sample for i in range(data.shape[0])])
    
    if train_test_split:
        X_train = data.loc[train_sample_idx].drop(y_label, axis = 1)
        y_train = data.loc[train_sample_idx, y_label]
        X_test = data.loc[~train_sample_idx].drop(y_label, axis = 1)
        y_test = data.loc[~train_sample_idx, y_label]
        return(X_train, y_train, X_test, y_test)
    else:
        X = data.drop(y_label, axis = 1)
        y = data[y_label]
        return(X, y)    
    
def plot_validation_curve(train_scores, validation_scores):
    #Plot validation curve
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(validation_scores, axis=1)
    test_scores_std = np.std(validation_scores, axis=1)

    plt.title("Validation Curve")
    plt.ylabel("Score")
    plt.ylim(0.5, 1.1)
    lw = 2
    plt.legend(loc="best")
    plt.show()

def _feature_reducer(X):
        import pandas as pd
        ''' Additional removal of features '''
        # Force X to an array if dataframe
        if type(X) == pd.core.frame.DataFrame:
            X = X.values
        required_rate = 0.5
        col_missing_rates = np.apply_along_axis(lambda x: (np.isnan(x)).mean(), 
                        arr= X, axis=0)
        keep_cols = [i for i,x in enumerate(col_missing_rates) if x < required_rate]
        return(X[:, keep_cols])

def NanReducer():
    from sklearn.preprocessing import FunctionTransformer
    transformer = FunctionTransformer(func = _feature_reducer, validate = False)
    return(transformer)