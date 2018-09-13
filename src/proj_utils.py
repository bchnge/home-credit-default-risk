'''
# proj_utils.py
## contains project specific utilities
## home-credit-default-risk
'''

import pandas as pd

def agg_supplement(supp_df):
    '''
    agg_supplement calculates aggregate statistics and merges based on unique key (assumes one-to-many relationship)
    '''

    # Numerics
    supp_numeric_mean = supp_df \
            .loc[:,(supp_df.dtypes != object) & (supp_df.columns != 'SK_ID_PREV')] \
            .groupby('SK_ID_CURR') \
            .mean() \
            .reset_index()
    supp_numeric_min = supp_df \
            .loc[:,(supp_df.dtypes != object) & (supp_df.columns != 'SK_ID_PREV')] \
            .groupby('SK_ID_CURR') \
            .min() \
            .reset_index()
    supp_numeric_max = supp_df \
            .loc[:,(supp_df.dtypes != object) & (supp_df.columns != 'SK_ID_PREV')] \
            .groupby('SK_ID_CURR') \
            .max() \
            .reset_index()
    supp = supp_numeric_mean \
        .merge(supp_numeric_max, how = 'inner', on = 'SK_ID_CURR', suffixes = ('_mean', '_max')) \
        .merge(supp_numeric_min, how = 'inner', on = 'SK_ID_CURR', suffixes = ('_max', '_min'))

    # Categorical
    objects = supp_df.columns[(supp_df.dtypes == object) & (supp_df.columns != 'SK_ID_PREV')]
    if len(objects) > 0 :
        tmp = pd.concat([pd.get_dummies(supp_df[o], dummy_na = True, prefix = o) for o in objects], axis = 1)
        tmp['SK_ID_CURR'] = supp_df.SK_ID_CURR
        supp_object_max = tmp \
            .groupby('SK_ID_CURR') \
            .max().reset_index()
        supp_object_min = tmp \
            .groupby('SK_ID_CURR') \
            .min().reset_index()
        supp_object_mean = tmp \
            .groupby('SK_ID_CURR') \
            .mean().reset_index()

        supp = supp \
        .merge(supp_object_mean, how = 'inner', on = 'SK_ID_CURR', suffixes = ('_min', '_mean')) \
        .merge(supp_object_max, how = 'inner', on = 'SK_ID_CURR', suffixes = ('_mean', '_max')) \
        .merge(supp_object_min, how = 'inner', on = 'SK_ID_CURR', suffixes = ('_max', '_min'))
    
    return(supp)

def load_data(data_path = '/vol2/competitions/home-credit-default-risk/', train = True, supp_dict = None):
    ''' 
    Create applications dataframe. 

        Parameters:
        train (True/False): whether or not the applications are in the training set
        supp_dict: dictionary of (supplement_file_name : supplement_aggregation)
    '''
    
    if train:
        print('Loading training applications')
        df = pd.read_csv(data_path + 'application_train.csv.zip')
    else:
        print('Loading test applications')
        df = pd.read_csv(data_path + 'application_test.csv.zip')
    
    if supp_dict:
        supp_idx = 1
        for supp_name in supp_dict.keys():
            print('Loading ' + supp_name)
            supp = agg_supplement(pd.read_csv(data_path + supp_name))                      
            df = df.merge(supp, how = 'left', on = 'SK_ID_CURR', 
                          suffixes = ('', '_'+str(supp_idx)))
            supp_idx += 1
    
    return(df)