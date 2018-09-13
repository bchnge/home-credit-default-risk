from src.utils import category_to_numeric, get_design_matrix_lbl
from numpy.random import choice
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from mlxtend.feature_selection import ColumnSelector
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from mlxtend.classifier import StackingClassifier

class Model:
    '''
    Model serves as a container for documenting, defining, training, and validating predictive models.
    '''
    def __init__(self, clf = None, name = None, desc = None):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier

        self.name = name
        self.desc = desc

        if clf is None:
            self.clf = RandomForestClassifier()
        else:
            self.clf = clf

        self.cv = StratifiedKFold(n_splits=5, random_state=123)

    def set_model(self, clf):
        self.clf = clf
        
    def train(self, X, y):
        self.clf.fit(X, y)
        
    def validate_model(self, X, y):
        from sklearn.model_selection import cross_val_score

        if hasattr(self, 'clf') is False:
            print('Stop. You must define a model before validating.')
        else:
            self.validation_scores = cross_val_score(self.clf, X, y, scoring = 'roc_auc', cv = self.cv)

class Dataset:
    '''
    Datasets contain methods for maintaining train/test matrices, preprocessing, feature elimination, and automatic feature engineering. Transformers can be saved as pipeline models.
    '''
    def __init__(self, train_data, test_data, ylabel):
        # Store engineered features
        self.ae_discovery_ratios = []
        
        # CLEAN DATA
        ## KEEP MUTUAL COLUMNS
        train_features = train_data.columns
        test_features = test_data.columns
        features = list(set(train_features) & set(test_features))
        
        train_data = train_data.loc[:, features + [ylabel]]
        test_data = test_data.loc[:, features]

        columns_object = list(test_data.columns[test_data.dtypes == object])
        columns_numeric = list(set(features) - set(columns_object))

        ## DROP NONVARYING COLUMNS
        ### Numeric columns
        std_threshold = 0 
        col_std = train_data.loc[:, columns_numeric].std(axis = 0)
        nonvariant_numeric = list(col_std[col_std <= std_threshold].index)

        ### Object columns
        uv_threshold = 1
        col_uv = train_data.loc[:, columns_object].apply(lambda x: len(x.unique()), axis = 0)
        nonvariant_object = list(col_uv[col_uv <= uv_threshold].index)

        train_data = train_data.drop(nonvariant_object + nonvariant_numeric, axis = 1)
        test_data = test_data.drop(nonvariant_object + nonvariant_numeric, axis = 1)

        ## HARMONIZE LABELS FOR OBJECTS BETWEEN TRAIN AND TEST SETS
        for c in columns_object:
            test_labels = set(test_data[c].unique())
            train_labels = set(train_data[c].unique())
            compl_labels = list((train_labels - test_labels).union(test_labels - train_labels))
            
            if len(compl_labels) > 0:
                train_data[c] = train_data[c].replace(compl_labels, np.nan)
                test_data[c] = test_data[c].replace(compl_labels, np.nan)
            
        ## BUILD FEATURE MATRIX AND TARGET DATAFRAMES    
        print('...creating training matrix')
        self.X_train, self.y_train = get_design_matrix_lbl(data = train_data, y_label = ylabel, features = None, convert_categorical = True)

        print('...creating test matrix')
        self.X_test = get_design_matrix_lbl(data = test_data, y_label = None, features = None, convert_categorical = True)      

    def ae_train_model(self, model):
        model.fit(self.X_train, self.y_train)        
        self.ae_feature_importances_dict = dict(zip(self.X_train.columns, model.feature_importances_))
        self.ae_feature_importances = model.feature_importances_        
    
    def autoengineer_ratios(self, ae_params = None, n_iter = 1000):
        if ae_params is None:
            ae_params = {'boosting_type': 'gbdt',
                    'max_depth' : -1,
                    'objective': 'binary',
                    'learning_rate': 0.0212,
                    'reg_alpha': 0.8,
                    'reg_lambda': 0.4,
                    'subsample': 1,
                    'feature_fraction': 0.3,
                    'device_type': 'gpu',
                    'metric' : 'auc',
                    'random_state': 123,
                    'n_estimators': 300, 
                    'num_leaves': 40, 
                    'max_bin': 255,
                    'min_data_in_leaf': 2400,
                    'min_data_in_bin': 5}
        
        def _fn_column_selector(X, k):
            ''' 
                select up to kth column
            '''
            return X[:,:k]

        ColumnSelector = FunctionTransformer(_fn_column_selector, validate = False)
        importance_weights = self.ae_feature_importances / self.ae_feature_importances.sum()        
        kfold = StratifiedKFold(n_splits=5, random_state=123)
        model = Pipeline([('selector', ColumnSelector),
                  ('clf', LGBMClassifier(**ae_params))])

        for i in range(n_iter):
            random_vars = list(choice(self.X_train.columns, 
                                                size = 2, p = importance_weights, 
                                                replace= False))

            X_tmp = self.X_train.loc[:, random_vars ]
            X_tmp['_DIV_'.join(random_vars)] = X_tmp.iloc[:,0] / (X_tmp.iloc[:,1] + 1)

            gs = GridSearchCV(estimator = model,
                              param_grid = {'selector__kw_args': [{'k':2},{'k':3}]},
                              scoring = 'roc_auc',
                              cv = kfold)
            gs.fit(X_tmp.values, self.y_train)
            perf_1, perf_2 = gs.cv_results_.get('mean_test_score')
            if perf_2 > perf_1:
                self.ae_discovery_ratios.append((random_vars[0], random_vars[1], perf_2/perf_1))
    def ae_augment(self, X):
        for ae in self.ae_discovery_ratios:
            X['_DIV_'.join([ae[0], ae[1]])] = X.iloc[:,ae[0]] / X.iloc[:, ae[1]]
        return(X)
        

class Tuner:
    def __init__(self, dataset, model):
        self.model = model
        self.X = dataset.X_train
        self.y = dataset.y_train
        self.ae_definitions = dataset.discovery_ratios

    def tune(self, kappa, pbounds, n_iters):
        from bayes_opt import BayesianOptimization

        integer_params = [p for p in pbounds.keys() if type(pbounds.get(p)[0]) == int and type(pbounds.get(p)[1])]

        print('params detected as integers: \n')
        print(integer_params)


        def _fn(**kwargs):
            # Each tuning requires a custom function to measure success

            from sklearn.model_selection import cross_val_score
            bo_model = self.clf
            for p in integer_params:
                if p in kwargs.keys():
                    kwargs[p] = int(kwargs.get(p))

            bo_model.set_params(kwargs)

            score = cross_val_score(self.clf, self.X, self.y, scoring = 'roc_auc', cv = 5)

            return(score.mean())

        self.bo = BayesianOptimization(f = _fn, pbounds = pbounds,random_state = 123)
        self.bo.maximize(kappa = kappa, n_iters = n_iters)

class ModelCollection():
    ''' A collection of Models. Contains methods for ensemble classification and comparing models
    '''
    def __init__(self, models_instructions):
        '''
        models_instructions should be a list of tuples [(Model, list_of_features)]
        '''
        from mlxtend.classifier import StackingClassifier
        from mlxtend.feature_selection import ColumnSelector
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        self.cv = StratifiedKFold(n_splits=5, random_state=123)
        self.models_instructions = models_instructions
        models = [Pipeline([('ColumnSelect', ColumnSelector(v[1])), ('Model', v[0].clf)]) for i,v in enumerate(models_instructions)]
        self.models = models
        self.clf_stack = Model(clf = StackingClassifier(classifiers = models, meta_classifier = LogisticRegression()), name = 'Stacked ensemble')
        
    def validate_model(self, X, y):
        from sklearn.model_selection import cross_validate

        if hasattr(self, 'clf_stack') is False:
            print('Stop. You must define a model before validating.')
        else:
            self.validation_scores_stack = cross_validate(self.clf_stack.clf, X, y, scoring = 'roc_auc', cv = self.cv)

            #self.validation_scores_individual = [(i, cross_validate(clf, X, y, scoring = 'roc_auc', cv = self.cv)) for i, clf in enumerate(self.models)]
            
    def compare_models(self):
            stack_df = pd.DataFrame(self.validation_scores_stack)
            individual_df = pd.concat([pd.DataFrame(score) for score in self.validation_scores_individual])
            result_df = pd.concat([stack_df, individual_df])
            return(result_df)
