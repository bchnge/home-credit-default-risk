from src import proj_utils
from src import utils
from src.classes import Model, Dataset, Tuner, ModelCollection

from lightgbm.sklearn import LGBMClassifier

# Load data
df_train = proj_utils.load_data(train = True,
                     supp_dict = {'previous_application.csv.zip' : 'max',
                                  'credit_card_balance.csv.zip' : 'mean',
                                  'installments_payments.csv.zip' : 'min',
                                  'POS_CASH_balance.csv.zip' : 'mean',
                                  'bureau.csv.zip' : 'max'
                                 })

df_test = proj_utils.load_data(train = False,
                     supp_dict = {'previous_application.csv.zip' : 'max',
                                  'credit_card_balance.csv.zip' : 'mean',
                                  'installments_payments.csv.zip' : 'min',
                                  'POS_CASH_balance.csv.zip' : 'mean',
                                  'bureau.csv.zip' : 'max'
                                 })

data = Dataset(df_train, df_test, 'TARGET')


# Clean and transform data
data.preprocess()

# Determine initial feature importances
data.ae_train_model(model = LGBMClassifier())

# Auto-discover ratios weighted by feature importance
data.autoengineer_ratios()

#############################################################
models = {}
for m in M:
    # Define a model
    models[m] = {}

    # Tune parameters
    tuner = Tuner(m, data.X_train, data.y_train)
    for k in range(1,5):
        tuner.tune(kappa = k, pbounds = {}, n_iters = 10)
    

# Compare and stack models
mc = ModelCollection(models)

                                 