# # Introduction
# # This kernel has been created by the Information Systems Lab at the University of Macedonia, Greece for the needs of the elective course Special Topics of Information Systems I at the Business Administration department of the University of Macedonia, Greece.



# Objective
# The objective of this Kernel is to predict the future behaviour (which products they will buy) based on the features that we have created in our EDA Notebooks.

# By the time you finish this example, you will be able to:

# Describe the steps of creating a predictive analytics model Use Python to manipulate ready features Use Python to create, combine, and delete data tables Use XGBoost to create a predictive model Apply the predictive model in order to make a prediction

# Problem Definition
# The data that Instacart opened up include orders of 200,000 Instacart users with each user having between 4 and 100 orders. Instacart indicates each order in the data as prior, train or test. Prior orders describe the past behaviour of a user while train and test orders regard the future behaviour that we need to predict. As a result, we want to predict which previously purchased products (prior orders) will be in a user’s next order (train and test orders). For the train orders Instacart reveals the results (i.e. the ordered products) while for the test orders we do not have this piece of information. Moreover, the future order of each user can be either train or test meaning that each user will be either a train or a test user. The setting of the Instacart problem is described in the figure below.



# Each user has purchased various products during their prior orders. Moreover, for each user we know the order_id of their future order. The goal is to predict which of these products will be in a user's future order. This is a classification problem because we need to predict whether each pair of user and product is a reorder or not. This is indicated by the value of the reordered variable, i.e. reordered=1 or reordered=0 (see figure below).



# As a result we need to come up and calculate various predictor variables (X) that will describe the characteristics of a product and the behaviour of a user regarding one or multiple products. We will do so by analysing the prior orders of the dataset. We will then use the train users to create a predictive model and the test users to make our actual prediction. As a result we create a table as the following one and we train an algorithm based on predictor variables (X) and response variable (Y).



# Method
# Our method includes the following steps:

# Import the ready features from EDA notebooks and reshape data: This step includes loading pkl (pickle) files into pandas DataFrames.
# Create the test and train DataFrames: In this step we create two distinct DataFrames that will be used in the creation and the use of the predictive model.
# Create the preditive model: In this step we employ XGBoost algorithm to create the predictive model through the train dataset.
# Apply the model: This step includes applying the model to predict the 'reordered' variable for the test dataset.
# Business Insights
# Python Skills
# Packages
# 1. Import packages and data
# We import the time package to calculate the execution time of our code.
# We import the gc package to free-up reserved memory by Python.

'''
# import the time package to calculate the execution time of the kernel
import time
#set on start variable the current time
start = time.time()
# run your code and create a new variable with the time
end = time.time()
#substract the start time from end time to calculate the execution time
elapsed = end - start'''

import time
start = time.time()

import pandas as pd # dataframes
import numpy as np # algebra & calculus

import os
print(os.listdir("../input"))
print(os.listdir("../input/instacart-market-basket-analysis"))
print(os.listdir("../input/ml-instacart-f1-0-38-part-one-features"))


import gc #clean-up memory

# Now we load the pickle file that contains the prd table with several features that we have created in our EDA notebooks

uxp = pd.read_pickle('../input/ml-instacart-f1-0-38-part-one-features/uxp.pkl')
#uxp = uxp.iloc[0:150000]
uxp.head()

'''
#### Remove triple quotes to trim your dataset and experiment with your data
### COMMANDS FOR CODING TESTING - Get 10% of users 
uxp = uxp.loc[uxp.user_id.isin(uxp.user_id.drop_duplicates().sample(frac=0.01, random_state=25))] 
uxp.head()
'''
# In addition, we load the original .csv files from Instacart that contains the orders and the products that have ben purchased

orders = pd.read_csv('../input/instacart-market-basket-analysis/orders.csv' )
order_products_train = pd.read_csv('../input/instacart-market-basket-analysis/order_products__train.csv')

#products = pd.read_csv('../input/instacart-market-basket-analysis/products.csv')
# We keep only the train and test orders, excluding all the prior orders (these that we used to create our features)

orders_last = orders[(orders.eval_set=='train') | (orders.eval_set=='test') ]
uxp = uxp.merge(orders_last, on='user_id', how='left')
uxp.head(10)

uxp_train = uxp[uxp.eval_set=='train']

uxp_train = uxp_train.merge(order_products_train, on=['product_id', 'order_id'], how='left' )

uxp_train = uxp_train.drop(['order_id','eval_set', 'add_to_cart_order'], axis=1)
uxp_train = uxp_train.fillna(0)
uxp_train.head(20)

uxp_test = uxp[uxp.eval_set=='test']
uxp_test = uxp_test.drop(['eval_set', 'order_id'], axis=1)
uxp_test = uxp_test.fillna(0)
uxp_test.head(20)

del uxp
del orders_last
gc.collect()
uxp_train = uxp_train.set_index(['user_id', 'product_id'])

'''#BALANCE REORDERED ROWS
uxp_train_bal = uxp_train.copy()
uxp_train_bal = uxp_train_bal[uxp_train_bal.reordered==0].sample(n=uxp_train_bal[uxp_train_bal.reordered==1].shape[0])
uxp_train_bal = pd.concat([uxp_train_bal, uxp_train[uxp_train.reordered==1]])
uxp_train_bal = uxp_train_bal.sample(frac=1)
uxp_train = uxp_train_bal.copy()
print(uxp_train.reordered.value_counts())
del uxp_train_bal
gc.collect()'''


uxp_test = uxp_test.set_index(['user_id', 'product_id'])
import xgboost
from sklearn.model_selection import train_test_split
uxp_train.loc[:, 'reordered'] = uxp_train.reordered.fillna(0)


# subsample
X_train, X_val, y_train, y_val = train_test_split(uxp_train.drop('reordered', axis=1), uxp_train.reordered,
                                                    test_size=0.2, random_state=42)

'''del uxp_train'''
gc.collect()

d_train = xgboost.DMatrix(X_train, y_train)
param = {'max_depth':10, 
         'eta':0.02,
         'colsample_bytree':0.4,
         'subsample':0.75,
         'silent':1,
         'nthread':27,
         'eval_metric':'logloss',
         'binary':'logistic',
         'tree_method':'hist'
}

watchlist= [(d_train, "train")]
bst = xgboost.train(params=param, dtrain=d_train, num_boost_round=1000, evals=watchlist, early_stopping_rounds=40, verbose_eval=5)
xgboost.plot_importance(bst)

del [X_train, X_val, y_train, y_val]
gc.collect()

d_test = xgboost.DMatrix(uxp_test)

uxp_test = uxp_test.reset_index()
uxp_test = uxp_test[['product_id', 'user_id']]

uxp_test["reordered"] = bst.predict(d_test)

del bst
orders_test = orders[orders.eval_set=='test']
uxp_test = uxp_test.merge(orders_test[["user_id", "order_id"]], on='user_id', how='left').drop('user_id', axis=1)
uxp_test.columns = ['product_id', 'prediction', 'order_id']
uxp_test.product_id = uxp_test.product_id.astype(int)
uxp_test.order_id = uxp_test.order_id.astype(int)
uxp_test.head()

del orders
del orders_test
gc.collect()

import numpy as np
from operator import itemgetter

class F1Optimizer():
    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)


def get_best_prediction(items, preds, pNone=None):
#    print("Maximize F1-Expectation")
#    print("=" * 23)
    items_preds = sorted(list(zip(items, preds)), key=itemgetter(1), reverse=True)
    P = [p for i,p in items_preds]
    L = [i for i,p in items_preds]
    
    opt = F1Optimizer.maximize_expectation(P)
    best_prediction = []
    best_prediction += (L[:opt[0]])
    if best_prediction == []:
        best_prediction = ['None']
            
#    print("Prediction {} yields best E[F1] of {}\n".format(best_prediction, f1_max))
    return ' '.join(list(map(str,best_prediction)))
import pandas as pd
import multiprocessing as mp
import time

#==============================================================================
# load
#==============================================================================
sub_item = uxp_test.groupby(['order_id','product_id']).prediction.mean().reset_index()
sub = sub_item.groupby('order_id').product_id.apply(list).to_frame()
sub['yhat'] = sub_item.groupby('order_id').prediction.apply(list)
sub.reset_index(inplace=True)

del uxp_test, sub_item
gc.collect()

def multi(i):
    if i%1000==0:
        print('{:.3f} min'.format((time.time()-st_time)/60))
    items = sub.loc[i,'product_id']
    preds = sub.loc[i,'yhat']
    ret = get_best_prediction(items, preds)
    return ret

st_time = time.time()
pool = mp.Pool(4)
callback = pool.map(multi, range(sub.shape[0]))

sub['products'] = callback
sub.head()

sub.reset_index(inplace=True)
sub = sub[['order_id', 'products']]
'''d = dict()
for row in uxp_test.itertuples():
    if row.reordered == 1:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in uxp_test.order_id:
    if order not in d:
        d[order] = 'None'
        
gc.collect()'''

'''sub = pd.DataFrame.from_dict(d, orient='index')
sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
'''

print(sub.shape[0])
print(sub.shape[0]==75000)

sub.to_csv('sub.csv', index=False)

print(os.listdir("../working/"))
submission = pd.read_csv("../working/sub.csv")
submission.head()
submission.shape[0]
end = time.time()
elapsed = end - start
elapsed


# https://www.kaggle.com/code/kokovidis/ml-instacart-f1-0-38-part-two-xgboost-f1-max
