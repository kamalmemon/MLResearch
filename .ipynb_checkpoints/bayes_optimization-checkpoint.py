import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import *
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from logger import *

warnings.filterwarnings('ignore')
sns.set()

df = pd.read_csv("data/Train_psolI3n.csv")
df.set_index("Email_ID", inplace=True)

df_org = df.copy()
df = df.dropna()

df['Customer_Location'] = df['Customer_Location'].astype("category").cat.codes
X = df.drop(['Email_Status'], axis=1)
Y = df['Email_Status']

## Oversampling Minority classes
from imblearn.over_sampling import SMOTE 
df_m = df.copy()
df_m = df_m[(df_m.Email_Status == 1) | (df_m.Email_Status == 2)]
X_m = df_m.drop(['Email_Status'], axis=1)
Y_m = df_m['Email_Status']
X = X.loc[Y[Y==0].index]
Y = Y[Y==0]
sm = SMOTE(random_state=np.random.randint(0, 100))
X_os_m , Y_os_m = sm .fit_resample(X_m, Y_m)
X_os = pd.concat([X, pd.DataFrame(X_os_m, columns= X.columns)], axis=0)
Y_os = pd.concat([Y, pd.Series(Y_os_m)], axis=0)

X_train, X_test, y_train, y_test = train_test_split(X_os, Y_os, 
                                                    test_size=0.3,
                                                    random_state=71, stratify=Y_os)

def hyperopt_train_test(params):
    t = params['type']
    del params['type']
    if t == 'naive_bayes':
        clf = BernoulliNB(**params)
    elif t == 'svm':
        clf = SVC(**params)
    elif t == 'randomforest':
        clf = RandomForestClassifier(**params)
    elif t == 'knn':
        clf = KNeighborsClassifier(**params)
    else:
        return 0
    return cross_val_score(clf, X_train, y_train, cv=3, n_jobs=-1).mean()

space = hp.choice('classifier_type', [
    {
        'type': 'naive_bayes',
        'alpha': hp.uniform('alpha', 0.0, 2.0)
    },
    {
        'type': 'svm',
        'C': hp.uniform('C', 0, 10.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20.0)
    },
    {
        'type': 'randomforest',
        'max_depth': hp.choice('max_depth', range(1,20)),
        'max_features': hp.choice('max_features', range(1,5)),
        'n_estimators': hp.choice('n_estimators', range(1,20)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        'class_weight': 'balanced'
    },
    {
        'type': 'knn',
        'n_neighbors': hp.choice('knn_n_neighbors', range(1,30))
    }
])
count = 0
best = 0
acc_hist = []
logger.info('Starting optimization...')
def f(params):
    global best, count
    count += 1
    acc = hyperopt_train_test(params.copy())
    acc_hist.append([acc, params])
    if acc > best:
        logger.info('new best:' + str(acc) + ' using: ' + str(params['type']))
        print ('new best:', acc, 'using', params['type'])
        best = acc
    #if count % 50 == 0:
    logger.info('iters: ' + str(count) + ', acc: ' + str(acc) + ' using: ' + str(params))
    print ('iters:', count, ', acc:', acc, 'using', params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=150, trials=trials)
logger.info('best: ' + str(best))
print('best: ', best)

pickle.dump(best, open('pickels/hyperopt_best.p', 'wb'))
pickle.dump(acc_hist, open('pickels/hyperopt_hist.p', 'wb'))

