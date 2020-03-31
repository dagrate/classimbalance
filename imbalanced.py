# -*- coding: utf-8 -*-
"""imbalanced.ipynb
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_curve,
                             auc, roc_auc_score)
from sklearn.utils import Bunch
import pickle as pkl

# pandas options
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


def plotfeatures(estimator, feat, top=20):
    coef = estimator.feature_importances_
    featimp = pd.Series(coef, feat).sort_values(ascending=False)
    plt.figure(figsize=(12,7))
    featimp[:top].plot(kind = 'barh')
    plt.xlabel('Feature Importance Score')
    plt.show


# we define a randomforest as the estimator
estimator = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=42, splitter='best')

df = pd.DataFrame(
    load_breast_cancer().data,
    columns=load_breast_cancer().feature_names)
df['target'] = load_breast_cancer().target

# without correction
print('\nClass Imbalance')
print('Negative Class Count', sum(df.target == 0))
print('Positive Class Count', sum(df.target == 1))

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['target']),
    df.target,
    test_size=0.3,
    shuffle=True
)

estimator.fit(X_train, y_train)
ypred = estimator.predict(X_test)
print('\nroc_auc_score:', np.round(roc_auc_score(y_test, ypred) ,3))
plotfeatures(estimator, df.drop(columns=['target']).columns, top=20)

# with extreme imbalance 2 per 212
dfpositive = df[df.target==1].iloc[:2]
dfnegative = df[df.target==0]
dfexp = dfnegative.append(dfpositive)

print('\nClass Imbalance')
print('Negative Class Count', sum(dfexp.target == 0))
print('Positive Class Count', sum(dfexp.target == 1))

X_train, X_test, y_train, y_test = train_test_split(
    dfexp.drop(columns=['target']),
    dfexp.target,
    test_size=0.3,
    shuffle=True
)

estimator.fit(X_train, y_train)
ypred = estimator.predict(X_test)
print('\nroc_auc_score:', np.round(roc_auc_score(y_test, ypred) ,3))
plotfeatures(estimator, df.drop(columns=['target']).columns, top=20)

# with small imbalance 125 per 212
dfpositive = df[df.target==1].iloc[:125]
dfnegative = df[df.target==0]
dfexp = dfnegative.append(dfpositive)

print('\nClass Imbalance')
print('Negative Class Count', sum(dfexp.target == 0))
print('Positive Class Count', sum(dfexp.target == 1))

X_train, X_test, y_train, y_test = train_test_split(
    dfexp.drop(columns=['target']),
    dfexp.target,
    test_size=0.3,
    shuffle=True
)

estimator.fit(X_train, y_train)
ypred = estimator.predict(X_test)
print('\nroc_auc_score:', np.round(roc_auc_score(y_test, ypred), 3))
plotfeatures(estimator, df.drop(columns=['target']).columns, top=20)
