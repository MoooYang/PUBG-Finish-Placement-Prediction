# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pubg_train = pd.read_csv('../input/train_V2.csv')
pubg_train_X = pubg_train.iloc[:,0:pubg_train.shape[1]-1]
pubg_train_Y = pubg_train.iloc[:,pubg_train.shape[1]-1]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(pubg_train_X['matchType'])
pubg_train_X['matchType'] = le.transform(pubg_train_X['matchType'])

from sklearn.preprocessing import Imputer
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
pubg_train_Y = pubg_train_Y.values.reshape(len(pubg_train_Y), 1)
mean_imputer = mean_imputer.fit(pubg_train_Y)
pubg_train_Y = mean_imputer.transform(pubg_train_Y)

def rank_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    agg = agg.groupby('matchId')[features].rank(pct=True)
    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])
 
def median_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop and '_mean_rank' not in col]
    agg = df.groupby(['matchId', 'groupId'])[features].median()
    return df.merge(agg, suffixes=['', '_median'], how='left', on=['matchId', 'groupId'])
    
pubg_train_X = rank_by_team(pubg_train_X)
pubg_train_X = median_by_team(pubg_train_X)
pubg_train_X = pubg_train_X.drop(columns=['Id','groupId','matchId'])

from sklearn.ensemble import RandomForestRegressor
m1 = RandomForestRegressor(n_estimators=40,min_samples_leaf=10,max_features='sqrt',n_jobs=-1)

m1.fit(pubg_train_X, pubg_train_Y)

pubg_test = pd.read_csv('../input/test_V2.csv')
test_group_ids = pubg_test['groupId']
pubg_test['matchType'] = le.transform(pubg_test['matchType'])
pubg_test = rank_by_team(pubg_test)
pubg_test = median_by_team(pubg_test)

pubg_test_X = pubg_test.drop(columns=['Id','groupId','matchId'])
y_pred = m1.predict(pubg_test_X)

pubg_test_X['groupId'] = test_group_ids
pubg_test_X['winPlacePerc'] = y_pred
pubg_test_X['winPlacePerc'] = pubg_test_X.groupby('groupId')['winPlacePerc'].transform('mean')

y_avg_pred = pubg_test_X['winPlacePerc']
ids = pubg_test.iloc[:,0]
submission = pd.DataFrame({'id': ids, 'winPlacePerc': y_avg_pred})
submission.to_csv('RFSoln.csv',index=False)