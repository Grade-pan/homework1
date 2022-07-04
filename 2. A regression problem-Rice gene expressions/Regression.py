import csv

import numpy as np
from scipy.stats import pearsonr
from sklearn import metrics, linear_model, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
import math

def load_data():
    data = np.load(r'./lch.npy')
    with open('./y.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row[2] for row in reader]
        column1.remove('TPM')
    label = [math.log2(float(x)+1.0) for x in column1]
    label = np.array(label)
    return data,label

data,label = load_data()
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=100)

def pingjia(y_test, y_pred):
    print("r2_score：")
    print(r2_score(y_test, y_pred))
    print("PCC")
    print(pearsonr(y_test, y_pred))


'''
#LASSO寻参 0.00001
alpha_range = np.arange(0.00001,1,0.05)
param_grid = {'alpha':alpha_range}
lasso = linear_model.Lasso()
lasso_search = GridSearchCV(lasso,param_grid,n_jobs=-1)
lasso_search.fit(X_train,y_train)
print(lasso_search.best_score_)
print(lasso_search.best_params_)
print(lasso_search.best_estimator_)
'''

#LASSO模型训练
model = linear_model.Lasso(alpha=1e-05)
y_pred = model.fit(X_train,y_train).predict(X_test)
print("LASSO:")
pingjia(y_test, y_pred)

'''
params={'alpha': [0.01,0.001,0.0001,0.00001]} 0.01
rdg_reg = Ridge()
clf = GridSearchCV(rdg_reg,params,cv=10,verbose = 1, scoring = 'neg_mean_squared_error',n_jobs=-1)
clf.fit(X_train,y_train)
print(clf.best_params_)
'''

#岭回归训练
model = linear_model.Ridge(alpha=0.01)
y_pred = model.fit(X_train,y_train).predict(X_test)
print("岭回归:")
pingjia(y_test, y_pred)

'''
#随机森林寻参
#{'max_features':range(5,76,10)}{"n_estimators": [100,150,200,300]} {'max_depth':range(20,50,3)}
param_grid = {'min_samples_split':range(2,41,3)}
grid_search = GridSearchCV(RandomForestRegressor(n_estimators=200,max_features=315,max_depth=29,min_samples_split=11), param_grid, cv=10,n_jobs=-1,verbose=100)
# 让模型对训练集和结果进行拟合
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
'''

#随机森林训练
rf = RandomForestRegressor(n_estimators=200,max_features=315,max_depth=29,min_samples_split=11,n_jobs=-1)
y_pred = rf.fit(X_train,y_train).predict(X_test)
print("随机森林:")
pingjia(y_test, y_pred)

'''
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [ 1e-2],'C': [1, 10]},
                    {'kernel': ['linear'], 'gamma': [ 1e-2],'C': [1, 10]},
                    {'kernel': ['poly'], 'gamma': [ 1e-2],'C': [1, 10]}
                    ]
svm = svm.SVR()
grid_search = GridSearchCV(svm, param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)}, cv=5,n_jobs=-1,verbose=100)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

'''
#SVM训练
svm = svm.SVR(C=10.0,gamma=1000.0,kernel='rbf')
y_pred = svm.fit(X_train,y_train).predict(X_test)
print("SVM:")
pingjia(y_test, y_pred)

