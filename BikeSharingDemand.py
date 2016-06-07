import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
from sklearn.grid_search import GridSearchCV
from datetime import datetime
from sklearn.metrics import explained_variance_score
from sklearn.ensemble import RandomForestRegressor

bike_data=pd.read_csv("biketrain.csv",header=0)
bike_data['month'] = pd.DatetimeIndex(bike_data.datetime).month
bike_data['day'] = pd.DatetimeIndex(bike_data.datetime).dayofweek
bike_data['hour'] = pd.DatetimeIndex(bike_data.datetime).hour
for col in ['casual', 'registered', 'count']:
    bike_data['log-' + col] = bike_data[col].apply(lambda x: np.log1p(x))
bike_data_target=bike_data['log-count']
features = ['season', 'workingday', 'weather',
        'temp', 'humidity', 'windspeed',
         'month', 'day', 'hour']
bike_data_train=bike_data[features]

# print(bike_data.corr())
clf_cal = RandomForestRegressor(n_estimators=1000, min_samples_split=11, oob_score=True)
clf_cal.fit(bike_data_train, bike_data_target)
print(clf_cal.feature_importances_)


X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    bike_data_train, bike_data_target, test_size=0.2, random_state=0)

# param_grid = {'learning_rate': [0.1, 0.05, 0.01],
#               'max_depth': [10, 15, 20],
#               'min_samples_leaf': [3, 5, 10, 20],
#               }
#
# est = ensemble.GradientBoostingRegressor(n_estimators=500)
# gs_cv = GridSearchCV(
#     est, param_grid, n_jobs=4).fit(
#     X_train, y_train)
#
# print(gs_cv.best_params_)   # 0.05 10 20#
#
# error_count = mean_absolute_error(y_test, gs_cv.predict(X_test))


# error_train=[]
# error_validation=[]
# for k in range(10, 501, 10):
#     clf = ensemble.GradientBoostingRegressor(
#         n_estimators=k, learning_rate = .05, max_depth = 10,
#         min_samples_leaf = 20)
#
#     clf.fit(X_train, y_train)
#     result = clf.predict(X_train)
#     error_train.append(
#         mean_absolute_error(result, y_train))
#
#     result = clf.predict(X_test)
#     error_validation.append(
#         mean_absolute_error(result, y_test))
#
# Plot the data
# x=range(10,501, 10)
# plt.style.use('ggplot')
# plt.plot(x, error_train, 'k')
# plt.plot(x, error_validation, 'b')
# plt.xlabel('Number of Estimators', fontsize=18)
# plt.ylabel('Error', fontsize=18)
# plt.legend(['Train', 'Validation'], fontsize=18)
# plt.title('Error vs. Number of Estimators', fontsize=20)
# plt.show()

clf = ensemble.GradientBoostingRegressor(
         n_estimators=80, learning_rate = .05, max_depth = 10,
         min_samples_leaf = 20)
clf.fit(X_train,y_train)
# result = clf.predict(X_test)
# result = np.expm1(result)
score=clf.score(X_test,y_test)
# df=pd.DataFrame({'datetime':test['datetime'], 'count':result})
# df.to_csv('results2.csv', index = False, columns=['datetime','count'])
print(score)