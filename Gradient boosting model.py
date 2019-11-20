import numpy as np
from numpy import loadtxt
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from math import sqrt
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score

# load data
dataset = loadtxt('Adhesivestrength-32.csv', delimiter=",")

# split data into X and y
y = dataset[:,4]
X = dataset[:,0:4]

# Standard Scalar for X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(np.array(X))
print('X=', X)
print('X_scaled=', X_scaled)
print('mean of X_scaled =', X_scaled.mean(axis=0))
print('std of X_scaled =', X_scaled.std(axis=0))

# Load 256 possible condition and standard scalar
X_all=loadtxt('data2019-4.csv', delimiter=",")
X_all_scaled = scaler.transform(np.array(X_all))
number=len(X_all)
y_all=np.zeros(number)
y_all=[]*number

# XGBoost Regressor
model = xgb.XGBRegressor(max_depth=5, gamma=3.3)

# k-fold split into train, test-validation data
n_splits=32
kf = KFold(n_splits)
kf.get_n_splits(X_scaled)
print(kf)  

# Round count and collect data
R=1
y_test_tot = np.zeros(32)
y_pred_test_tot = np.zeros(32)

# 32 Loop of prediction 
for train_valid_index, test_index in kf.split(X_scaled):
   print("Round", R, "-", "TRAIN+VALID:", train_valid_index, "TEST:", test_index)
   X_train_valid, X_test = X_scaled[train_valid_index], X_scaled[test_index]
   y_train_valid, y_test = y[train_valid_index], y[test_index] 
   R=R+1
   
   # Split into train data and validation data
   X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.2, random_state=4)
   
   # Early stopping
   eval_set = [(X_valid, y_valid)]
   model.fit(X_train, y_train, early_stopping_rounds=5, eval_metric='rmse', eval_set=eval_set, verbose=True)
   
   # Prediction for train, test and validation data
   y_pred_test = model.predict(X_test, ntree_limit=model.best_ntree_limit)
   y_pred_train = model.predict(X_train, ntree_limit=model.best_ntree_limit)
   y_pred_valid = model.predict(X_valid, ntree_limit=model.best_ntree_limit)
      
   # Prediction for all conditions
   y_pred_all = model.predict(X_all_scaled, ntree_limit=model.best_ntree_limit)
   
   # Collect data
   if R==2:
       y_all=y_pred_all
   else:
       y_all=np.vstack((y_all,y_pred_all))
       
   # Evaludation RSME, MAE and R^2 for train, test and validation data
   print('Validation data')
   print("Root mean squared error: %.1f" % sqrt(mean_squared_error(y_valid, y_pred_valid)))
   print('Mean absolute error: %.1f' % mean_absolute_error(y_valid,y_pred_valid))
   print('Variance score: %.2f' % r2_score(y_valid, y_pred_valid))
   print(' ')
   
   print('Test data')
   print("Root mean squared error: %.1f" % sqrt(mean_squared_error(y_test, y_pred_test)))
   print('Mean absolute error: %.1f' % mean_absolute_error(y_test,y_pred_test))
   print(' ')
   y_test_tot[R-2]=y_test
   y_pred_test_tot[R-2]=y_pred_test
   
   print('Train data')
   print("Root mean squared error: %.1f" % sqrt(mean_squared_error(y_train, y_pred_train)))
   print('Mean absolute error: %.1f' % mean_absolute_error(y_train,y_pred_train))
   print('Variance score: %.2f' % r2_score(y_train, y_pred_train))
   print(' ')
   
   # Plot observation VS prediction
   print('Predction VS observation plot for validation data')
   fig, ax = plt.subplots()
   ax.scatter(y_valid, y_pred_valid, edgecolors=(0, 0, 0))
   ax.plot([0, 35], [0, 35], 'k--', lw=4)
   plt.ylim(0, 35)
   plt.xlim(0, 35)
   ax.set_xlabel('observed y (valid data)')
   ax.set_ylabel('Predicted y (valid data)')
   plt.show()

   print('Predction VS observation plot for test data')
   fig, ax = plt.subplots()
   ax.scatter(y_test, y_pred_test, edgecolors=(0, 0, 0))
   ax.plot([0, 35], [0, 35], 'k--', lw=4)
   plt.ylim(0, 35)
   plt.xlim(0, 35)
   ax.set_xlabel('observed y (test data)')
   ax.set_ylabel('Predicted y (test data)')
   plt.show()

   print('Predction VS observation plot for train data')
   fig, ax = plt.subplots()
   ax.scatter(y_train, y_pred_train, edgecolors=(0, 0, 0))
   ax.plot([0, 35], [0, 35], 'k--', lw=4)
   plt.ylim(0, 35)
   plt.xlim(0, 35)
   ax.set_xlabel('observed y (train data)')
   ax.set_ylabel('Predicted y (train data)')
   plt.show()
   print('--------------------------------------------------------')
   
print(' ')
print('Summary')

# Plot observation VS prediction
fig, ax = plt.subplots()
ax.scatter(y_test_tot, y_pred_test_tot, edgecolors=(0, 1, 0))
ax.plot([0, 35], [0, 35], 'k--', lw=4)
plt.ylim(0, 35)
plt.xlim(0, 35)
ax.set_xlabel('observed y (test data)')
ax.set_ylabel('Predicted y (test data)')
plt.show()

print('Variance score: %.2f' % r2_score(y_test_tot, y_pred_test_tot))
print('Mean absolute error: %.1f' % mean_absolute_error(y_test_tot, y_pred_test_tot))
print('Root mean square error: %.1f' % sqrt(mean_squared_error(y_test_tot, y_pred_test_tot)))

# feature importance
print(' ')
print('Feature importance')
print(model.feature_importances_)
    

np.savetxt('prediction_2019-1.csv', y_all, fmt='%.2f', delimiter=',')