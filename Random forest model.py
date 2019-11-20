import numpy as np
from numpy import loadtxt
from sklearn.model_selection import KFold
from sklearn import preprocessing, linear_model
from numpy import loadtxt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from math import sqrt
import numpy as numpy
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as pyplot
from sklearn.ensemble import RandomForestRegressor

# load data
dataset = loadtxt('Adhesivestrength-32.csv', delimiter=",")

# split data into X and y
y = dataset[:,4]
X = dataset[:,0:4]

# Standard Scalar
scaler = StandardScaler()
print(scaler.fit(np.array(X)))
X_scaled = scaler.transform(np.array(X))
print('X=', X)
print('X_scaled=', X_scaled)
print('mean of X_scaled =', X_scaled.mean(axis=0))
print('std of X_scaled =', X_scaled.std(axis=0))

# RF Regressor
model = linear_model.ElasticNet(alpha=0.1)

# k-fold split into train, test data
n_splits=32
kf = KFold(n_splits)
kf.get_n_splits(X_scaled)
print(kf)  

R=1
y_test_tot = np.zeros(32)
y_pred_test_tot = np.zeros(32)

for train_index, test_index in kf.split(X_scaled):
   print("Round", R, "-", "TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X_scaled[train_index], X_scaled[test_index]
   y_train, y_test = y[train_index], y[test_index] 
   R=R+1
 
   # Early stopping
   model.fit(X_train, y_train)
   
   # Prediction for train and test data
   y_pred_test = model.predict(X_test)
   y_pred_train = model.predict(X_train)
   
   # Evaludation RSME, MAE and R^2 for train and test data   
   print('Test data')
   print("Root mean squared error: %.2f" % sqrt(mean_squared_error(y_test, y_pred_test)))
   print('Mean absolute error: %.2f' % mean_absolute_error(y_test,y_pred_test))
   print(' ')
   y_test_tot[R-2]=y_test
   y_pred_test_tot[R-2]=y_pred_test
   
   print('Train data')
   print("Root mean squared error: %.2f" % sqrt(mean_squared_error(y_train, y_pred_train)))
   print('Mean absolute error: %.2f' % mean_absolute_error(y_train,y_pred_train))
   print('Variance score: %.2f' % r2_score(y_train, y_pred_train))
   print(' ')
   
   # Plot observation VS prediction
   fig, ax = plt.subplots()
   ax.scatter(y_test, y_pred_test, edgecolors=(0, 1, 0))
   ax.plot([0, 30], [0, 30], 'k--', lw=4)
   plt.ylim(0, 30)
   plt.xlim(0, 30)
   ax.set_xlabel('observed y (test data)')
   ax.set_ylabel('Predicted y (test data)')
   plt.show()

   fig, ax = plt.subplots()
   ax.scatter(y_train, y_pred_train, edgecolors=(0, 0, 0))
   ax.plot([0, 30], [0, 30], 'k--', lw=4)
   plt.ylim(0, 30)
   plt.xlim(0, 30)
   ax.set_xlabel('observed y (train data)')
   ax.set_ylabel('Predicted y (train data)')
   plt.show()

print('Summary')

# Plot observation VS prediction
fig, ax = plt.subplots()
ax.scatter(y_test_tot, y_pred_test_tot, edgecolors=(0, 1, 0))
ax.plot([-5, 35], [-5, 35], 'k--', lw=4)
plt.ylim(-5, 35)
plt.xlim(-5, 35)
ax.set_xlabel('observed y (test data)')
ax.set_ylabel('Predicted y (test data)')
plt.show()

print('Variance score: %.2f' % r2_score(y_test_tot, y_pred_test_tot))
print('Mean absolute error: %.2f' % mean_absolute_error(y_test_tot, y_pred_test_tot))
print('Root mean absolute error: %.2f' % sqrt(mean_squared_error(y_test_tot, y_pred_test_tot)))

# feature importance
print(model.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)),
model.feature_importances_)
pyplot.show()
