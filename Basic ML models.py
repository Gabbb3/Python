######################### Splitting dataframe for modelling #########################
Y = df["y"]
X = df.drop(["y"], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.2)

#### Scoring Metrics ###
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

##################################### Random Forest Classifier ########################################

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from math import *

param_grid = { 
            "n_estimators"      : [10,20,30],
#             "max_features"      : ["auto", "sqrt", "log2"],
#             "min_samples_split" : [2,4,8],
#             "bootstrap": [True, False],
            }

# clfRF = RandomForestClassifier(n_estimators=10, max_features = int(sqrt(X_train.shape[1])))
clfRF = RandomForestClassifier()
clfRFCV = GridSearchCV(clfRF, param_grid=param_grid, cv=5, scoring = "accuracy")
clfRFCV = clfRFCV.fit(X_train, Y_train)

predRF = clfRFCV.predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % 
      (X_test.shape[0],(Y_test!= predRF).sum()))             
score = 1 - (Y_test!=predRF).sum() / X_test.shape[0]
print(score) ## 0.8155

# Feature Importance
feat_labels = X.columns.values
importances = clfRFCV.best_estimator_.feature_importances_
indices = np.argsort(importances)

plt.title('RF Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feat_labels[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig("RFFeatureImportance", bbox_inches="tight")

##################################### Decision Tree Classifier ########################################

from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

param_grid = {'max_depth': np.arange(3, 10)}

clfDT = tree.DecisionTreeClassifier()
clfDTCV = GridSearchCV(clfDT, param_grid=param_grid, cv=5, scoring = "accuracy")
clfDTCV = clfDTCV.fit(X_train, Y_train)
print(clfDTCV.best_score_, clfDTCV.best_params_)

all_accuracies = cross_val_score(estimator=clfDTCV, X=X_train, y=Y_train, cv=5)
print(all_accuracies)
print("The average accuracy is {}".format(all_accuracies.mean())) ## 0.8178

predDT = clfDTCV.predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d" % 
#       (X_test.shape[0],(Y_test!= predDT).sum()))
# score = 1 - (Y_test!=predDT).sum() / X_test.shape[0]
# print(score)

report = classification_report(Y_test, predDT)
print(report)

from sklearn.metrics import accuracy_score
print("Accuracy for Random Forest on CV data: ",accuracy_score(Y_test,predDT))

F1 = f1_score(Y_test, predDT, average=None)
print("The F1-measure for class Y=1 is {}".format(F1[1])) ## 0.505

##################################### Neural Network Classifier ########################################

from keras.models import Sequential
from keras.layers import Dense

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, Y_train, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X_train, Y_train)
print('Accuracy: %.2f' % (accuracy*100))
# make class predictions with the model
predictions = model.predict_classes(X)


##################################### Random Forest Regressor ########################################

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

estimator = RandomForestRegressor()
param_grid = { 
            "n_estimators"      : [10,20,30],
#             "max_features"      : ["auto", "sqrt", "log2"],
#             "min_samples_split" : [2,4,8],
#             "bootstrap": [True, False],
            }

RFR = GridSearchCV(estimator, param_grid, cv=5, scoring="r2")
RFR.fit(X_train, Y_train)
    
# Feature Importance
feat_labels = X.columns.values
importances = RFR.best_estimator_.feature_importances_
indices = np.argsort(importances)

plt.title('RF Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feat_labels[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig("RFFeatureImportance", bbox_inches="tight")

print(RFR.best_score_ , RFR.best_params_)

predRFR = RFR.predict(X_test)
r2_score(predRFR, Y_test)
print("The r2 on test data set is {}.".format(r2_score(predRFR,Y_test)))

##################################### Neural Network - Regression ########################################
https://www.tensorflow.org/tutorials/keras/regression

from keras.models import Sequential
from keras.layers import Dense

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='mse', optimizer='adam')
# fit the keras model on the dataset
model.fit(X, y, epochs=10, batch_size=10)
Pred_y = model.predict(X)
