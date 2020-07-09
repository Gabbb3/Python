# TO DO: ADD XGBOOST, LSTM, SVM, Logistic Regression
######################### Splitting dataframe for modelling #########################
Y = df["y"]
X = df.drop(["y"], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.2)

#### Scoring Metrics ###
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

##################################### Random Forest Classifier ########################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from math import *

param_grid = { 
            "n_estimators"      : [100, 200, 300], # default=100
#             "max_features"      : ["auto", "sqrt", "log2"], #default=auto
#             "min_samples_split" : [2,4,8], #default=2
#             "bootstrap": [True, False], #default=True
            }

RF = RandomForestClassifier()
RF_cv = GridSearchCV(RF, param_grid=param_grid, cv=5, scoring = "accuracy")
RF_cv = RF_cv.fit(X_train, Y_train)
print(RF_cv.best_score_, RF_cv.best_params_)
pred_RF = RF_cv.predict(X_test)

# Feature Importance
feat_labels = X.columns.values
importances = RF_cv.best_estimator_.feature_importances_
indices = np.argsort(importances)

plt.title('RF Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feat_labels[i] for i in indices])
plt.xlabel('Relative Importance')
# plt.savefig("RFFeatureImportance", bbox_inches="tight")

##################################### Decision Tree Classifier ########################################

from sklearn import tree
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': np.arange(3, 10)}

DT = tree.DecisionTreeClassifier()
DT_cv = GridSearchCV(DT, param_grid=param_grid, cv=5, scoring = "accuracy")
DT_cv = DT_cv.fit(X_train, Y_train)
print(DT_cv.best_score_, DT_cv.best_params_)
pred_DT = DT_cv.predict(X_test)

##################################### Naive Bayes Classifier ########################################

from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()   # Build NB classifier - No parameters available to be tuned
NB = NB.fit(X_train, Y_train)
pred_NB = NB.predict(X_test)

##################################### ADABoost Classifier ########################################

from sklearn.ensemble import AdaBoostClassifier

param_grid = { 
            "n_estimators"      : [100, 200, 300], # default=100
#             "max_features"      : ["auto", "sqrt", "log2"], #default=auto
#             "min_samples_split" : [2,4,8], #default=2
#             "bootstrap": [True, False], #default=True
            }

ADA = AdaBoostClassifier()
ADA_cv = GridSearchCV(ADA, param_grid=param_grid, cv=5, scoring = "accuracy")
ADA_cv = ADA_cv.fit(X_train, Y_train)
pred_ADA = ADA.predict(X_test)

##################################### Gradient Boosting Classifier ########################################

from sklearn.ensemble import GradientBoostingClassifier

param_grid = {'n_estimators':range(20,81,10)}

# First grid search uses "default" lr, min_samples_split etc. to find optimal n_estimators. Thereafter, we will tune for the rest etc.
gsearch1 = GridSearchCV(
            estimator = GradientBoostingClassifier(
                        learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10),
param_grid = param_grid, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(X_train, Y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_ # To print gridsearch cv scores

# Using the optimal n_estimators, we then tune for others first in the following priority to reduce computational load
# 1) max_depth & min_samples_split - param_grid2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)} #max_depth between 5-8, min_samples_split=~0.5-1% of total
# 2) min_samples_leaf - param_grid3 = {'min_samples_leaf':range(30,71,10)} #50-100
# 3) max_features - param_grid4 = {'max_features':range(7,20,2)} #sqrt, auto, log2
# 4) subsample - param_grid5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}

##################################### Neural Network Classifier (keras) ########################################

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
NN_pred = model.predict_classes(X_test)

##################################### Voting Classifier ########################################

from sklearn.ensemble import VotingClassifier

clf_vc = VotingClassifier(estimators=[('DT', DT_cv), ("NB", NB_cv), ("RF", RF_cv)], voting = "hard") #Soft voting = average of probabilities. Hard Voting = Average of predicted classes
clf_vc.fit(X_train, Y_train)
# clf_vc.predict_proba(X_test) # For only soft voting
pred_clf_cv = clf_cv.predict(X_test)

################################################# Classifier Evaluations ####################################################

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, pred) # Rows are actual labels, while columns are predicted labels

# Classification report (Precision, Recall, F1, Accuracy)
report = classification_report(Y_test, pred_RF)
print(report)

# Accuracy - CV Training
all_accuracies = cross_val_score(estimator=RF_cv, X=X_train, y=Y_train, cv=5)
print(all_accuracies)
print("The average accuracy is {}".format(all_accuracies.mean()))

# Accuracy
print("Number of mislabeled points out of a total %d points : %d" % 
      (X_test.shape[0],(Y_test!= pred_RF).sum()))
accuracy = 1 - (Y_test!=pred_RF).sum() / X_test.shape[0]
print(accuracy)

# Accuracy (sklearn)
from sklearn.metrics import accuracy_score
print("Accuracy for Random Forest on CV data: ", accuracy_score(Y_test,pred_RF))

# F1 (sklearn)
F1 = f1_score(Y_test, pred_DT, average=None)
print("The F1-measure for class Y=1 is {}".format(F1[1]))




################################################## Random Forest Regressor #####################################################

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

param_grid = { 
            "n_estimators"      : [100, 200, 300], # default=100
#             "max_features"      : ["auto", "sqrt", "log2"], #default=auto
#             "min_samples_split" : [2,4,8], #default=2
#             "bootstrap": [True, False], #default=True
            }


RFR = RandomForestRegressor()
RFR_cv = GridSearchCV(RFR, param_grid, cv=5, scoring="r2")
RFR_cv.fit(X_train, Y_train)
    
# Feature Importance
feat_labels = X.columns.values
importances = RFR_cv.best_estimator_.feature_importances_
indices = np.argsort(importances)

plt.title('RF Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feat_labels[i] for i in indices])
plt.xlabel('Relative Importance')
# plt.savefig("RFFeatureImportance", bbox_inches="tight")

print(RFR_cv.best_score_ , RFR_cv.best_params_)

pred_RFR = RFR_cv.predict(X_test)
r2_score(pred_RFR, Y_test)
print("The r2 on test data set is {}.".format(r2_score(pred_RFR,Y_test)))

##################################### Neural Network - Regression ########################################
# https://www.tensorflow.org/tutorials/keras/regression

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
pred_NNR = model.predict(X_test)
