# New file: Oversampling methods SMOTE, ADASYN
# Other file: PDPBox, P&C and left_join back
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

rf_importance = pd.DataFrame()
rf_importance["features"] = feat_labels
rf_importance["importances"] = importances
rf_importance = rf_importance.sort_values(["importances"], ascending=0)

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

##################################### Logistic Regression Classifier ########################################

from sklearn.linear_model import LogisticRegression

# Removing low variance inputs due to initial error from fitting logistic model
from sklearn.feature_selection import VarianceThreshold

def variance_threshold_selector(data, threshold=0.5):
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
    # https://stackoverflow.com/a/39813304/1956309
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]
# min_variance = .9 * (1 - .9)  # You can play here with different values.
min_variance = 0.01
low_variance_removed = variance_threshold_selector(X_train, min_variance) 
print('columns removed:', low_variance_removed.columns)
X_train_var = low_variance_removed

param_grid={"C":np.logspace(-3,3,7), 
            "penalty":["l1, l2"], 
            "max_iter": [1000, 2000]}

#### Option 1. Using sklearn ####
LR_cv = GridSearchCV(LogisticRegression(),grid,cv=5)
LR_cv = LR_cv.fit(X_train_var,Y_train) ## Initial ERROR: Check levels of inputs - Likely to have 99%~ in one level for an input

# View best hyperparameters
print('Best C:', LR_cv.best_estimator_.get_params())

# Using abs(coef) to get feature importance
feat_labels = X_train_var.columns.values
importances = LR_cv.best_estimator_.coef_[0]
indices = np.argsort(importances)
lr_importance = pd.DataFrame()
lr_importance["features"] = feat_labels
lr_importance["importances"] = importances
lr_importance = lr_importance.sort_values(["importances"], ascending=0)

for i in range(len(lr_importance["importances"])):
    if lr_importance["importances"][i] < 0:
        lr_importance["importances"][i] = -lr_importance["importances"][i] # Changing values of lasso coeff to absolute values

lr_importance = lr_importance.sort_values(["importances"], ascending=0)
lr_importance

# Remember to remove the low variance columns 
X_test_var = X_test[low_variance_removed.columns]

#### Option 2. Using statsmodel ####
import statsmodels.api as sm
from statsmodels.formula.api import logit

# Describe model
mod = sm.Logit(Y_train.astype(float), X_train_var.astype(float))    

# Fit model
clf_logit = mod.fit(maxiter=3000)      
# Summarize model
print(clf_logit.summary())  
print(clf_logit.get_margeff().summary())

# Feature importance - Remove pvalue >=0.05 and sort by coef
df_logit = pd.DataFrame([X_test_var.columns.values, clf_logit.params, clg_logit.pvalues]).T
df_logit.columns = ["features", "coef", "pvalues"]
df_logit["pvalues"] = pd.to_numeric(df_logit["pvalues"])
df_logit["coef"] = pd.to_numeric(df_logit["coef"])
df_logit = df_logit.drop(df_logit[df_logit.pvalues > 0.05].index)
df_logit = df_logit.dropna()
logit_importance = df_logit.sort_values(["coef"], ascending=0)
logit_importance

# ROC Curve & AUC Score - If used often, make it into a function instead
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

pred_logit = clf_logit.predict(X_test_var)
logit_auc = roc_auc_score(Y_test, pred_logit)
ns_probs = [0 for _ in range(len(Y_test))]
ns_auc = roc_auc_score(Y_test, ns_probs)
Y_test_array = np.array(Y_test)
Y_test_array = Y_test_array.astype(float)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (logit_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(Y_test_array, ns_probs)
logit_fpr, logit_tpr, _ = roc_curve(Y_test_array, pred_logit)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(logit_fpr, logit_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

##################################### Support Vector Machine (SVM) Classifier ########################################

kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']

def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return SVC(kernel='linear', gamma="auto")

for i in range(4):
# Train a SVC model using different kernal
    svclassifier = getClassifier(i) 
    svclassifier.fit(X_train, Y_train)
# Make prediction
    y_pred = svclassifier.predict(X_test)
# Evaluate our model
    print("Evaluation:", kernels[i], "kernel")
    print(classification_report(Y_test,y_pred))

# Using the best kernel (in this case, using linear)
clf_sv = getClassifier(3) 
clf_sv.fit(X_train, Y_train)
feat_labels = X_train.columns.values
importances = clf_sv.coef_.T

# Feature importance
indices = np.argsort(importances)
svc_importance = pd.DataFrame()
svc_importance["features"] = feat_labels
svc_importance["importances"] = importances
svc_importance = svc_importance.sort_values(["importances"], ascending=0)

for i in range(len(svc_importance["importances"])):
    if svc_importance["importances"][i] < 0:
        svc_importance["importances"][i] = -svc_importance["importances"][i] # Changing values of svc coeff to absolute values

svc_importance = svc_importance.sort_values(by=['importances'], ascending=False)
svc_importance

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

##################################### XGBoost Classifier ########################################
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

param_grid = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

# Param training can be done similar to the above Gradient Boosting Classifer instead of below
xg_cv = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_grid, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

xg_cv.fit(X_train,Y_train)

# xg.cv_results_
xg_cv.best_params_, xg_cv.best_score_

# Using above max_depth and min_child_weight to tune gamma
param_grid2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}

xg_cv2 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=9,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_grid2, scoring='roc_auc',n_jobs=4, cv=5)

xg_cv2.fit(X_train,Y_train)
xg_cv2.best_params_, xg_cv2.best_score_

# Prediction
pred_xg = xg_cv2.predict(X_test)

# Feature importance
feat_labels = X_test.columns.values
importances = xg_cv2.best_estimator_.feature_importances_
indices = np.argsort(importances)
xg_importance = pd.DataFrame()
xg_importance["features"] = feat_labels
xg_importance["importances"] = importances
xg_importance = xg_importance.sort_values(["importances"], ascending=0)
xg_importance

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

##################################### LSTM - Classification ########################################

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional
from keras.optimizers import SGD
from keras.optimizers import Adam

# Ensure that data is scaled
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)

# Dataframe (df) will be in this format (Person1, Target_Value, t1_feature1, t1_feature2 ; Person1, Target_Value, t2_feature1, t1_feature2)
# Reshape into 3D array (sample, timesteps, n_features)
X = df.drop([Target_Value], axis=1)
X_3d = array(X).reshape(len(X_train[col].unique()), 10, len(X_train.columns)-1) # 10 timesteps, Col refers to unique identifier across sets of samples (e.g. Name) (-1 to remove "Person" ID from n_features)

Y = df[[Person, Target_Value]].drop_duplicates()
Y[Target_Value].value_counts()
Y = Y[Target_Value]

# OHE Target Variable (Y)
from keras.utils.np_utils import to_categorical
Y = Y.astype('category')
Y = Y.cat.codes
Y = to_categorical(Y, num_classes=2)

# Train Test Split (80/20)
X_train_3d, X_test_3d, Y_train, Y_test = train_test_split(X_3d, Y, shuffle=True, test_size=0.2)

# Simple LSTM model
model = Sequential()
model.add(LSTM(150, activation='relu', return_sequences=True, input_shape=(10, 14))) #input_shape = (time_steps, n_features)
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2)) # Add drop-out layer to prevent overfitting
model.add(Dense(2, activation='softmax')) # https://keras.io/api/layers/activations/
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy']) #loss: https://keras.io/api/losses/

model.fit(X_train_3d, Y_train, epochs=50, verbose=1, shuffle=False, batch_size=200, validation_data=(X_test_3d, Y_test))
print(model.summary())

# Feature Importance (SHAP)

# Prediction
pred_LSTM_train_scaled = model.predict(X_train_3d)
pred_LSTM_test_scaled = model.predict(X_test_3d)
# invert predictions
pred_LSTM_train = scaler.inverse_transform(pred_LSTM_train_scaled)
Actual_Y_train = scaler.inverse_transform([Y_train])
pred_LSTM_test = scaler.inverse_transform(pred_LSTM_test_scaled)
Actual_Y_test = scaler.inverse_transform([Y_test])

# Note: There also exists the ability to retain memory between batches if needed.
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

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
RFR_cv = GridSearchCV(RFR, param_grid, cv=5, scoring="neg_mean_squared_error")
RFR_cv.fit(X_train, Y_train)
print(RFR_cv.best_score_ , RFR_cv.best_params_)
    
# Feature Importance
feat_labels = X.columns.values
importances = RFR_cv.best_estimator_.feature_importances_
indices = np.argsort(importances)

rf_importance = pd.DataFrame()
rf_importance["features"] = feat_labels
rf_importance["importances"] = importances
rf_importance = rf_importance.sort_values(["importances"], ascending=0)

plt.title('RF Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feat_labels[i] for i in indices])
plt.xlabel('Relative Importance')
# plt.savefig("RFFeatureImportance", bbox_inches="tight")

pred_RFR = RFR_cv.predict(X_test)
r2_score(pred_RFR, Y_test)
print("The r2 on test data set is {}.".format(r2_score(pred_RFR,Y_test)))


################################################## Lasso Regression (L1)  #####################################################
from sklearn.linear_model import Lasso

param_grid = {"alpha": [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20], "max_iter": [1000, 2000, 3000]}

LS = GridSearchCV(Lasso(), param_grid, scoring="neg_mean_squared_error", cv=5)
LS_cv = LS.fit(X_train, Y_train)
print(LS_cv.best_score_ , LS_cv.best_params_)

# Using lasso coef as importance
best_LS = LS_cv.best_estimator_
best_LS = best_LS.fit(X_train, Y_train)
pred_LS = best_LS.predict(X_test)
for i in range(len(best_LS.coef_)):
    if best_LS.coef_[i] < 0:
        best_LS.coef_[i] = -best_LS.coef_[i] # Changing values of lasso coeff to absolute values
LS_importance = pd.DataFrame()
LS_importance["features"] = X_train.columns
LS_importance["importances"] = best_LS.coef_
LS_importance = LS_importance.sort_values(["importances"], ascending=0)
LS_importance

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

##################################### LSTM - Classification ########################################
# Most predictions like Revenue etc. use timestep=1

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional
from keras.optimizers import SGD
from keras.optimizers import Adam

# Ensure that data is scaled
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)

# Dataframe (df) will be in this format (Person1, Target_Value, t1_feature1, t1_feature2 ; Person1, Target_Value, t2_feature1, t1_feature2)
# Reshape into 3D array (sample, timesteps, n_features)
X = df.drop([Target_Value], axis=1)
X_3d = array(X).reshape(len(X_train[col].unique()), timesteps, len(X_train.columns)-1) # 10 timesteps, Col refers to unique identifier across sets of samples (e.g. Name) (-1 to remove "Person" ID from n_features)

Y = df[[Person, Target_Value]].drop_duplicates()
Y = Y[Target_Value]

# OHE Target Variable (Y)
from keras.utils.np_utils import to_categorical
Y = Y.astype('category')
Y = Y.cat.codes
Y = to_categorical(Y, num_classes=2)

# Train Test Split (80/20)
X_train_3d, X_test_3d, Y_train, Y_test = train_test_split(X_3d, Y, shuffle=True, test_size=0.2)

# Simple LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(timesteps, look_back))) # lookback=1: X=t, y=t+1
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam') #loss: https://keras.io/api/losses/

model.fit(X_train_3d, Y_train, epochs=50, verbose=1, shuffle=False, batch_size=200, validation_data=(X_test_3d, Y_test))
print(model.summary())

# Feature Importance (SHAP)

# Prediction
pred_LSTM_train_scaled = model.predict(X_train_3d)
pred_LSTM_test_scaled = model.predict(X_test_3d)
# invert predictions
pred_LSTM_train = scaler.inverse_transform(pred_LSTM_train_scaled)
Actual_Y_train = scaler.inverse_transform([Y_train])
pred_LSTM_test = scaler.inverse_transform(pred_LSTM_test_scaled)
Actual_Y_test = scaler.inverse_transform([Y_test])

# Note: There also exists the ability to retain memory between batches if needed.
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

################################################# Regression Evaluations ####################################################

# Mean squared error
LS_mse = metrics.mean_squared_error(Y_test, pred_LS)
LS_mse
