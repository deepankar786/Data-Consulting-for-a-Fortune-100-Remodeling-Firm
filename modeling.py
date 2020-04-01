# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 00:58:36 2020

@author: deepa
"""

""""#importing dependencies"""
import pandas as pd
import numpy as np
import zipfile
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
%matplotlib inline
from matplotlib import*
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import catboost as cb
from sklearn.neural_network import MLPClassifier
import psutil
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import VotingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
import scipy
import h2o
from h2o.automl import H2OAutoML, get_leaderboard


"""getting model data ready for fitting and prediction"""
#importing data created after feature engineering in the other patch
model_data = pd.read_csv('model_data_new.csv')

#creating list of columns to drop and dropping them from model data
col_drop = ['Unnamed: 0','set_for_year','taken_at_year','lead_id','home_city','home_zip_code','day_of_month','set_for_week_of_year','setter_lead_count_4w','setter_issue_count_4w','setter_lead_count_6m','setter_issue_count_6m','setter_performance_4w','leads_per_zip_4w','issues_per_zip_4w','leads_per_zip_6m','issues_per_zip_6m','issue_rate_per_zip_4w','leads_per_territory_4w','issues_per_territory_4w','leads_per_territory_6m','issues_per_territory_6m','issue_rate_per_territory_4w','taken_at_week_of_year','marketter_lead_count_1w','marketter_issue_count_1w','marketter_lead_count_4w','marketter_issue_count_4w','marketter_lead_count_6m','marketter_issue_count_6m','marketter_performance_4w']
model_data.drop(col_drop,axis = 1, inplace=True)

#list of new territories created 
new_terr = ['Austin','Charlotte','Dallas','Denver','Nashville','Tampa']

#segregating data into stable and new territories
model_data_stable = model_data.loc[~model_data.territory_name.isin(new_terr)]
model_data_new = model_data.loc[model_data.territory_name.isin(new_terr)]

#creating one-hot-encoded versions of categorical variables
data = pd.get_dummies(model_data_stable, prefix=['set_type','ls_status','src_catg_desc','src_grp_desc','home_state','territory_name','product','month_of_year','hour_of_day','quarter'], columns = ['set_type','ls_status','src_catg_desc','src_grp_desc','home_state','territory_name','product','month_of_year','hour_of_day','quarter'])

#segregating data into minority and majority class
data_majority = data[data.target==0]
data_minority = data[data.target==1]

#undersampling majority class data to have even split of issue and non-issue cases and concatenating
data_majority_undersampled = data_majority.sample(data_minority.shape[0])
# data_minority_undersampled = data_minority.sample(50000)
data_final = pd.concat([data_majority_undersampled, data_minority], axis=0)

data_final.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in data_final.columns]

#specifying columns
X = ['lead_gap','setter_experience','setter_performance_6m','issue_rate_per_zip_6m','issue_rate_per_territory_6m','year_lead_count','year_issue_count','quarter_lead_count','quarter_issue_count','marketter_experience','marketter_performance_6m','set_type_Add_On','set_type_Cancel_Save','set_type_New','set_type_Rehash','set_type_Reset','ls_status_active','ls_status_declined','ls_status_expired','ls_status_flagged','ls_status_not_eligible','src_catg_desc_BJ_s_Wholesale_Club','src_catg_desc_BJ_s_locations_handled_for_us_by_Smart_Circle','src_catg_desc_BJs_contact','src_catg_desc_Call_in__Job_Sign__Self_Gen__PWS_Website','src_catg_desc_Canvassing_by_Smart_Circle_employees','src_catg_desc_Concerts','src_catg_desc_Database_Campaign','src_catg_desc_Fairs_or_small_events','src_catg_desc_Field_Marketing_by_PWS_employees','src_catg_desc_Field_Marketing_by_ProBound','src_catg_desc_Flea_Market','src_catg_desc_Home_Shows','src_catg_desc_In_House','src_catg_desc_Inbound__Outbound_and_Installer_Referrals','src_catg_desc_Internet_Sources','src_catg_desc_Kmart','src_catg_desc_Mall_Kiosk','src_catg_desc_Malls','src_catg_desc_NASCAR_race_events','src_catg_desc_Previous_Customer','src_catg_desc_Print','src_catg_desc_Radio','src_catg_desc_Sam_s_Club_Future_Callback','src_catg_desc_Sam_s_Club_leads_that_are_Futures_and_120_s','src_catg_desc_Sam_s_Club_wholesale_location','src_catg_desc_Sporting_events','src_catg_desc_Television','src_catg_desc_Trade_Show_Events','src_catg_desc_Walmart_Contact','src_catg_desc_Walmarts','src_catg_desc_appt_generated_from_ECRA_marketing_program','src_grp_desc_BJs__Kmart__Sam_s_Club__Costco__Malls__Walmart','src_grp_desc_Event__Showroom__Cowtown','src_grp_desc_Prev__Cust___Net','src_grp_desc_Print__Radio__PWS_Website__Television__Referral__Job_Sign__Self_Gen__Call_In','src_grp_desc_TM__Canv__Referral_Outbound__Old_Lead','home_state_CO','home_state_CT','home_state_DC','home_state_DE','home_state_FL','home_state_GA','home_state_IL','home_state_IN','home_state_MA','home_state_MD','home_state_ME','home_state_MI','home_state_NC','home_state_NH','home_state_NJ','home_state_NY','home_state_OH','home_state_PA','home_state_RI','home_state_TX','home_state_VA','home_state_VT','home_state_WI','territory_name_Atlanta','territory_name_Boston','territory_name_Chicago','territory_name_Connecticut','territory_name_Detroit','territory_name_Houston','territory_name_Long_Island','territory_name_Maryland','territory_name_New_Jersey','territory_name_Philadelphia','product_DOORS','product_GUTTERS','product_OTHERS','product_ROOFING','product_SIDINGS','product_SOLAR','product_WINDOW','month_of_year_1','month_of_year_2','month_of_year_3','month_of_year_4','month_of_year_5','month_of_year_6','month_of_year_7','month_of_year_8','month_of_year_9','month_of_year_10','month_of_year_11','month_of_year_12','hour_of_day_0','hour_of_day_7','hour_of_day_8','hour_of_day_9','hour_of_day_10','hour_of_day_11','hour_of_day_12','hour_of_day_13','hour_of_day_14','hour_of_day_15','hour_of_day_16','hour_of_day_17','hour_of_day_18','hour_of_day_19','hour_of_day_20','hour_of_day_21','quarter_1','quarter_2','quarter_3','quarter_4']
Y = ['target']

#splitting into train test data with 80:20 split
split = 0.2
X_train, X_test, y_train, y_test = train_test_split(data_final[X], data_final[Y], test_size=split, random_state=12345)

#scaling data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#repeating the process of creating minority and majority class data for CAT Boost
data_majority_new = model_data_stable[model_data_stable.target==0]
data_minority_new = model_data_stable[model_data_stable.target==1]
data_majority_undersampled_new = data_majority_new.sample(data_minority_new.shape[0])
# data_minority_undersampled_new = data_minority_new.sample(50000)
data_final_new = pd.concat([data_majority_undersampled_new, data_minority_new], axis=0)

X_new = ['ls_status','src_catg_desc','src_grp_desc','home_state','territory_name','product','lead_gap','month_of_year','hour_of_day','quarter','setter_experience','setter_performance_6m','issue_rate_per_zip_6m','issue_rate_per_territory_6m','year_lead_count','year_issue_count','quarter_lead_count','quarter_issue_count','marketter_experience','marketter_performance_6m']

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(data_final_new[X_new], data_final_new[Y], test_size=0.2, random_state=12345)

X_train_new['month_of_year'] = X_train_new.month_of_year.astype('object')
X_train_new['hour_of_day'] = X_train_new.hour_of_day.astype('object')
X_train_new['quarter'] = X_train_new.quarter.astype('object')

X_test_new['month_of_year'] = X_test_new.month_of_year.astype('object')
X_test_new['hour_of_day'] = X_test_new.hour_of_day.astype('object')
X_test_new['quarter'] = X_test_new.quarter.astype('object')

categorical_features_indices = np.where(X_train_new.dtypes == np.object)[0]



"""Building machine learning models: 
We will try 6 different classifiers to find the best classifier that will best generalize the unseen(test) data."""
seed = 1234
'''Now initialize all the classifiers object.'''

'''#1.Random Forest Classifier'''
rf = RandomForestClassifier(random_state = seed, n_estimators = 100)

'''#2.Gradient Boosting Classifier'''
gbc = GradientBoostingClassifier(random_state = seed)

'''#3.Light Gradient Boosting Classifier'''
lgb = lgb.LGBMClassifier()

'''#4.Extreme Gradient Boosting Classifier'''
xgb = XGBClassifier()

'''#5.Cat Boosting Classifier'''
cb = cb.CatBoostClassifier()

'''#6.MLP Classifier'''
mlp = MLPClassifier(hidden_layer_sizes=(70,30,10),activation='relu',solver='adam',alpha=0.001,
                    batch_size=1000,learning_rate='constant',learning_rate_init=0.01,max_iter=200)

#fitting and predicting using Random Forest Classifier
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_test)

print("AUC",roc_auc_score(y_test,  rf_pred[:,1]))
print("Log Loss",log_loss(y_test,  rf_pred[:,1]))
print("Brier Score",brier_score_loss(y_test,  rf_pred[:,1]))

#fitting and predicting using Random Forest Classifier
gbc.fit(X_train, y_train)
gbc_pred = gbc.predict_proba(X_test)

print("AUC",roc_auc_score(y_test,  gbc_pred[:,1]))
print("Log Loss",log_loss(y_test,  gbc_pred[:,1]))
print("Brier Score",brier_score_loss(y_test,  gbc_pred[:,1]))

#fitting and predicting using Light Gradient Boosting Classifier
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict_proba(X_test)

print("AUC",roc_auc_score(y_test,  lgb_pred[:,1]))
print("Log Loss",log_loss(y_test,  lgb_pred[:,1]))
print("Brier Score",brier_score_loss(y_test,  lgb_pred[:,1]))

#fitting and predicting using Cat Boosting Classifier
cb.fit(X_train_new, y_train_new, cat_features=categorical_features_indices, eval_set=(X_test_new, y_test_new),plot=False)
cb_pred = cb.predict_proba(X_test_new)

print("AUC",roc_auc_score(y_test_new,  cb_pred[:,1]))
print("Log Loss",log_loss(y_test_new,  cb_pred[:,1]))
print("Brier Score",brier_score_loss(y_test_new,  cb_pred[:,1]))

#fitting and predicting using Extreme Gradient Boosting Classifier
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict_proba(X_test)

print("AUC",roc_auc_score(y_test,  xgb_pred[:,1]))
print("Log Loss",log_loss(y_test,  xgb_pred[:,1]))
print("Brier Score",brier_score_loss(y_test,  xgb_pred[:,1]))

#fitting and predicting using Multi Layer Perceptron Model (Neural Networks)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict_proba(X_test)

print("AUC",roc_auc_score(y_test,  mlp_pred[:,1]))
print("Log Loss",log_loss(y_test,  mlp_pred[:,1]))
print("Brier Score",brier_score_loss(y_test,  mlp_pred[:,1]))

"""ensembling"""
#employing a Ensemble with soft Voting classifier with all the classifiers
model = VotingClassifier(estimators=[('xgb', xgb), ('lgb', lgb), 
                                     ('cb',cb), ('gbc', gbc), ('mlp', mlp)], voting='soft')
model.fit(X_train,y_train)
model_pred = model_en.predict_proba(X_test)
model_pred_new = model_en.predict(X_test)

print("AUC",roc_auc_score(y_test,  model_pred[:,1]))
print("Log Loss",log_loss(y_test,  model_pred[:,1]))
print("Brier Score",brier_score_loss(y_test,  model_pred[:,1]))
print("F1 Score",f1_score(y_test,  model_pred_new))
print("Precision Score",precision_score(y_test,  model_pred_new))

#neural network with keras
input_dim = 129
# Neural network
model_nn = Sequential()
model_nn.add(Dense(100, input_dim=input_dim, activation='relu'))
model_nn.add(Dropout(0.2))
model_nn.add(BatchNormalization())
model_nn.add(Dense(50, activation='relu'))
model_nn.add(Dropout(0.2))
model_nn.add(BatchNormalization())
# model_nn.add(Dense(30, activation='relu'))
# model_nn.add(Dropout(0.2))
# model_nn.add(BatchNormalization())
model_nn.add(Dense(10, activation='relu'))
model_nn.add(Dropout(0.2))
model_nn.add(BatchNormalization())
model_nn.add(Dense(1, activation='sigmoid'))
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_nn.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])

model_nn.fit(X_train, y_train, epochs=50, batch_size=500)

pred_nn = model_nn.predict_proba(X_test)

print("AUC",roc_auc_score(y_test, pred_nn))
print("Log Loss",log_loss(y_test, pred_nn))
print("Brier Score",brier_score_loss(y_test, pred_nn))

#H20
h2o.init()

train = h2o.H2OFrame(pd.concat([X_train,y_train], axis = 1))
test = h2o.H2OFrame(pd.concat([X_test,y_test], axis = 1))

# Identify predictors and response
x = train.columns
y = "target"
x.remove(y)

# For binary classification, response should be a factor
train['target'] = train['target'].asfactor()
test['target'] = test['target'].asfactor()

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# AutoML Leaderboard
lb = aml.leaderboard

# Optionally add extra model information to the leaderboard
lb = get_leaderboard(aml, extra_columns='ALL')

# Print all rows (instead of default 10 rows)
lb.head(rows=lb.nrows)

# The leader model is stored here
aml.leader

# If you need to generate predictions on a test set, you can make
# predictions directly on the `"H2OAutoML"` object, or on the leader
# model object directly

preds = aml.predict(test)

# or:
preds = aml.leader.predict(test)

lb.head(rows=lb.nrows)

aml_pred = preds.as_data_frame()

print("AUC",roc_auc_score(y_test,aml_pred['p1']))
print("Log Loss",log_loss(y_test,aml_pred['p1']))
print("Brier Score",brier_score_loss(y_test,aml_pred['p1']))

# Get model ids for all models in the AutoML Leaderboard
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
# Get the "All Models" Stacked Ensemble model
se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])
# Get the Stacked Ensemble metalearner model
metalearner = h2o.get_model(se.metalearner()['name'])

%matplotlib inline
metalearner.std_coef_plot()


#plotting probability calibration plots
import sklearn.calibration

ens_y, ens_x = sklearn.calibration.calibration_curve(y_test, model_pred_s[:,1], normalize=False, n_bins=25, strategy='uniform')
lgb_y, lgb_x = sklearn.calibration.calibration_curve(y_test, lgb_pred[:,1], normalize=False, n_bins=25, strategy='uniform')
cb_y, cb_x = sklearn.calibration.calibration_curve(y_test, cb_pred[:,1], normalize=False, n_bins=25, strategy='uniform')
xgb_y, xgb_x = sklearn.calibration.calibration_curve(y_test, xgb_pred[:,1], normalize=False, n_bins=25, strategy='uniform')
mlp_y, mlp_x = sklearn.calibration.calibration_curve(y_test, mlp_pred[:,1], normalize=False, n_bins=25, strategy='uniform')

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

import matplotlib.backends.backend_pdf as pdf
fig, ax = plt.subplots(figsize=(10,7))
plt.plot(lgb_x,lgb_y, marker='o', linewidth=1, label='Ensemble')
plt.plot(mlp_x,mlp_y, marker='o', linewidth=1, label='ANN')
plt.plot(ens_x, ens_y, marker='o', linewidth=1, label='Lightgbm')
plt.plot(xgb_x,xgb_y, marker='o', linewidth=1, label='XGBoost')
plt.plot(cb_x,cb_y, marker='o', linewidth=1, label='CatBoost')

pdf_construct = pdf.PdfPages("calplot.pdf")
# reference line, legends, and axis labels
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle('Calibration plot')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability in each bin')
plt.legend()
plt.show()
pdf_construct.savefig(fig)
pdf_construct.close()

#plotting lift and gain charts
import scikitplot as skplt
fig = plt.figure(figsize=(10,9))
skplt.metrics.plot_lift_curve(y_test,pred_ens_new)
plt.grid(False)
plt.savefig('lift.pdf',format='pdf')
skplt.metrics.plot_cumulative_gain(y_test, pred_ens_new)
plt.grid(False)
plt.savefig('gain.pdf',format='pdf')