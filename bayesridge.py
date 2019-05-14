#Bayesian ridge
#import functions needed to set up the dataset
from prepare_data import create_testdata
from preprocessing import preprocessing

#import functions specific to this model
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
import numpy as np

#create datasets
df=preprocessing()
features_train,windspeed_train,features_test,windspeed_test=create_testdata(df,'200')

kfold=10 #default 10-fold cross validation

outer_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)

reg = linear_model.BayesianRidge()

nested_score = cross_val_score(reg, features_train, windspeed_train, scoring='neg_mean_squared_error',cv=outer_cv)
print("Model has mse of {} with standard deviation {}.".format(nested_score.mean(),nested_score.std()))


reg.fit(features_train, windspeed_train)
y_predict=reg.predict(features_test)
final_error=-np.sum(np.square(y_predict-windspeed_test))/len(windspeed_test)
print('mse error on test set is {}.'.format(final_error))