#polynomial regression with effective quadrature

#Perform munual grid search with Gaussian, Uniform, Chebshev, Gamma and Beta distribution of different orders.

#linear regression 

#import functions needed to set up the dataset
from prepare_data import create_testdata
from preprocessing import preprocessing

#import functions specific to this model
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#create datasets
df=preprocessing()
features_train,windspeed_train,features_test,windspeed_test=create_testdata(df,'50')

kfold=10 #default 10-fold cross validation

param_dist = {"linearregression__normalize": [True, False],"polynomialfeatures__degree":[2],"polynomialfeatures__include_bias":[True, False],"polynomialfeatures__interaction_only":[True, False]}
scoring = {'mse': 'neg_mean_squared_error', 'corr': 'r2'}

#hard to impute by country mean so compromise is to have an overall mean strategy

poly = PolynomialFeatures(degree=2)
lm = linear_model.LinearRegression()
pipeline=make_pipeline(poly,lm)

# Choose cross-validation techniques for the inner and outer loops,
# independently of the dataset.
# E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
inner_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)
outer_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)

clf = GridSearchCV(estimator=pipeline, param_grid=param_dist,scoring='neg_mean_squared_error', cv=inner_cv)
# Nested CV with parameter optimization
nested_score = cross_val_score(pipeline, features_train, windspeed_train, scoring='neg_mean_squared_error',cv=outer_cv)

clf.fit(features_train,windspeed_train)

print("Best model has mse of {0:6f}.".format(nested_score.mean()))
print(clf.best_params_)

#now predict the windspeed for test set and output final error (mse)
windspeed_predict=clf.predict(features_test)
final_error=-np.sum(np.square(windspeed_predict-windspeed_test))/len(windspeed_test)
print(final_error)



