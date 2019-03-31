#polynomial regression with effective quadrature

#Perform munual grid search with Gaussian, Uniform, Chebshev, Gamma and Beta distribution of different orders.

#linear regression 

#import functions needed to set up the dataset
from fillNAN import fill_nan
from prepare_data import create_testdata
from preprocessing import preprocessing

#import functions specific to this model
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

#create datasets
trainval,test=create_testdata()
X,y=preprocessing(trainval,'DTU',False)


kfold=10 #default 10-fold cross validation

# specify parameters to compute grid search on the the parameter scoring function
param_dist = {"linearregression__normalize": [True, False],"polynomialfeatures__degree":[2,3,4],"polynomialfeatures__include_bias":[True, False],"polynomialfeatures__interaction_only":[True, False]}
scoring = {'mse': 'neg_mean_squared_error', 'corr': 'r2'}

#hard to impute by country mean so compromise is to have an overall mean strategy
poly = PolynomialFeatures(degree=3)
imputer=fill_nan()
lm = linear_model.LinearRegression()
pipeline=make_pipeline(imputer,poly,lm)

# Choose cross-validation techniques for the inner and outer loops,
# independently of the dataset.
# E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
inner_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)
outer_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)

clf = GridSearchCV(estimator=pipeline, param_grid=param_dist,scoring='neg_mean_squared_error', cv=inner_cv)
# Nested CV with parameter optimization
nested_score = cross_val_score(clf, X, y, scoring='neg_mean_squared_error',cv=outer_cv)

clf.fit(X,y)

print("Best model has mse of {0:6f}.".format(nested_score.mean()))
print(clf.best_params_)

#now predict the windspeed for test set and output final error (mse)
X_test,y_test=preprocessing(test,'DTU',False)
imputer.fit(X)
X_test=imputer.transform(X_test)
y_predict=clf.predict(X_test)
final_error=-np.sum(np.square(y_predict-y_test))/len(y_test)
print(final_error)



