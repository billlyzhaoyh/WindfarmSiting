#pls regression
#linear regression 

#import functions needed to set up the dataset
from fillNAN import fill_nan
from prepare_data import create_testdata
from preprocessing import preprocessing

#import functions specific to this model
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
import numpy as np

#create datasets
trainval,test=create_testdata()
X,y=preprocessing(trainval,'DTU',False)


kfold=10 #default 10-fold cross validation

# specify parameters to compute grid search on the the parameter scoring function
from scipy.stats import randint as sp_randint
# specify parameters and distributions to sample from
param_dist = {"plsregression__n_components": sp_randint(2, 9),
              "plsregression__scale": [True, False]}


#hard to impute by country mean so compromise is to have an overall mean strategy
imputer=fill_nan()
pls = PLSRegression()
pipeline=make_pipeline(imputer,pls)

# Choose cross-validation techniques for the inner and outer loops,
# independently of the dataset.
# E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
inner_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)
outer_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)
# run randomized search
n_iter_search = 5
clf = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist,n_iter=n_iter_search,scoring='neg_mean_squared_error', cv=inner_cv)
# Nested CV with parameter optimization
nested_score = cross_val_score(clf, X, y, scoring='neg_mean_squared_error',cv=outer_cv)

clf.fit(X,y)

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
report(clf.cv_results_)

print("Best model has mse of {0:6f}.".format(nested_score.mean()))
print(clf.best_params_)

#now predict the windspeed for test set and output final error (mse)
X_test,y_test=preprocessing(test,'DTU',False)
imputer.fit(X)
X_test=imputer.transform(X_test)
y_predict=clf.predict(X_test)
print(y_predict,y_test)
final_error=-np.sum(np.square(y_predict-y_test))/(len(y_test))
print(final_error)



