from sklearn.linear_model import Ridge
#import functions needed to set up the dataset
from prepare_data import create_testdata
from preprocessing import preprocessing

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
import numpy as np

#create datasets
df=preprocessing()
features_train,windspeed_train,features_test,windspeed_test=create_testdata(df,'200')

kfold=10 #default 10-fold cross validation

ridge=Ridge(alpha=.1)
# specify parameters to compute grid search on the the parameter scoring function
param_dist = {"alpha": [0.05, 0.1, 0.15]}

# Choose cross-validation techniques for the inner and outer loops,
# independently of the dataset.
# E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
inner_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)
outer_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)

clf = GridSearchCV(estimator=ridge, param_grid=param_dist,scoring='neg_mean_squared_error', cv=inner_cv)
# Nested CV with parameter optimization
nested_score = cross_val_score(clf, features_train, windspeed_train, scoring='neg_mean_squared_error',cv=outer_cv)

clf.fit(features_train,windspeed_train)

print("Best model has mse of {} with standard deviation {}.".format(nested_score.mean(),nested_score.std()))
print(clf.best_params_)

#now predict the windspeed for test set and output final error (mse)

y_predict=clf.predict(features_test)
final_error=-np.sum(np.square(y_predict-windspeed_test))/len(windspeed_test)
print('mse error on test set is {}.'.format(final_error))
