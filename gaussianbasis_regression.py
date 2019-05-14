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
features_train,windspeed_train,features_test,windspeed_test=create_testdata(df,'50')

from sklearn.base import BaseEstimator, TransformerMixin
class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for multi-dimensional input"""
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        # create N centers spread along the data range
        # create centres along each columns and combine them together
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)

kfold=10 #default 10-fold cross validation
param_dist = {"gaussianfeatures__N": [13,14,15]}
#the linear model
lm = linear_model.LinearRegression()
gauss = GaussianFeatures(10)
pipeline=make_pipeline(gauss,lm)

inner_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)
outer_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)

clf = GridSearchCV(estimator=pipeline, param_grid=param_dist,scoring='neg_mean_squared_error', cv=inner_cv)
nested_score = cross_val_score(pipeline, features_train.values, windspeed_train, scoring='neg_mean_squared_error',cv=outer_cv)
clf.fit(features_train.values,windspeed_train)
print("Best model has mse of {} with standard deviation {}.".format(nested_score.mean(),nested_score.std()))
print(clf.best_params_)

windspeed_predict=clf.predict(features_test.values)
final_error=-np.sum(np.square(windspeed_predict-windspeed_test))/len(windspeed_test)
print(final_error)


