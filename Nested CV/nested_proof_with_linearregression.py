#linear regression 

#import functions needed to set up the dataset
from fillNAN import fill_nan
from prepare_data import create_testdata
from preprocessing import preprocessing

#import functions specific to this model
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate, KFold
from sklearn.pipeline import make_pipeline
import numpy as np
from matplotlib import pyplot as plt

#create datasets
trainval,test=create_testdata()
X,y=preprocessing(trainval,'DTU',False)

# Number of random trials
NUM_TRIALS = 30

kfold=10 #default 10-fold cross validation

# specify parameters to compute grid search on the the parameter scoring function
param_dist = {"linearregression__normalize": [True, False]}
scoring = {'mse': 'neg_mean_squared_error', 'corr': 'r2'}

#hard to impute by country mean so compromise is to have an overall mean strategy
imputer=fill_nan()
lm = linear_model.LinearRegression()
pipeline=make_pipeline(imputer,lm)

# Arrays to store scores
non_nested_scores = np.zeros(NUM_TRIALS) #only mse
nested_scores = np.zeros((NUM_TRIALS,2))

# Loop for each trial
for i in range(NUM_TRIALS):

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    inner_cv = KFold(n_splits=kfold, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=kfold, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=pipeline, param_grid=param_dist,scoring=scoring, cv=inner_cv,refit='mse')
    clf.fit(X, y)
    non_nested_scores[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score = cross_validate(clf, X=X,y=y,cv=outer_cv,scoring=scoring)
    nested_scores[i,0] = nested_score['test_mse'].mean()
    nested_scores[i,1] = nested_score['test_corr'].mean()

score_difference = non_nested_scores - nested_scores[:,0]

print("Average difference of {0:6f} with std. dev. of {1:6f}.".format(score_difference.mean(), score_difference.std()))

# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
nested_line, = plt.plot(nested_scores[:,0], color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation Comparison over 30 trials",
          x=.5, y=1.1, fontsize="15")
# Plot bar chart of the difference.
plt.subplot(212)
difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
plt.xlabel("Individual Trial #")
plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
plt.ylabel("score difference", fontsize="14")

plt.show()
