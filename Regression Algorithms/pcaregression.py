#pca regression
from sklearn.linear_model import LinearRegression
#import functions needed to set up the dataset
from prepare_data import create_testdata
from preprocessing import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
import numpy as np

#create datasets
df=preprocessing()
features_train,windspeed_train,features_test,windspeed_test=create_testdata(df,'50')

#seperate out the one-hot encoding variables before standardization 
df1 = features_train.iloc[:, :17]
df2 = features_train.iloc[:, 17:]
df3 = features_test.iloc[:, :17]
df4 = features_test.iloc[:, 17:]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df1)
dftrain=scaler.transform(df1)
dftest=scaler.transform(df3)

df1=pd.DataFrame(data=dftrain,index=df1.index.values,columns=df1.columns.values)
df3=pd.DataFrame(data=dftest,index=df3.index.values,columns=df3.columns.values)

features_train=pd.concat([df1,df2], axis=1)
features_test=pd.concat([df3,df4], axis=1)

kfold=10 #default 10-fold cross validation

pca=PCA()
reg = LinearRegression()
pipeline=make_pipeline(pca,reg)
# specify parameters to compute grid search on the the parameter scoring function
param_dist = {"alpha": [1e-3, 1e-2, 1e-1, 1]}

# Choose cross-validation techniques for the inner and outer loops,
# independently of the dataset.
# E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
inner_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)
outer_cv = KFold(n_splits=kfold, shuffle=True, random_state=1)

clf = GridSearchCV(estimator=pipeline, param_grid=param_dist,scoring='neg_mean_squared_error', cv=inner_cv)
# Nested CV with parameter optimization
nested_score = cross_val_score(clf, features_train, windspeed_train, scoring='neg_mean_squared_error',cv=outer_cv)

clf.fit(features_train,windspeed_train)

print("Best model has mse of {} with standard deviation {}.".format(nested_score.mean(),nested_score.std()))
print(clf.best_params_)

np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

#now predict the windspeed for test set and output final error (mse)

y_predict=clf.predict(features_test)
final_error=-np.sum(np.square(y_predict-windspeed_test))/len(windspeed_test)
print('mse error on test set is {}.'.format(final_error))