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
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

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

#hard to impute by country mean so compromise is to have an overall mean strategy

poly = PolynomialFeatures(degree=2)
lm = linear_model.LinearRegression()
pipeline=make_pipeline(poly,lm)

# Choose cross-validation techniques for the inner and outer loops,
# independently of the dataset.
# E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.

cv = KFold(n_splits=kfold, shuffle=True, random_state=1)

# Nested CV with parameter optimization
nested_score = cross_val_score(pipeline, features_train, windspeed_train, scoring='neg_mean_squared_error',cv=cv)


#estimated training error
print("Model is estimated to have generalisation error of {} with standard deviation {}.".format(nested_score.mean(),nested_score.std()))

pipeline.fit(features_train,windspeed_train)

#regr=pipeline.named_steps['lm']

#now predict the windspeed for test set and output final error (mse)
windspeed_predict=pipeline.predict(features_test)
final_error=-np.sum(np.square(windspeed_predict-windspeed_test))/len(windspeed_test)
print(final_error)



