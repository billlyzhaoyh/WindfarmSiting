#scratch1
#import functions needed to set up the dataset
from fillNAN import fill_nan
from prepare_data import create_testdata
from preprocessing import preprocessing
import numpy as np
df=preprocessing()
India=df.loc[df['Country'] == 'India']
Italy=df.loc[df['Country'] == 'Italy']
France=df.loc[df['Country'] == 'France']
features_train,windspeed_train,features_test,windspeed_test=create_testdata(India,'200')
col_todrop=['Lat','Long','India']
features_train=features_train.drop(col_todrop,axis=1)
features_test=features_test.drop(col_todrop,axis=1)
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
kfold=10
cv = KFold(n_splits=kfold, shuffle=True, random_state=1)
lm = linear_model.LinearRegression()
lm.fit(features_train,windspeed_train)
nested_score = cross_val_score(lm, features_train, windspeed_train, scoring='neg_mean_squared_error',cv=cv)
print("Best model has mse of {} with standard deviation {}.".format(nested_score.mean(),nested_score.std()))
y_predict=lm.predict(features_test)
final_error=-np.sum(np.square(y_predict-windspeed_test))/len(windspeed_test)
print('mse error on test set is {}.'.format(final_error))
#feature importance 
import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(lm, random_state=1).fit(features_test, windspeed_test)
explaination=eli5.explain_weights_df(perm, feature_names = features_test.columns.tolist(),top=10)
print(explaination)