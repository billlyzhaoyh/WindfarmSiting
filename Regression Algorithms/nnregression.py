#neuralnet regression

import pandas as pd
import numpy as np

#import functions needed to set up the dataset
from prepare_data import create_testdata
from preprocessing import preprocessing

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

from keras import Sequential
from keras.layers import Dense
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(25, input_dim=25, kernel_initializer='normal', activation='relu'))
    regressor.add(Dense(25, input_dim=25, kernel_initializer='normal', activation='relu'))
    regressor.add(Dense(1, kernel_initializer='normal'))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    return regressor

from keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=build_regressor, batch_size=32,epochs=100)

results=regressor.fit(features_train,windspeed_train)

y_predict=regressor.predict(features_test)
final_error=-np.sum(np.square(y_predict-windspeed_test))/len(windspeed_test)
print('mse error on test set is {}.'.format(final_error))