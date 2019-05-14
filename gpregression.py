#gaussian process regression
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

subset_data_x=features_train.iloc[0:5000,:-8]
subset_data_y=windspeed_train[0:5000]

#also clip out the one hote encoding variables 

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,WhiteKernel,Matern,RationalQuadratic,ExpSineSquared,DotProduct

#kernel = RBF(2, (1e-23, 1e10) ) + WhiteKernel(1, (1e-23, 1e5))
#         WhiteKernel(2, (1e-23, 1e5))
#np.ones(subset_data_x.shape[1])
#kernel=3.0 * Matern(length_scale=1, length_scale_bounds=(1e-1, 10.0),nu=1.5) + WhiteKernel(1, (1e-23, 1e5))
kernel=1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)+ WhiteKernel(1, (1e-23, 1e5))
#kernel=1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,length_scale_bounds=(0.1, 10.0),periodicity_bounds=(1.0, 10.0))+ WhiteKernel(1, (1e-23, 1e5))
#kernel= 3* (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2)+ WhiteKernel(1, (1e-23, 1e5))

#default kernel
gpr = GaussianProcessRegressor(random_state=0,n_restarts_optimizer=2,kernel=kernel)

gpr.fit(subset_data_x,subset_data_y)

y_predict=gpr.predict(features_test.iloc[:,:-8])
final_error=-np.sum(np.square(y_predict-windspeed_test))/len(windspeed_test)
print('mse error on test set is {}.'.format(final_error))

print(gpr.kernel_)