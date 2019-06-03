#prepare the dataset for analysis

#lock away 10% data that is not used by the model for training and hyperparameter tuning. Split data points based on contry to obtain stratified samples
import pandas as pd 
import numpy as np

def split_testdata(df):
	df_country=df.groupby('Country',as_index=False)
	dfObj = pd.DataFrame(columns=df.columns.values)
	#take 10% at random and lock it away 
	for name, group in df_country:
	    sample01=group.sample(int(np.floor(len(group)*0.1)),random_state=0)
	    dfObj = dfObj.append(sample01, ignore_index=False)
	df1=pd.concat([df, dfObj]).drop_duplicates(keep=False)
	#join the trainset and test set back together and do one-hot encoding
	total_set=pd.concat((df1, dfObj),keys=('train','test'))
	one_hot=pd.get_dummies(total_set['Country'])
	total_set=total_set.drop('Country',axis = 1)
	total_set=total_set.join(one_hot)
	train=total_set.loc['train']
	test=total_set.loc['test']
	return train,test


def create_testdata(df,mode):
	import pandas as pd 
	import numpy as np
	df_country=df.groupby('Country',as_index=False)
	dfObj = pd.DataFrame(columns=df.columns.values)
	#take 10% at random and lock it away 
	for name, group in df_country:
	    sample01=group.sample(int(np.floor(len(group)*0.1)),random_state=0)
	    dfObj = dfObj.append(sample01, ignore_index=False)
	df1=pd.concat([df, dfObj]).drop_duplicates(keep=False)
	#join the trainset and test set back together and do one-hot encoding
	total_set=pd.concat((df1, dfObj),keys=('train','test'))
	one_hot=pd.get_dummies(total_set['Country'])
	total_set=total_set.drop('Country',axis = 1)
	total_set=total_set.join(one_hot)
	train=total_set.loc['train']
	test=total_set.loc['test']
	if mode=='50':
		windspeed_train = train.pop('50').values
		features_train=train.drop(['100','200'],axis=1)
		windspeed_test = test.pop('50').values
		features_test=test.drop(['100','200'],axis=1)
	elif mode=='100':
		windspeed_train = train.pop('100').values
		features_train=train.drop(['50','200'],axis=1)
		windspeed_test = test.pop('100').values
		features_test=test.drop(['50','200'],axis=1)
	else:
		windspeed_train = train.pop('200').values
		features_train=train.drop(['50','100'],axis=1)
		windspeed_test = test.pop('200').values
		features_test=test.drop(['50','100'],axis=1)
	return features_train,windspeed_train,features_test,windspeed_test





