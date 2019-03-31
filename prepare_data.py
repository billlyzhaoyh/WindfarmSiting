#prepare the dataset for analysis

#lock away 10% data that is not used by the model for training and hyperparameter tuning. Split data points based on contry to obtain stratified samples

def create_testdata():
	import pandas as pd 
	import numpy as np

	df = pd.read_csv("Data/v1.csv", sep=",")
	df_country=df.groupby('Country',as_index=False)
	dfObj = pd.DataFrame(columns=df.columns.values)
	#take 10% at random and lock it away 
	for name, group in df_country:
	    sample01=group.sample(int(np.floor(len(group)*0.1)))
	    dfObj = dfObj.append(sample01, ignore_index=False)
	df1=pd.concat([df, dfObj]).drop_duplicates(keep=False)
	return df1,dfObj

#implement nested CV



