#main processing function that drops unnecesarry columns and combine features to give the desired output X (features to train on) and y (windspeed to regress on)
#can either extract GWS wind speed data (less variant) or DTU wind speed data as the target variable and there is also the option to include transmission data or not

def create_testdata(mode,strategy):
    import pandas as pd 
    import numpy as np
    if strategy == 1:
        df_input = pd.read_csv("Data/strategy1.csv", sep=",",low_memory=False)
    elif strategy == 2:
        df_input = pd.read_csv("Data/strategy2.csv", sep=",",low_memory=False)
    else:
        raise Exception('There is no such strategy as: {}'.format(strategy))
    df=df_input.drop(['Unnamed: 0'],axis=1)
    df['Lat']=df['Latitude(y)']
    df['Long']=df['Longitude(x)']
    df['50']=df['Windspeed_50 (m/s)']
    df['100']=df['Windspeed_100 (m/s)']
    df['200']=df['Windspeed_200 (m/s)']   
    df=df.drop(['Windspeed_50 (m/s)','Windspeed_100 (m/s)','Windspeed_200 (m/s)','Latitude(y)','Longitude(x)'],axis=1)
    df=df.drop(['Wind power density_100m(W/m2 )'],axis=1)
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
    elif mode=='200':
        windspeed_train = train.pop('200').values
        features_train=train.drop(['50','100'],axis=1)
        windspeed_test = test.pop('200').values
        features_test=test.drop(['50','100'],axis=1)
    else:
        raise Exception('There is no such mode as: {}'.format(mode))
    return features_train,windspeed_train,features_test,windspeed_test