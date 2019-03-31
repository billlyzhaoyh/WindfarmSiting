#main processing function that drops unnecesarry columns and combine features to give the desired output X (features to train on) and y (windspeed to regress on)
#can either extract GWS wind speed data (less variant) or DTU wind speed data as the target variable and there is also the option to include transmission data or not

def preprocessing(df_input,mode,Transmission_line):
    #need to exclude data with transmission line and build models specifically for that
    import pandas as pd
    filtered_df = df_input[df_input['Distance to the closest Transmission line (kms)'].notnull()]
    df_new=pd.concat([df_input,filtered_df]).drop_duplicates(keep=False)
    if Transmission_line is True:
        df=filtered_df
    else:
        df=df_new
    if mode == 'DTU':
        #dataset1 with DTU
        DTU=df.drop(['GlobalWS80 (m/s)'],axis=1)
        #combine the three wind speeds into one by simple averaging
        col_drop=['DTU_50m_WindSp(m/s)','DTU_100m_WindSp(m/s)','DTU_200m_WindSp(m/s)']
        DTU=DTU.dropna(axis=0,subset=col_drop)
        DTU['DTU'] = (DTU['DTU_50m_WindSp(m/s)'] + DTU['DTU_100m_WindSp(m/s)'] + DTU['DTU_200m_WindSp(m/s)'])/3
        DTU1=DTU.drop(['DTU_50m_WindSp(m/s)','DTU_100m_WindSp(m/s)','DTU_200m_WindSp(m/s)','Altitude','Name','Longitude(x)','Latitude(y)','Distance between turbines (in decimal degrees)','Unnamed: 25','Unnamed: 26','meters','Kilometers','ID'],axis=1)
        #drop rows where all the data points are NaN     
        DTU1=DTU1.dropna(axis=1,how='all')
        #one-hot encoding for Country data
        one_hot=pd.get_dummies(DTU1['Country'])
        DTU1=DTU1.drop('Country',axis = 1)
        DTU1=DTU1.join(one_hot)
        features=DTU1
        windspeed = DTU1.pop('DTU').values
    elif mode == 'GWS':
        #data for model 2
        GLOBAL=df.drop(['DTU_50m_WindSp(m/s)','DTU_100m_WindSp(m/s)','DTU_200m_WindSp(m/s)','Altitude','Name','Longitude(x)','Latitude(y)','Distance between turbines (in decimal degrees)','Unnamed: 25','Unnamed: 26','meters','Kilometers','ID'],axis=1)
        col_drop=['GlobalWS80 (m/s)']
        GLOBAL=GLOBAL.dropna(axis=0,subset=col_drop) 
        GLOBAL['GWS']=GLOBAL['GlobalWS80 (m/s)']
        GLOBAL1=GLOBAL.drop(col_drop,axis=1)
        #drop the data points where GWS = 0
        GLOBAL1 = GLOBAL1[GLOBAL1.GWS != 0]
        #drop rows where all the data points are NaN     
        GLOBAL1=GLOBAL1.dropna(axis=1,how='all')
        #one-hot encoding for Country data
        one_hot=pd.get_dummies(GLOBAL1['Country'])
        GLOBAL1=GLOBAL1.drop('Country',axis = 1)
        GLOBAL1=GLOBAL1.join(one_hot)
        features=GLOBAL1
        windspeed = GLOBAL1.pop('GWS').values
    return features,windspeed