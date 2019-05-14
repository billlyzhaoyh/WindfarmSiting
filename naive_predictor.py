#a naive predictor that predicts mean value of wind speed for each country
#import functions needed to set up the dataset
from fillNAN import fill_nan
from prepare_data import create_testdata
from preprocessing import preprocessing
import numpy as np
import pandas as pd

#create datasets
trainval,test=create_testdata()
X,y=preprocessing(trainval,'DTU',False)

print(X.columns)
#print(X['country'].value_count())