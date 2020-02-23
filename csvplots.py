import csv
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def get_slope(filename):
    df_ETF = pandas.read_csv(filename)
    ds_ETF_Close = df_ETF['Close']
    ds_time = pandas.Series(list(range(len(df_ETF))))

    df_ETF_Final = pandas.DataFrame()
    df_ETF_Final['Time'] = ds_time/len(df_ETF)
    df_ETF_Final['Close'] = ds_ETF_Close

    X = df_ETF_Final.iloc[:, 0].values.reshape(-1, 1)  
    Y = df_ETF_Final.iloc[:, 1].values.reshape(-1, 1)  
    linear_regressor = LinearRegression()  
    linear_regressor.fit(X, Y)
    Y_pred = linear_regressor.predict(X) 

    return float(Y_pred[len(Y_pred) - 1] - Y_pred[0])

def get_all_data(filename):
    df_ETF = pandas.read_csv(filename)
    ds_ETF_Close = df_ETF['Close']
    ds_time = pandas.Series(list(range(len(df_ETF))))

    df_ETF_Final = pandas.DataFrame()
    df_ETF_Final['Time'] = ds_time/len(df_ETF)
    df_ETF_Final['Close'] = ds_ETF_Close

    X = df_ETF_Final.iloc[:, 0].values.reshape(-1, 1)  
    Y = df_ETF_Final.iloc[:, 1].values.reshape(-1, 1)  
    linear_regressor = LinearRegression()  
    linear_regressor.fit(X, Y)
    Y_pred = linear_regressor.predict(X) 

    return [X, Y, Y_pred]

df_industry_matrix = pandas.DataFrame()
df_industry_matrix['Industry'] = pandas.Series(['Tech', 'Agri', 'Real Estate', 'Energy'])

SARS_slopes = []
SARS_slopes.append(get_slope('Data/Agri_SARS.csv'))
SARS_slopes.append(get_slope('Data/Energy_SARS.csv'))
SARS_slopes.append(get_slope('Data/RealEstate_SARS.csv'))
SARS_slopes.append(get_slope('Data/Tech_SARS.csv'))

df_industry_matrix['SARS Slope'] = pandas.Series(SARS_slopes)

FLU_slopes = []
FLU_slopes.append(get_slope('Data/Agri_FLU.csv'))
FLU_slopes.append(get_slope('Data/Energy_FLU.csv'))
FLU_slopes.append(get_slope('Data/RealEstate_FLU.csv'))
FLU_slopes.append(get_slope('Data/Tech_FLU.csv'))

df_industry_matrix['Slope FLU'] = pandas.Series(FLU_slopes)

print(df_industry_matrix)

