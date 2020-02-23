import csv
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer

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

df_industry_matrix = pandas.DataFrame()
df_industry_matrix['Industry'] = pandas.Series(['Agri', 'Energy', 'Real Estate', 'Tech', 'Healthcare', 'Manufacturing'])

SARS_slopes = []
SARS_slopes.append(get_slope('Data/Agri_SARS.csv'))
SARS_slopes.append(get_slope('Data/Energy_SARS.csv'))
SARS_slopes.append(get_slope('Data/RealEstate_SARS.csv'))
SARS_slopes.append(get_slope('Data/Tech_SARS.csv'))
SARS_slopes.append(get_slope('Data/Healthcare_SARS.csv'))
SARS_slopes.append(get_slope('Data/Manufacturing_SARS.csv'))

df_industry_matrix['SARS Slope'] = pandas.Series(SARS_slopes)

FLU_slopes = []
FLU_slopes.append(get_slope('Data/Agri_FLU.csv'))
FLU_slopes.append(get_slope('Data/Energy_FLU.csv'))
FLU_slopes.append(get_slope('Data/RealEstate_FLU.csv'))
FLU_slopes.append(get_slope('Data/Tech_FLU.csv'))
FLU_slopes.append(get_slope('Data/Healthcare_FLU.csv'))
FLU_slopes.append(get_slope('Data/Manufacturing_FLU.csv'))

df_industry_matrix['Flu Slopes'] = pandas.Series(FLU_slopes)

TOTAL_slopes = []
TOTAL_slopes.append(get_slope('Data/Agricultural_total.csv'))
TOTAL_slopes.append(get_slope('Data/Energy_total.csv'))
TOTAL_slopes.append(get_slope('Data/RealEstate_total.csv'))
TOTAL_slopes.append(get_slope('Data/Tech_total.csv'))
TOTAL_slopes.append(get_slope('Data/Healthcare_total.csv'))
TOTAL_slopes.append(get_slope('Data/Manufacturing_total.csv'))

x = df_industry_matrix['Flu Slopes'].to_numpy().reshape(1,-1)
norm = Normalizer().fit(x)
norm_flu = norm.transform(x)[0]

y = df_industry_matrix['SARS Slope'].to_numpy().reshape(1,-1)
norm = Normalizer().fit(y)
norm_sars = norm.transform(y)[0]

avg_slope_disease = (norm_flu + norm_sars)/2

total_np = np.asarray(TOTAL_slopes).reshape(1,-1)
norm = Normalizer().fit(total_np)
norm_overall = norm.transform(total_np)[0]

df_norm_industry = pandas.DataFrame()
df_norm_industry['Industry'] = pandas.Series(['Agri', 'Energy', 'Real Estate', 'Tech', 'Healthcare', 'Manufacturing'])

df_norm_industry['SARS Slope'] = pandas.Series(norm_sars)
df_norm_industry['Flu Slope'] = pandas.Series(norm_flu)
df_norm_industry['Average Slope Disease'] = pandas.Series(avg_slope_disease)
df_norm_industry['Control'] =  pandas.Series(norm_overall)

print(df_norm_industry) 