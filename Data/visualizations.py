import csv
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

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

a_list = get_all_data('Data/Tech_total.csv')
plt.scatter(a_list[0], a_list[1])
plt.plot(a_list[0], a_list[2], color='blue')

a_list = get_all_data('Data/Tech_FLU.csv')
plt.scatter(a_list[0], a_list[1])
plt.plot(a_list[0], a_list[2], color='orange')

a_list = get_all_data('Data/Tech_SARS.csv')
plt.scatter(a_list[0], a_list[1])
plt.plot(a_list[0], a_list[2], color='green')

plt.show()
