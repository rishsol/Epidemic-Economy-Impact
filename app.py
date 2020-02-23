import pickle
import copy
import pathlib
import dash
import math
import datetime as dt
import pandas
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.tools as tls
import plotly.graph_objs as go
import statistics
from sklearn.linear_model import LinearRegression
import numpy as np
from plotly.subplots import make_subplots
import base64

app = dash.Dash(__name__)
server = app.server

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

    df_formated = pandas.DataFrame()
    x_data = []
    y_data = []
    ypred_data = []

    for data in X:
        x_data.append(float(data))

    df_formated['X'] = pandas.Series(x_data)

    for data in Y:
        y_data.append(float(data))
    
    df_formated['Y'] = pandas.Series(y_data)

    for data in Y_pred:
        ypred_data.append(float(data))
    
    df_formated['YPred'] = pandas.Series(ypred_data)

    return df_formated

df_total_tech = get_all_data('Data/Tech_total.csv')
df_flu_tech = get_all_data('Data/Tech_FLU.csv')
df_sars_tech = get_all_data('Data/Tech_SARS.csv')

fig1 = make_subplots(rows=2,cols=1,
    subplot_titles=('Tech ETF (2000-2010)', 'Tech ETF during SARS and Swine Flu Epidemic'))
fig1.add_trace(go.Scatter(x=df_total_tech['X'], y=df_total_tech['Y'],
                    mode='markers',
                    name='Actual Total Tech'),
                    row=1,col=1)
fig1.add_trace(go.Scatter(x=df_total_tech['X'], y=df_total_tech['YPred'],
                    mode='markers',
                    name='Regression Total',
                    marker_color='lightblue'),
                    row=1,col=1)

fig1.add_trace(go.Scatter(x=df_flu_tech['X'], y=df_flu_tech['Y'],
                    mode='markers',
                    name='Actual Flu Tech',
                    marker_color='green'),
                    row=2,col=1)
fig1.add_trace(go.Scatter(x=df_flu_tech['X'], y=df_flu_tech['YPred'],
                    mode='markers',
                    name='Regression Flu Tech',
                    marker_color='lightgreen'),
                    row=2,col=1)
fig1.add_trace(go.Scatter(x=df_sars_tech['X'], y=df_sars_tech['Y'],
                    mode='markers',
                    name='Actual SARS Tech',
                    marker_color='red'),
                    row=2,col=1)
fig1.add_trace(go.Scatter(x=df_sars_tech['X'], y=df_sars_tech['YPred'],
                    mode='markers',
                    name='Regression SARS Tech',
                    marker_color='pink'),
                    row=2,col=1)

fig1.update_yaxes(title_text="USD", row=1, col=1)
fig1.update_xaxes(title_text='Proportion of epidemic completed', row=2, col=1)
fig1.update_yaxes(title_text="USD", row=2, col=1)

df_total_agri = get_all_data('Data/Agricultural_total.csv')
df_flu_agri = get_all_data('Data/Agri_FLU.csv')
df_sars_agri = get_all_data('Data/Agri_SARS.csv')

fig2 = make_subplots(rows=2,cols=1,
    subplot_titles=('Agriculture ETF (2000-2010)', 'Agriculture ETF during SARS and Swine Flu Epidemic'))
fig2.add_trace(go.Scatter(x=df_total_agri['X'], y=df_total_agri['Y'],
                    mode='markers',
                    name='Actual Total Agriculture'),
                    row=1,col=1)
fig2.add_trace(go.Scatter(x=df_total_agri['X'], y=df_total_agri['YPred'],
                    mode='markers',
                    name='Regression Total Agriculture',
                    marker_color='lightblue'),
                    row=1,col=1)
fig2.add_trace(go.Scatter(x=df_flu_agri['X'], y=df_flu_agri['Y'],
                    mode='markers',
                    name='Actual Flu Agriculture',
                    marker_color='green'),
                    row=2,col=1)
fig2.add_trace(go.Scatter(x=df_flu_agri['X'], y=df_flu_agri['YPred'],
                    mode='markers',
                    name='Regression Flu Agriculture',
                    marker_color='lightgreen'),
                    row=2,col=1)
fig2.add_trace(go.Scatter(x=df_sars_agri['X'], y=df_sars_agri['Y'],
                    mode='markers',
                    name='Actual SARS Agriculture',
                    marker_color='red'),
                    row=2,col=1)
fig2.add_trace(go.Scatter(x=df_sars_agri['X'], y=df_sars_agri['YPred'],
                    mode='markers',
                    name='Agriculture Regression SARS',
                    marker_color='pink'),
                    row=2,col=1)

df_total_energy = get_all_data('Data/Energy_total.csv')
df_flu_energy = get_all_data('Data/Energy_FLU.csv')
df_sars_energy = get_all_data('Data/Energy_SARS.csv')

fig3 = make_subplots(rows=2,cols=1,
    subplot_titles=('Energy ETF (2000-2010)', 'Energy ETF during SARS and Swine Flu Epidemic'))
fig3.add_trace(go.Scatter(x=df_total_energy['X'], y=df_total_energy['Y'],
                    mode='markers',
                    name='Actual Total Energy'),
                    row=1,col=1)
fig3.add_trace(go.Scatter(x=df_total_energy['X'], y=df_total_energy['YPred'],
                    mode='markers',
                    name='Regression Total Energy',
                    marker_color='lightblue'),
                    row=1,col=1)

fig3.add_trace(go.Scatter(x=df_flu_energy['X'], y=df_flu_energy['Y'],
                    mode='markers',
                    name='Actual Flu Energy',
                    marker_color='green'),
                    row=2,col=1)
fig3.add_trace(go.Scatter(x=df_flu_energy['X'], y=df_flu_energy['YPred'],
                    mode='markers',
                    name='Regression Flu Energy',
                    marker_color='lightgreen'),
                    row=2,col=1)
fig3.add_trace(go.Scatter(x=df_sars_energy['X'], y=df_sars_energy['Y'],
                    mode='markers',
                    name='Actual SARS Energy',
                    marker_color='red'),
                    row=2,col=1)
fig3.add_trace(go.Scatter(x=df_sars_energy['X'], y=df_sars_energy['YPred'],
                    mode='markers',
                    name='Regression SARS Energy',
                    marker_color='pink'),
                    row=2,col=1)

df_total_health = get_all_data('Data/Healthcare_total.csv')
df_flu_health = get_all_data('Data/Healthcare_FLU.csv')
df_sars_health = get_all_data('Data/Healthcare_SARS.csv')

fig4 = make_subplots(rows=2,cols=1,
    subplot_titles=('Healthcare ETF (2000-2010)', 'Healthcare ETF during SARS and Swine Flu Epidemic'))
fig4.add_trace(go.Scatter(x=df_total_health['X'], y=df_total_health['Y'],
                    mode='markers',
                    name='Actual Total Healthcare'),
                    row=1,col=1)
fig4.add_trace(go.Scatter(x=df_total_health['X'], y=df_total_health['YPred'],
                    mode='markers',
                    name='Regression Total Healthcare',
                    marker_color='lightblue'),
                    row=1,col=1)
fig4.add_trace(go.Scatter(x=df_flu_health['X'], y=df_flu_health['Y'],
                    mode='markers',
                    name='Actual Flu Healthcare',
                    marker_color='green'),
                    row=2,col=1)
fig4.add_trace(go.Scatter(x=df_flu_health['X'], y=df_flu_health['YPred'],
                    mode='markers',
                    name='Regression Flu Healthcare',
                    marker_color='lightgreen'),
                    row=2,col=1)
fig4.add_trace(go.Scatter(x=df_sars_health['X'], y=df_sars_health['Y'],
                    mode='markers',
                    name='Actual SARS Healthcare',
                    marker_color='red'),
                    row=2,col=1)
fig4.add_trace(go.Scatter(x=df_sars_health['X'], y=df_sars_health['YPred'],
                    mode='markers',
                    name='Regression SARS Healthcare',
                    marker_color='pink'),
                    row=2,col=1)

df_total_manu= get_all_data('Data/Manufacturing_total.csv')
df_flu_manu = get_all_data('Data/Manufacturing_FLU.csv')
df_sars_manu = get_all_data('Data/Manufacturing_SARS.csv')

fig5 = make_subplots(rows=2,cols=1,
    subplot_titles=('Manufacturing ETF (2000-2010)', 'Manufacturing ETF during SARS and Swine Flu Epidemic'))
fig5.add_trace(go.Scatter(x=df_total_manu['X'], y=df_total_manu['Y'],
                    mode='markers',
                    name='Actual Total Manufacturing'),
                    row=1,col=1)
fig5.add_trace(go.Scatter(x=df_total_manu['X'], y=df_total_manu['YPred'],
                    mode='markers',
                    name='Regression Total Manufacturing',
                    marker_color='lightblue'),
                    row=1,col=1)
fig5.add_trace(go.Scatter(x=df_flu_manu['X'], y=df_flu_manu['Y'],
                    mode='markers',
                    name='Actual Flu Manufacturing',
                    marker_color='green'),
                    row=2,col=1)
fig5.add_trace(go.Scatter(x=df_flu_manu['X'], y=df_flu_manu['YPred'],
                    mode='markers',
                    name='Regression Flu Manufacturing',
                    marker_color='lightgreen'),
                    row=2,col=1)
fig5.add_trace(go.Scatter(x=df_sars_manu['X'], y=df_sars_manu['Y'],
                    mode='markers',
                    name='Actual SARS Manufacturing',
                    marker_color='red'),
                    row=2,col=1)
fig5.add_trace(go.Scatter(x=df_sars_manu['X'], y=df_sars_manu['YPred'],
                    mode='markers',
                    name='Regression SARS Manufacturing',
                    marker_color='pink'),
                    row=2,col=1)

df_total_re= get_all_data('Data/RealEstate_total.csv')
df_flu_re = get_all_data('Data/RealEstate_FLU.csv')
df_sars_re = get_all_data('Data/RealEstate_SARS.csv')

fig6 = make_subplots(rows=2,cols=1,
    subplot_titles=('Real Estate ETF (2000-2010)', 'Real Estate ETF during SARS and Swine Flu Epidemic'))
fig6.add_trace(go.Scatter(x=df_total_re['X'], y=df_total_re['Y'],
                    mode='markers',
                    name='Actual Total Real Estate'),
                    row=1,col=1)
fig6.add_trace(go.Scatter(x=df_total_re['X'], y=df_total_re['YPred'],
                    mode='markers',
                    name='Regression Total Real Estate',
                    marker_color='lightblue'),
                    row=1,col=1)
fig6.add_trace(go.Scatter(x=df_flu_re['X'], y=df_flu_re['Y'],
                    mode='markers',
                    name='Actual Flu Real Estate',
                    marker_color='green'),
                    row=2,col=1)
fig6.add_trace(go.Scatter(x=df_flu_re['X'], y=df_flu_re['YPred'],
                    mode='markers',
                    name='Regression Flu Real Estate',
                    marker_color='lightgreen'),
                    row=2,col=1)
fig6.add_trace(go.Scatter(x=df_sars_re['X'], y=df_sars_re['Y'],
                    mode='markers',
                    name='Actual SARS Real Estate',
                    marker_color='red'),
                    row=2,col=1)
fig6.add_trace(go.Scatter(x=df_sars_re['X'], y=df_sars_re['YPred'],
                    mode='markers',
                    name='Regression SARS Real Estate',
                    marker_color='pink'),
                    row=2,col=1)


app.layout = html.Div(children=[
    html.H1(children='Effects of Epidemics on Various ETFs'),
    dcc.Graph(
        id = 'tech',
        figure = fig1
    ),
    dcc.Graph(
        id = 'agri',
        figure = fig2
    ),
    dcc.Graph(
        id = 'energy',
        figure = fig3
    ),
    dcc.Graph(
        id = 'healthcare',
        figure = fig4
    ),
    dcc.Graph(
        id = 'manu',
        figure = fig5
    ),
    dcc.Graph(
        id = 're',
        figure = fig6
    ),
    html.Div(
        [
            html.Img(
                src=app.get_asset_url('AggregateSlope.png'),
                id="agg-slope",
                style= {
                    'display': 'block',
                    'height': '360px',
                    'width': 'auto',
                    'margin-left': 'auto',
                    'margin-right': 'auto'
                }
            )
        ]
    ),
    html.Div(
        [
            html.P('The above graph gives normalize rates of change of various ETFs during 2000-2010 and during the SARS and Swine flu outbreaks. Agriculture and Energy did relatively worse over the course of the two epidemics than real estate, tech, healthcare, and manufacturing',
                style= {
                    'align': 'center'
                }
            ),
            html.P('To maintain financial security, a potential investor would want to avoid investing in the first two industries during a time of plague',
                style= {
                    'align': 'center'
                }
            )
        ]
    ),
    html.Div(
        [
            html.Img(
                src=app.get_asset_url('EpidemicVar.png'),
                id="var",
                style= {
                    'display': 'block',
                    'height': '360px',
                    'width': 'auto',
                    'margin-left': 'auto',
                    'margin-right': 'auto'
                }
            )
        ]
    ),
    html.Div(
        [
            html.P('A negative value indicates that rate of growth of the corresponding ETF dropped during time of epidemic and vice versa',
                style= {
                    'align': 'center'
                }
            ),
        ]
    ),
])

if __name__ == '__main__':
    app.run_server(debug=True)

