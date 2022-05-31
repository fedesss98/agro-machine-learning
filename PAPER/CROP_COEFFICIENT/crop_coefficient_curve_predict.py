#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 21:40:41 2022

@author: Federico Amato

Modelli di ML per predire il coefficiente K = ETa / ET0 .
Si usano sempre i dati del campo in deficit.
Le predizioni sono su una frazione del dataset totale,
data dalla classe "train_test_split".

I risultati non sono buoni.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from MODULES import et_functions as et

def plot_both(ax1, ax2, days, trend, seasonal):
    et.plot_axis(ax1, [days, trend], grid=True, title='Trend')
    et.plot_axis(ax2, [days, seasonal], grid=True, title='Seasonal')

def plot_just_one(ax1, days, y, title):
    et.plot_axis(ax1, [days, y], grid=True, title=title)

MLP_PARAMS = {
    'hidden_layer_sizes': (60,60),
    'activation': 'relu', # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
    'solver': 'adam', #‘lbfgs’, ‘sgd’, ‘adam’
    'random_state': 2,
    }
MODELS = {
    'Multi-Layer Perceptron': MLPRegressor(**MLP_PARAMS),
    'Random Forest': RandomForestRegressor(),
    'Linear Regressor': LinearRegression(),
    'Support Vector Machine': SVR(),
    }
DATABASE = '../../CSV/db_villabate_deficit.csv'
DATAFRAME_PAR = {
    'columns': [
        'θ 10',
        'θ 20',
        'θ 30',
        'θ 40',
        'θ 50',
        'θ 60',
        'U 2m',
        'Rshw dwn',
        'RHmin',
        'RHmax',
        'Tmin',
        'Tmax',
        'ET0',
        'ETa',
        ],
    'start': '2018-01-01',
    'drop_index': True,
    }

YEARS = ['2018','2019','2020']

TO_PLOT = [
        'K predicted',
        'K measured',
        'Potenziali PD',
        'Sap Flow',
    ]

# Si crea il DataFrame complessivo con i dati della coltivazione
df = et.make_dataframe(DATABASE, **DATAFRAME_PAR)
df_measured = et.make_dataframe(DATABASE, **DATAFRAME_PAR)
# L'output richiesto è il coefficiente K = ETa / ET0
k_measured = df_measured['ETa'] / df_measured['ET0']

# Si passa alla matrice X delle Features e vettore y dell'Output
X = df_measured.iloc[:,:-1]
y = k_measured

# Si usa una frazione del set per l'addestramento
# e una per il test e la predizione di K.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
y_predict = dict()
for name, model in MODELS.items():
    # Fitting the model to data
    model.fit(X_train, y_train)
    # Testing the model and taking the score (R^2)
    R2 = model.score(X_test, y_test)
    print(f'Model Score of {name}: {R2}')
    # Predict the set
    y_predict[name] = [model.predict(X_test), R2]

#%% GRAFICI DELLE PREDIZIONI (VALORI MISURATI E PREDETTI)
ax_par = {
    'plot_type': 'scatter',
    'size': 150,
    }
fig_par = {
    'title': 'Coefficient K = ETa/ET0 measured and predicted',
    'xlabel': 'Time',
    'ylabel': 'ETa/ET0',
    'grid': False,
    'date_ticks': 3,
    }
fig, ax = plt.subplots(figsize=(15,8))
ax.set_ylim(0,1.75)
# Grafico dei K misurati usati per l'Addestramento
et.plot_axis(ax, [y_train.index, y_train.values, 'lightgrey'], **ax_par, **fig_par, legend='Train')
# Grafico dei K misurati usati per il Test
et.plot_axis(ax, [y_test.index, y_test.values, 'black'], **ax_par, legend='Test')
# Grafici dei K predetti usando le Features per il test
for name in MODELS.keys():
    legend = f'Prediction with {name} ($R^2 = {y_predict[name][1]:.2f}$)'
    et.plot_axis(ax, [y_test.index, y_predict[name][0]], **ax_par, legend=legend)
ax.legend()

#%% GRAFICI LINEARI DELLE PREDIZIONI
for name in MODELS.keys():
    fig2, ax2 = plt.subplots()
    fig2_par = {
        'title': f'Prediction with {name} ($R^2 = {y_predict[name][1]:.2f}$)',
        'plot_type': 'scatter',
        'xlabel': 'K Measured',
        'ylabel': 'K Predicted',
        'grid': True,
        'size': 80,
        }
    et.plot_axis(ax2, [y_test, y_predict[name][0], 'orange'], **fig2_par)
    x_max = y_test.max()
    y_max = y_predict[name][0].max()
    x_min = y_test.min()
    y_min = y_predict[name][0].min()
    et.plot_axis(ax2, [np.linspace(x_min,x_max),np.linspace(y_min,y_max), 'red'], plot_type='line')
