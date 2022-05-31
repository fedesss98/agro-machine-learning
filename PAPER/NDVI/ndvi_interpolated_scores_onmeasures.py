#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 08:35:27 2022

@author: Federico Amato
I punti interpolati quanto sono buoni
per predire i punti realmente misurati?
"""

import pandas as pd
import numpy as np

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

from MODULES import et_functions as et

DATABASE = '../../CSV/db_villabate_deficit.csv'
DATABASE_ND = '../../CSV/NDVI/deficit_imputed_NDVI_NDWI.csv'
ND_MEASURES = '../../NDVI/db_deficit_ndvi_ndwi.csv'
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
        ],
    'method': 'drop',
    'nn': 4,
    'drop_index': True,
    'date_format': '%d/%m/%Y',
    }

FIG_PARAMS = {'plot_type':'scatter', 'date_ticks': 4,}
FIGFONT = {'weight':'bold', 'fontname':'Helvetica'}
MLP_PARAMS = {'hidden_layer_sizes': (60,60,60),
              'activation': 'tanh',
              'max_iter': 500,
              'alpha': 0.1}
MODELS = {'MLP': MLPRegressor(**MLP_PARAMS),
          'RF': RandomForestRegressor()}

# Si raccoglie l'intero set di dati della coltivazione
df = et.make_dataframe(DATABASE, **DATAFRAME_PAR)
# Estrae i dati di NDVI e NDWI
df_nd = et.make_dataframe(DATABASE_ND, columns=['NDVI-IL','NDWI-IL'], drop_index=True)
# Estrae le misure di NDVI e NDWI
nd_measured = et.make_dataframe(ND_MEASURES,
                                drop_index=True,
                                columns=['NDVI','NDWI'],
                                date_format='%d/%m/%y')

# Unisce i dataframes di ND imputato e misurato, con Nan nelle righe vuote
df_nd = df_nd.join(nd_measured)
# Unisce i dataframes selezionando le righe presenti in entrambi
df_total = df.join(df_nd, how='inner')

# Lista dei giorni con misure di ND
# questi dovranno essere predetti
measures_days = [day for day in df_total.index if day in nd_measured.index]
not_measures_days = [day for day in df_total.index if day not in measures_days]
X_train = df_total.loc[not_measures_days,:'ET0']
y_train_ndvi = df_total.loc[not_measures_days,'NDVI-IL']
y_train_ndwi = df_total.loc[not_measures_days,'NDWI-IL']
X_test = df_total.loc[measures_days,:'ET0']
y_measured_ndvi = df_total.loc[measures_days,'NDVI']
y_measured_ndwi = df_total.loc[measures_days,'NDWI']

#%% NDVI PREDICTIONS
ndvi_predictions = pd.DataFrame([], columns=['Model','R2','MSE','Prediction'])
print('\nNDVI')
print(f'{"Modello":<10} {"Punteggio":>10}')
for name, model in MODELS.items():
    model.fit(X_train, y_train_ndvi)
    y_predicted_ndvi = model.predict(X_test)
    R2 = r2_score(y_measured_ndvi, y_predicted_ndvi)
    MSE = mean_squared_error(y_measured_ndvi, y_predicted_ndvi)
    results = {
        'Model':name,
        'R2':R2,
        'MSE':MSE,
        'Prediction':y_predicted_ndvi}
    ndvi_predictions = ndvi_predictions.append(results, ignore_index=True)
    print(f'-{name:<10}{R2:^10.4f}')

#%% NDWI PREDICTIONS
ndwi_predictions = pd.DataFrame([], columns=['Model','R2','MSE','Prediction'])
print('\nNDWI')
print(f'{"Modello":<10} {"Punteggio":>10}')
for name, model in MODELS.items():
    model.fit(X_train, y_train_ndwi)
    y_predicted_ndwi = model.predict(X_test)
    R2 = r2_score(y_measured_ndwi, y_predicted_ndwi)
    MSE = mean_squared_error(y_measured_ndwi, y_predicted_ndwi)
    results = {
        'Model':name,
        'R2':R2,
        'MSE':MSE,
        'Prediction':y_predicted_ndwi}
    ndwi_predictions = ndwi_predictions.append(results, ignore_index=True)
    print(f'-{name:<10}{R2:^10.4f}')

#%% NDVI PLOT
# Si plotta solo la predizione con maggiore R2:
# idxmax() restituisce l'indice dove R2 è maggiore
best_row = ndvi_predictions.loc[ndvi_predictions['R2'].idxmax(),:]
fig, ax_ndvi = plt.subplots()
fig.suptitle('NDVI Measured vs Prediction by Interpolated data', **FIGFONT)
et.plot_axis(ax_ndvi, [measures_days, y_measured_ndvi, 'grey'],
             legend='NDVI measured',**FIG_PARAMS)
et.plot_axis(ax_ndvi, [measures_days, best_row['Prediction'], 'orange'], plot_type = 'scatter',
             legend=f'NDVI predicted by {best_row["Model"]}')
fig.text(0.3,0.89, f'$R^2$ = {best_row["R2"]:.4} / MSE = {best_row["MSE"]:.4}')
ax_ndvi.legend()

#%% NDWI PLOT
best_row = ndwi_predictions.loc[ndvi_predictions['R2'].idxmax(),:]
fig, ax_ndwi = plt.subplots()
fig.suptitle('NDWI Measured vs Prediction by Interpolated data', **FIGFONT)
et.plot_axis(ax_ndwi, [measures_days, y_measured_ndwi, 'grey'],
             legend='NDWI measured',**FIG_PARAMS)
et.plot_axis(ax_ndwi, [measures_days, y_predicted_ndwi, 'orange'], plot_type = 'scatter',
             legend=f'NDWI predicted by {best_row["Model"]}')
fig.text(0.3,0.89, f'$R^2$ = {R2:.4} / MSE = {MSE:.4}')
ax_ndwi.legend()
