#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:41:32 2022

@author: Federico Amato
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer

from MODULES.imputiteratorClass import ImputeIterator
import MODULES.et_functions as et

from sklearn.metrics import mean_squared_error

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
        'ETa'
        ],
    'start': '2018-01-01',
    'method': 'impute',
    'nn': 5,
    'drop_index':True,
    }
# Si estrae il DataFrame completo delle Features
df = et.make_dataframe(DATABASE,
                       columns = ['θ 10','θ 20','θ 30','θ 40','θ 50','θ 60','U 2m',
                                  'Rshw dwn','RHmin','RHmax','Tmin','Tmax','ET0'],
                       start = '2018-01-01',
                       drop_index = True
                       )

# Si estraggono le misure di NDVI
target = et.make_dataframe(ND_MEASURES,
                           columns=['NDVI'],
                           method='drop',
                           drop_index = True,
                           date_format='%d/%m/%y',
                           )

# Si uniscono Features e Target
X = df.join(target)
# Si fa una prima imputazione di X (righe di Features o Target mancanti)
imputer = KNNImputer(n_neighbors=5, weights="uniform")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

#%% IMPUTAZIONE ITERATIVA

# Iterazioni massime per l'Imputazione Iterativa:
iter_limit = 100
invalid_series_limit = 10
# Oggetto ImputeIterator per l'Imputazione Iterativa
IT = ImputeIterator(iter_limit = iter_limit, invalid_series_limit = invalid_series_limit, verbose=True)
# Si fitta l'Imputatore sul set di Features e Target già imputato
IT.fit(X)
# Si imputa il Target a partire da quello misurato
imputed = IT.impute(target)

#%% ERRORE DI SCALING
# Si prendono le date con misure di Target usate nell'Imputazione
target_visti_idx = IT.target_visti.index
# Si prendono le date con i soli valori Imputati
imputed_idx = [idx for idx in imputed.index if idx not in target.index]
# Si calcola l'errore dovuto allo Scaling tra le misure di Target
mse_scaling = mean_squared_error(target.loc[target_visti_idx], imputed.loc[target_visti_idx])

#%% PLOTS
# NDVI Imputed
fig,ax = plt.subplots(figsize=(8,6))
fig.suptitle(f'Iterative Imputation of ET0 for {iter_limit} iterations', fontsize=18)
et.plot_axis(ax, [target.index, target.values, 'red'],
             plot_type='scatter', date_ticks=3, legend='Measured')
et.plot_axis(ax, [imputed.loc[target_visti_idx].index, imputed.loc[target_visti_idx].values, 'green'],
             plot_type='scatter', date_ticks=3, legend='Measured Scaled and Rescaled')
et.plot_axis(ax, [imputed.loc[imputed_idx].index, imputed.loc[imputed_idx].values, 'blue'],
              plot_type='scatter', legend='Imputed')
ax.set_ylabel('ET0')
ax.set_title(f'MSE Measured Scaled & Rescaled: {mse_scaling:.4}', color='gray', fontsize=14)
ax.legend()

# NDVI Imputed Zoom
fig,ax = plt.subplots(figsize=(8,6))
fig.suptitle(f'Iterative Imputation of ET0 for {iter_limit} iterations', fontsize=18)
et.plot_axis(ax, [target.index, target.values, 'red'],
             plot_type='scatter', date_ticks=3, legend='Measured')
et.plot_axis(ax, [imputed.loc[target_visti_idx].index, imputed.loc[target_visti_idx].values, 'green'],
             plot_type='scatter', date_ticks=3, legend='Measured Scaled and Rescaled')
et.plot_axis(ax, [imputed.loc[imputed_idx].index, imputed.loc[imputed_idx].values, 'blue'],
              plot_type='scatter', legend='Imputed')
ax.set_ylabel('ET0')
ax.set_xlim((17700,18130))
ax.set_title(f'MSE Measured Scaled & Rescaled: {mse_scaling:.4}', color='gray', fontsize=14)
ax.legend()
