#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:08:59 2022

@author: Federico Amato
"""
import numpy as np
import matplotlib.pyplot as plt

from MODULES.imputiteratorClass import ImputeIterator
import MODULES.et_functions as et

from sklearn.metrics import mean_squared_error

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
# Si estrae il DataFrame completo di Features e Target,
# imputando con un KNN tutte le colonne.
# (Anche il Target sarà così già imputato)
df = et.make_dataframe('../../CSV/db_villabate_deficit.csv', **DATAFRAME_PAR)
# Si estraggono le sole misure di ETa per replicarne il set su ET0
target_ETa = et.make_dataframe('../../CSV/db_villabate_deficit.csv',
                           method='drop',
                           drop_index = True,
                           columns=['ETa'])
# Si prendono le misure di ET0 che corrispondono a quelle di ETa
target = df['ET0'].loc[target_ETa.index]

# Iterazioni massime per l'Imputazione Iterativa:
iter_limit = 150
invalid_series_limit = 100
# Oggetto ImputeIterator per l'Imputazione Iterativa
it = ImputeIterator(iter_limit = iter_limit, invalid_series_limit = invalid_series_limit, verbose=True)
# Si fitta l'Imputatore sul set di Features e Target già imputato
it.fit(df)
# Si imputa il Target a partire da quello misurato
imputed = it.impute(target, maivisti_size=0.1)
# Si prendono le date con misure di Target usate nell'Imputazione
visti_idx = it.target_visti.index
# Si prendono le date con i soli valori Imputati
imputed_idx = [idx for idx in imputed.index if idx not in target.index]
# Si calcola l'errore dovuto allo Scaling tra le misure di Target
mse = mean_squared_error(target.loc[visti_idx], imputed.loc[visti_idx])

#%% PLOTS
# ET0 Imputed
fig,ax = plt.subplots(figsize=(8,6))
fig.suptitle(f'Iterative Imputation of ET0 for {iter_limit} iterations', fontsize=18)
et.plot_axis(ax, [target.index, target.values, 'red'],
             plot_type='scatter', date_ticks=3, legend='Measured')
et.plot_axis(ax, [imputed.loc[visti_idx].index, imputed.loc[visti_idx].values, 'green'],
             plot_type='scatter', date_ticks=3, legend='Measured Scaled and Rescaled')
et.plot_axis(ax, [imputed.loc[imputed_idx].index, imputed.loc[imputed_idx].values, 'blue'],
             plot_type='scatter', legend='Imputed')
ax.set_ylabel('ET0')
ax.set_title(f'MSE Measured Scaled & Rescaled: {mse:.4}', color='gray', fontsize=14)
ax.legend()

#%% Validation Score
fig,ax = plt.subplots(figsize=(8,6))
fig.suptitle(f'Validation Score for {iter_limit} iterations', fontsize=18)
et.plot_axis(ax, [np.arange(len(it.score_v)), it.score_v], grid=True)
ax.axhline(max(it.score_v), c='red', ls='--')
ax.text(0.1, max(it.score_v), f'Max = {max(it.score_v):.2}', c='red',
        bbox={'facecolor':'white'}, fontsize=12)
ax.set_ylabel('Validation Score')
