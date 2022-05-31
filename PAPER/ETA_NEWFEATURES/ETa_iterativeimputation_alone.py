#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 06:03:46 2022

@author: Federico Amato
Classe Imputiterator
Fitta su Features X e imputa il set target
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from MODULES.imputiteratorClass2 import ImputeIterator
from MODULES import et_functions as et

DATABASE = '../../CSV/db_villabate_deficit_3.csv'

features = et.make_dataframe(
    DATABASE,
    columns=['θ 10', 'θ 20', 'θ 30', 'θ 40', 'θ 50', 'θ 60', 'U 2',
             'Rs', 'RHmin', 'RHmax', 'Tmin', 'Tmax', 'ETo', 'ETa'],
    start='2018-01-01',
    method='impute',
    nn=5,
    drop_index=True,
    )
# PULIZIA
# Si rimuovono le misure di Febbraio 2020, poco significative
features.drop(index=features.loc['2020-02'].index, inplace=True)

# Si inseriscono i contenuti idrici al suolo medi
features.insert(6, 'soil_humidity', features.iloc[:, 0:6].mean(axis=1))
# e si eliminano quelli alle diverse profondità
features.drop(features.columns[0:6], axis=1, inplace=True)
features.insert(0, 'gregorian_day', features.index.dayofyear)

target = et.make_dataframe(
    DATABASE,
    columns=['ETa'],
    start='2018-01-01',
    method='drop',
    drop_index=True,
    )

iter_limit = 10
it = ImputeIterator(iter_limit=iter_limit, verbose=True)
it.fit(features, target)
imputed = it.impute(target)

imputed_measures_idx = [idx for idx in imputed.index if idx in target.index]
mse = mean_squared_error(target.loc[imputed_measures_idx],
                         imputed.loc[imputed_measures_idx])

# %% PLOTS
# ETa Imputed
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'Iterative Imputation of ETa for {iter_limit} iterations',
             fontsize=18)
et.plot_axis(ax, [target.index, target.values, 'red'],
             plot_type='scatter', date_ticks=3, legend='Measured')
et.plot_axis(ax, [imputed.index, imputed.values, 'blue'],
             plot_type='scatter', legend='Imputed')
ax.set_ylabel('ETa [mm/day]', fontsize=10)
# ax.set_title(f'MSE Measured-Imputed: {mse:.4}', color='gray', fontsize=14)
ax.legend()

# Validation Score
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'Validation Score for {iter_limit} iterations', fontsize=18)
et.plot_axis(ax, [np.arange(len(it.score_mv)), it.score_mv], date_ticks=3)
ax.axhline(max(it.score_mv)[0], c='red', ls='--')
ax.text(0.1, max(it.score_mv)[0], f'Max = {max(it.score_mv)[0]:.2}', c='red',
        bbox={'facecolor': 'white'}, fontsize=12)
ax.set_ylabel('Validation Score')
