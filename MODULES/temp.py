#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 08:25:57 2022

@author: Federico Amato
Test di ImputeIterator
"""
import numpy as np
import matplotlib.pyplot as plt

from MODULES.classes.imputeiterator import ImputeIterator
import MODULES.et_functions as et

from sklearn.metrics import mean_squared_error

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

mlp_params = {
    'hidden_layer_sizes': (100,100,100),
    'activation': 'relu', # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
    'solver': 'adam', # ‘lbfgs’, ‘sgd’, ‘adam’
    'max_iter': 10000,
    'alpha': 0.0001,
    'learning_rate': 'constant',
    }

MLP = MLPRegressor(**mlp_params)
RF = RandomForestRegressor()

DATABASE = '../CSV/db_villabate_deficit.csv'

features = et.make_dataframe(
    DATABASE,
    columns = ['θ 10', 'θ 20', 'θ 30', 'θ 40', 'θ 50', 'θ 60', 'U 2m',
               'Rshw dwn', 'RHmin', 'RHmax', 'Tmin', 'Tmax', 'ET0'],
    start = '2018',
    method = 'impute', nn = 5, drop_index = True
    )
target = et.make_dataframe(
    DATABASE,
    columns=['ETa'],
    method='drop', drop_index = True,
    )

iter_limit = 50
invalid_series_limit = 50
it = ImputeIterator(iter_limit = iter_limit,
                    invalid_series_limit=invalid_series_limit,
                    verbose=True)
it.fit(features)
imputed = it.impute(target, maivisti_size=0.1)
imputed_measures_idx = [idx for idx in imputed.index if idx in target.index]
mse = mean_squared_error(target.loc[imputed_measures_idx], imputed.loc[imputed_measures_idx])

#%% PLOTS
# ETa Imputed
fig,ax = plt.subplots(figsize=(8,6))
fig.suptitle(f'Iterative Imputation of ETa for {iter_limit} iterations', fontsize=18)
et.plot_axis(ax, [target.index, target.values, 'red'], plot_type='scatter', date_ticks=3, legend='Measured')
et.plot_axis(ax, [imputed.index, imputed.values, 'blue'], plot_type='scatter', legend='Imputed')
ax.set_ylabel('ETa')
ax.set_title(f'MSE Measured-Imputed: {mse:.4}', color='gray', fontsize=14)
ax.legend()

# Validation Score
fig,ax = plt.subplots(figsize=(8,6))
fig.suptitle(f'Validation Score for {iter_limit} iterations', fontsize=18)
et.plot_axis(ax, [np.arange(len(it.score_v)), it.score_v], date_ticks=3)
ax.axhline(max(it.score_v), c='red', ls='--')
ax.text(0.1, max(it.score_v), f'Max = {max(it.score_v):.2}', c='red',
        bbox={'facecolor':'white'}, fontsize=12)
ax.set_ylabel('Validation Score')
