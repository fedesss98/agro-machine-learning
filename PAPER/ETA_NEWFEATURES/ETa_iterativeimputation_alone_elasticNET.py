#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 06:03:46 2022

@author: Federico Amato
Imputazione Iterativa di ETa usando un MLP e un ElasticNET.
Il secondo è più veloce ma meno efficiente.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV

# from MODULES.classes.imputeiterator import ImputeIterator
from MODULES.imputiteratorClass import ImputeIterator
from MODULES import et_functions as et

# GRID SEARCH
grid_params = {
    'activation': ['tanh', 'relu'],
    'alpha': [0.001, 0.1],
    }
# Grid-Search Regressor
mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100),
                     max_iter=10000,
                     random_state=432
                     )
grid = GridSearchCV(mlp, grid_params, verbose = 3,)

DATABASE = '../../CSV/db_villabate_deficit.csv'

features = et.make_dataframe(DATABASE,
                       columns = ['θ 10','θ 20','θ 30','θ 40','θ 50','θ 60','U 2m',
                                  'Rshw dwn','RHmin','RHmax','Tmin','Tmax','ET0',],
                       start = '2018-01-01',
                       method = 'impute',
                       nn = 5,
                       drop_index = True,
                       )

target = et.make_dataframe(DATABASE,
                           columns = ['ETa'],
                           start = '2018-01-01',
                           method = 'drop',
                           drop_index = True,
                           )

model = ElasticNetCV(cv=5, l1_ratio=[.1,.5,.9,.99,1], fit_intercept=False)
model.fit(features.loc[target.index], target.values.ravel())
mlp.fit(features.loc[target.index], target.values.ravel())
features_predict = features.loc[~features.index.isin(target.index)]
target_imputed_elnet = model.predict(features_predict)
target_imputed_mlp = mlp.predict(features_predict)

fig,ax = plt.subplots()
et.plot_axis(ax, [target.index, target.values,'green'], plot_type='scatter')
et.plot_axis(ax, [features_predict.index, target_imputed_elnet,'blue'], plot_type='scatter')
et.plot_axis(ax, [features_predict.index, target_imputed_mlp,'red'], plot_type='scatter')
ax.grid()

target_imputed_elnet = pd.DataFrame(target_imputed_elnet,
                                    columns=['ETa'], index=features_predict.index)
target_imputed_elnet = pd.concat([target_imputed_elnet, target])
target_imputed_mlp = pd.DataFrame(target_imputed_mlp,
                                    columns=['ETa'], index=features_predict.index)
target_imputed_mlp = pd.concat([target_imputed_mlp, target])


#%% ELASTIC NET
X_elnet = features.join(target_imputed_elnet)
it_elnet = ImputeIterator(iter_limit = 300,invalid_series_limit=50, verbose=True)
it_elnet.fit(X_elnet)
it_elnet.model = model
it_elnet.model_final = model
y = it_elnet.impute(target)
it_elnet.plot_imputation()

#%% MLP
X_mlp = features.join(target_imputed_mlp)
it_mlp = ImputeIterator(iter_limit = 150,invalid_series_limit=50, verbose=True)
it_mlp.fit(X_mlp)
it_mlp.model_final = mlp
y = it_mlp.impute(target)
it_mlp.plot_imputation()

#%% MLP - RF
X_mlp = features.join(target_imputed_mlp)
it = ImputeIterator(iter_limit = 150,invalid_series_limit=50, verbose=True)
it.fit(X_mlp)
y = it.impute(target)
it.plot_imputation()
