#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:27:03 2022

@author: Federico Amato

MODELLI:
    ['Rs', 'U2', 'RHmin', 'RHmax', 'Tmin',
     'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 1
    ['Rs', 'RHmin', 'RHmax', 'Tmin', 'Tmax',
     'SWC', 'NDVI', 'NDWI', 'DOY'],  # 2
    ['Rs', 'RHmin', 'Tmin', 'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 3
    ['Rs', 'RHmin', 'Tmin', 'Tmax', 'SWC', 'NDWI', 'DOY'],  # 4
    ['Rs', 'Tmin', 'Tmax', 'SWC', 'NDWI', 'DOY'],  # 5
    ['Rs', 'RHmin', 'RHmax', 'Tmin', 'Tmax'],  # 6
    ['ETo', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 7
    ['ETo', 'NDVI', 'NDWI', 'DOY'],  # 8
    ['Rs', 'Tmin', 'Tmax', 'DOY'],  # 9
    ['Rs', 'Tavg', 'RHavg', 'DOY'],  # 10
"""
import numpy as np
import pandas as pd

# from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import mean_squared_error

# from MODULES.imputiteratorClass2 import ImputeIterator
from MODULES import et_functions as et

MODELLI = [
        ['Rs', 'U2', 'RHmin', 'RHmax', 'Tmin',
         'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 1
        ['Rs', 'RHmin', 'RHmax', 'Tmin', 'Tmax',
         'SWC', 'NDVI', 'NDWI', 'DOY'],  # 2
        ['Rs', 'RHmin', 'Tmin', 'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 3
        ['Rs', 'RHmin', 'Tmin', 'Tmax', 'SWC', 'NDWI', 'DOY'],  # 4
        ['Rs', 'Tmin', 'Tmax', 'SWC', 'NDWI', 'DOY'],  # 5
        ['Rs', 'RHmin', 'RHmax', 'Tmin', 'Tmax'],  # 6
        ['ETo', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 7
        ['ETo', 'NDVI', 'NDWI', 'DOY'],  # 8
        ['Rs', 'Tmin', 'Tmax', 'DOY'],  # 9
        ['Rs', 'Tavg', 'RHavg', 'DOY'],  # 10
    ]


DATABASE = '../../CSV/db_villabate_deficit_3.csv'
DATABASE_ND = '../../CSV/NDVI/deficit_imputed_NDVI_NDWI.csv'

db = et.make_dataframe(DATABASE, drop_index=True)
ndvi_ndwi = et.make_dataframe(DATABASE_ND, drop_index=True)

# PULIZIA
# Si rimuovono le misure di Febbraio 2020, poco significative
db.drop(index=db.loc['2020-02'].index, inplace=True)
ndvi_ndwi.drop(index=ndvi_ndwi.loc['2020-02'].index, inplace=True)

# Si inseriscono i contenuti idrici al suolo medi
db.insert(6, 'SWC', db.iloc[:, 0:6].mean(axis=1, numeric_only=True))
# Si inseriscono numeri cardinali dei giorni dell'anno
db.insert(0, 'DOY', db.index.dayofyear)

db.drop(['Average NDVI', 'Std NDVI', 'Average NDWI', 'Std NDWI'],
        axis=1, inplace=True)
# Si rinomina la colonna U 2
db.rename(columns={'U 2': 'U2'}, inplace=True)
# Si inserisce la colonna di temperature medie
db.insert(17, 'Tavg',
          db.loc[:, ['Tmin', 'Tmax']].mean(axis=1, numeric_only=True))
# e la colonna della umidità dell'aria media
db.insert(20, 'RHavg',
          db.loc[:, ['RHmin', 'RHmax']].mean(axis=1, numeric_only=True))


# %% NDWI / NDVI IMPUTATION
new_ndvi_ndwi_idx = [idx for idx in db.index if idx not in ndvi_ndwi.index]
# model = MLPRegressor(hidden_layer_sizes=(60, 60), max_iter=500)
model = RandomForestRegressor()
X = db.loc[ndvi_ndwi.index,
           ['DOY', 'θ 10', 'θ 20', 'θ 30', 'θ 40', 'θ 50', 'θ 60', 'SWC',
            'P', 'Tmin', 'Tmax', 'RHmin', 'RHmax', 'Rs', 'U2', 'ETo']
           ].dropna()
x_index = X.index
X_predict = db.loc[new_ndvi_ndwi_idx,
                   ['DOY', 'θ 10', 'θ 20', 'θ 30', 'θ 40', 'θ 50', 'θ 60', 'SWC',
                    'P', 'Tmin', 'Tmax', 'RHmin', 'RHmax', 'Rs', 'U2', 'ETo']
                   ]
y_ndvi = ndvi_ndwi['NDVI-IL'].loc[x_index]
y_ndwi = ndvi_ndwi['NDWI-IL'].loc[x_index]

# Fitting NDVI
X_train, X_test, y_train, y_test = train_test_split(X, y_ndvi)
model.fit(X_train, y_train.ravel())
print(f'Score with model {model}:\n{model.score(X_test, y_test):.5}')
# Prediction
ndvi_predict = model.predict(X_predict).reshape(-1, 1)
# Fitting NDWI
X_train, X_test, y_train, y_test = train_test_split(X, y_ndwi)
model.fit(X_train, y_train.ravel())
print(f'Score with model {model}:\n{model.score(X_test, y_test):.5}')
# Prediction
ndwi_predict = model.predict(X_predict).reshape(-1, 1)


predictions = np.append(ndvi_predict, ndwi_predict, axis=1)
ndvi_ndwi_new = pd.DataFrame(predictions,
                             columns=['NDVI-ML', 'NDWI-ML'],
                             index=new_ndvi_ndwi_idx)

ndvi_ndwi = pd.concat([ndvi_ndwi.loc[:, ['NDVI-ML', 'NDWI-ML']], ndvi_ndwi_new])
ndvi_ndwi.sort_index(inplace=True)

db.insert(23, 'NDVI', ndvi_ndwi['NDVI-ML'])
db.insert(23, 'NDWI', ndvi_ndwi['NDWI-ML'])

# %% SAVE CSV
NEW_DATABASE = '../../CSV/db_villabate_deficit_4.csv'

db.to_csv(NEW_DATABASE, sep=';', decimal=',')
