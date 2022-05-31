#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:51:25 2022

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

# Si estrae il DataFrame completo delle Features
features = et.make_dataframe(
    DATABASE,
    columns=['θ 10', 'θ 20', 'θ 30', 'θ 40', 'θ 50', 'θ 60', 'U 2m',
             'Rshw dwn', 'RHmin', 'RHmax', 'Tmin', 'Tmax', 'ET0'],
    start='2018-01-01',
    drop_index=True
    )

# Si estraggono le misure di NDVI
ndvi = et.make_dataframe(ND_MEASURES,
                         columns=['NDVI'],
                         method='drop',
                         drop_index=True,
                         date_format='%d/%m/%y',
                         )

# Si estraggono le misure di NDWI
ndwi = et.make_dataframe(ND_MEASURES,
                         columns=['NDWI'],
                         method='drop',
                         drop_index=True,
                         date_format='%d/%m/%y',
                         )

eta = et.make_dataframe(DATABASE,
                        columns=['ETa'],
                        start='2018-01-01',
                        method='drop',
                        drop_index=True
                        )


# Iterazioni massime per l'Imputazione Iterativa:
iter_limit = 1
inv_series_lim = 50

# %% ETa: PRIMA IMPUTAZIONE

# Si uniscono Features e Target
X = features.join(eta)
# Si fa una prima imputazione di X (righe di Features o Target mancanti)
imputer = KNNImputer(n_neighbors=5, weights="distance")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# %% ETa: IMPUTAZIONE ITERATIVA

# Oggetto ImputeIterator per l'Imputazione Iterativa
it = ImputeIterator(iter_limit=iter_limit,
                    inv_series_lim=inv_series_lim,
                    verbose=True)

# Si fitta l'Imputatore sul set di Features e Target già imputato
it.fit(X)
# Si imputa il Target a partire da quello misurato
print('*** ETa Imputation ***\n'
      f'Features used:\n{list(X.columns)}')
eta_imputed = it.impute(eta)
eta_imputed.name = 'ETa Imputed'

it.plot_imputation()

# %% NDVI: PRIMA IMPUTAZIONE

# Si uniscono Features e Target
X = features.join(ndvi)
# Si fa una prima imputazione di X (righe di Features o Target mancanti)
imputer = KNNImputer(n_neighbors=5, weights="distance")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# %% NDVI: IMPUTAZIONE ITERATIVA

# Oggetto ImputeIterator per l'Imputazione Iterativa
it_ndvi = ImputeIterator(iter_limit=iter_limit,
                         inv_series_lim=inv_series_lim,
                         verbose=True)

# Si fitta l'Imputatore sul set di Features e Target già imputato
it_ndvi.fit(X)
# Si imputa il Target a partire da quello misurato
print('*** NDVI Imputation ***\n'
      f'Features used:\n{list(X.columns)}')
ndvi_imputed = it_ndvi.impute(ndvi)
ndvi_imputed.name = 'NDVI Imputed'
# Si aggiornano le features
features = features.join(ndvi_imputed)
it_ndvi.plot_imputation()

#%% NDWI: PRIMA IMPUTAZIONE

# Si uniscono Features e Target
X = features.join(ndwi)
# Si fa una prima imputazione di X (righe di Features o Target mancanti)
imputer = KNNImputer(n_neighbors=5, weights="distance")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# %% NDVI: IMPUTAZIONE ITERATIVA

# Oggetto ImputeIterator per l'Imputazione Iterativa
it_ndwi = ImputeIterator(iter_limit=iter_limit,
                         inv_series_lim=inv_series_lim,
                         verbose=True)

# Si fitta l'Imputatore sul set di Features e Target già imputato
it_ndwi.fit(X)
# Si imputa il Target a partire da quello misurato
print('*** NDWI Imputation ***\n'
      f'Features used:\n{list(X.columns)}')
ndwi_imputed = it_ndwi.impute(ndwi)
ndwi_imputed.name = 'NDWI Imputed'
# Si aggiornano le features
features = features.join(ndwi_imputed)
it_ndwi.plot_imputation()

# %% ETa: PRIMA IMPUTAZIONE

# Si uniscono Features e Target
X = features.join(eta)
X.to_csv('../../CSV/database_ndvindwi_imputed.csv', sep=';', decimal=',')
# Si fa una prima imputazione di X (righe di Features o Target mancanti)
imputer = KNNImputer(n_neighbors=5, weights="uniform")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# %% ETa: IMPUTAZIONE ITERATIVA

# Oggetto ImputeIterator per l'Imputazione Iterativa
it_eta = ImputeIterator(iter_limit=iter_limit,
                        inv_series_lim=inv_series_lim,
                        verbose=True)

# Si fitta l'Imputatore sul set di Features e Target già imputato
it_eta.fit(X)
# Si imputa il Target a partire da quello misurato
print('*** ETa Imputation ***\n'
      f'Features used:\n{list(X.columns)}')
eta_imputed = it_eta.impute(eta)
eta_imputed.name = 'ETa Imputed'

it_eta.plot_imputation()
