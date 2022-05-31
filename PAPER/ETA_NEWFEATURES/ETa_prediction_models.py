#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:25:08 2022

@author: Federico Amato
Predizioni di Eta con diversi modelli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import IsolationForest

from MODULES import et_functions as et

# def make_superdataframe():
#     # Si raccoglie l'intero set di dati della coltivazione
#     df = et.make_dataframe(database, **dataframe_par)
#     # Estrae i dati di tensione di vapore saturo
#     df_es = et.make_dataframe(database_es, drop_index=True)
#     # Estraei dati di ETa
#     df_eta = et.make_dataframe(database, columns='ETa', method='drop', drop_index=True)
#     # Estrae i dati di NDVI e NDWI
#     df_ndvi = et.make_dataframe(database_ndvi, columns='NDVI', method='drop', date_format='%d/%m/%y', drop_index=True)
#     df_ndwi = et.make_dataframe(database_ndwi, columns='NDWI', method='drop', drop_index=True)

#     # Unisce i dataframes
#     df_es = df_es.join(df_eta)
#     df_total = df.join([df_ndvi, df_ndwi, df_es], how='inner')

MLP_PARAMS = {
    'hidden_layer_sizes': (100,100,100),
    'activation': 'relu', # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
    'solver': 'adam', #‘lbfgs’, ‘sgd’, ‘adam’
    'random_state': 2,
    'max_iter': 1000,
    }
MODELS = {
    'Multi-Layer Perceptron': MLPRegressor(**MLP_PARAMS),
    'Random Forest': RandomForestRegressor(),
    'Linear Regressor': LinearRegression(),
    'Support Vector Machine': SVR(),
    }
DATABASE = '../../CSV/db_villabate_deficit_5.csv'
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

# Si crea il DataFrame complessivo con i dati della coltivazione.
# I dati mancanti vengono imputati con 5 Neighbors
df = et.make_dataframe(DATABASE, **DATAFRAME_PAR, method='impute', nn=5)
# E il DataFrame filtrato con le righe con misure di ETa
df_measured = et.make_dataframe(DATABASE, **DATAFRAME_PAR)

# Si passa alla matrice X delle Features e vettore y dell'Output
X = df_measured.iloc[:,:-1]
y = df_measured['ETa']

# Si usa una frazione del set per l'addestramento
# e una per il test e la predizione di K.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
y_prediction = dict()
bestR2 = [0,0]

# Si cercano outliers nel set di addestramento.
# Modelli:
#   - IsolationForest
#   - LocalOutlierFactor (LOF)
#   - OneClassSVM (SVM)
outliers = et.remove_outliers(X_train, X_train, 'IsolationForest')
inliners = np.array([not i for i in outliers])
X_train, y_train = X_train.iloc[inliners, :], y_train.iloc[inliners]

for name, model in MODELS.items():
    # Fitting the model to data
    model.fit(X_train, y_train)
    # Testing the model and taking the score (R^2)
    R2 = model.score(X_test, y_test)
    print(f'Score of {name}: {R2}')
    # Predict the set
    y_predicted = model.predict(X_test)
    y_prediction[name] = [y_predicted, R2]
    if R2 > bestR2[0]:
        bestR2 = [R2,name]
        y_predicted_best = y_predicted

print(f'\nBest score on prediction: {bestR2[0]} - {bestR2[1]}')


#%% GRAFICI DELLE PREDIZIONI (VALORI MISURATI E PREDETTI)
ax_par = {
    'plot_type': 'scatter',
    'size': 150,
    }
fig_par = {
    'title': 'ETa measured and predicted with different models',
    'xlabel': 'Time',
    'ylabel': 'ETa',
    'grid': False,
    'date_ticks': 3,
    }
fig, ax = plt.subplots(figsize=(15,8))
# Grafico degli ETa misurati usati per l'Addestramento
et.plot_axis(ax, [y_train.index, y_train.values, 'lightgrey'], **ax_par, **fig_par, legend='Train')
# Grafico degli ETa  misurati usati per il Test
et.plot_axis(ax, [y_test.index, y_test.values, 'black'], **ax_par, legend='Test')
# Grafici degli ETa predetti usando le Features per il test
for name in MODELS.keys():
    legend = f'Prediction with {name} ($R^2 = {y_prediction[name][1]:.2f}$)'
    et.plot_axis(ax, [y_test.index, y_prediction[name][0]], **ax_par, legend=legend)
ax.legend()
