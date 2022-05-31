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
from sklearn.ensemble import IsolationForest

from MODULES import et_functions as et


def plot_both(ax1, ax2, days, trend, seasonal):
    et.plot_axis(ax1, [days, trend], grid=True, title="Trend")
    et.plot_axis(ax2, [days, seasonal], grid=True, title="Seasonal")


def plot_just_one(ax1, days, y, title):
    et.plot_axis(ax1, [days, y], grid=True, title=title)


MLP_PARAMS = {
    "hidden_layer_sizes": (100, 100, 100),
    "activation": "relu",  # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
    "solver": "adam",  # ‘lbfgs’, ‘sgd’, ‘adam’
    "random_state": 2,
    "max_iter": 1000,
}
MODELS = {
    "Multi-Layer Perceptron": MLPRegressor(**MLP_PARAMS),
    "Random Forest": RandomForestRegressor(),
    "Linear Regressor": LinearRegression(),
    "Support Vector Machine": SVR(),
}
DATABASE = "../../CSV/db_villabate_deficit.csv"
DATAFRAME_PAR = {
    "columns": [
        "θ 10",
        "θ 20",
        "θ 30",
        "θ 40",
        "θ 50",
        "θ 60",
        "U 2m",
        "Rshw dwn",
        "RHmin",
        "RHmax",
        "Tmin",
        "Tmax",
        "ET0",
        "ETa",
    ],
    "start": "2018-01-01",
    "drop_index": True,
}

YEARS = ["2018", "2019", "2020"]

TO_PLOT = [
    "K predicted",
    "K measured",
    "Potenziali PD",
    "Sap Flow",
]

# Si crea il DataFrame complessivo con i dati della coltivazione.
# I dati mancanti vengono imputati con 5 Neighbors
df = et.make_dataframe(DATABASE, **DATAFRAME_PAR, method="impute", nn=5)
# E il DataFrame filtrato con le righe con misure di ETa
df_measured = et.make_dataframe(DATABASE, **DATAFRAME_PAR)

# Si passa alla matrice X delle Features e vettore y dell'Output
X = df_measured.iloc[:, :-1]
y = df_measured["ETa"]

# Si usa una frazione del set per l'addestramento
# e una per il test e la predizione di K.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
y_prediction = dict()
bestR2 = [0, 0]

# Si cercano outliers nel set di addestramento.
# Modelli:
#   - IsolationForest
#   - LocalOutlierFactor (LOF)
#   - OneClassSVM (SVM)
outliers = et.remove_outliers(X_train, X_train, "IsolationForest")
inliners = np.array([not i for i in outliers])
X_train, y_train = X_train.iloc[inliners, :], y_train.iloc[inliners]

for name, model in MODELS.items():
    # Fitting the model to data
    model.fit(X_train, y_train)
    # Testing the model and taking the score (R^2)
    R2 = model.score(X_test, y_test)
    print(f"Model Score of {name}: {R2}")
    # Predict the set
    y_predicted = model.predict(X_test)
    y_prediction[name] = [y_predicted, R2]
    if R2 > bestR2[0]:
        bestR2 = [R2, name]
        y_predicted_best = y_predicted

print(f"\nBest score on prediction: {bestR2[0]} - {bestR2[1]}")

# Predizioni sull'intero set
X_total = df.iloc[:, :-1]
MODELS[bestR2[1]].fit(X, y)
eta_predicted = MODELS[bestR2[1]].predict(X_total)

# Da ETa e ET0 si ricava K, sia predetto che misurato
k_total = eta_predicted / df["ET0"]
k_measured = df_measured["ETa"] / df_measured["ET0"]

# Si cercano degli Outliers nel K predetto, fittando sul K misurato
# outliers = et.remove_outliers(k_measured, k_total, 'IsolationForest', origin = 'DataFrame')
# k_total.iloc[outliers] = np.nan
#%% GRAFICI DELLE PREDIZIONI (VALORI MISURATI E PREDETTI)
ax_par = {
    "plot_type": "scatter",
    "size": 150,
}
fig_par = {
    "title": "ETa measured and predicted with different models",
    "xlabel": "Time",
    "ylabel": "ETa",
    "grid": False,
    "date_ticks": 3,
}
fig, ax = plt.subplots(figsize=(15, 8))
# Grafico degli ETa misurati usati per l'Addestramento
et.plot_axis(
    ax,
    [y_train.index, y_train.values, "lightgrey"],
    **ax_par,
    **fig_par,
    legend="Train",
)
# Grafico degli ETa  misurati usati per il Test
et.plot_axis(ax, [y_test.index, y_test.values, "black"], **ax_par, legend="Test")
# Grafici degli ETa predetti usando le Features per il test
for name in MODELS.keys():
    legend = f"Prediction with {name} ($R^2 = {y_prediction[name][1]:.2f}$)"
    et.plot_axis(ax, [y_test.index, y_prediction[name][0]], **ax_par, legend=legend)
ax.legend()

#%% GRAFICI LINEARI DELLE PREDIZIONI
"""
# Quanto solo lontani i valori predetti da quelli misurati?
# Quanto bene si assestano su una retta di pendenza 1?
for name in MODELS.keys():
    fig2, ax2 = plt.subplots()
    fig2_par = {
        'title': f'ETa Prediction with {name} ($R^2 = {y_prediction[name][1]:.2f}$)',
        'plot_type': 'scatter',
        'xlabel': 'ETa Measured',
        'ylabel': 'ETa Predicted',
        'grid': True,
        'size': 80,
        }
    et.plot_axis(ax2, [y_test, y_prediction[name][0], 'orange'], **fig2_par)
    x_max = y_test.max()
    y_max = y_prediction[name][0].max()
    x_min = y_test.min()
    y_min = y_prediction[name][0].min()
    et.plot_axis(ax2, [np.linspace(x_min,x_max),np.linspace(y_min,y_max), 'red'], plot_type='line')
"""
#%% GRAFICO DELLA PREDIZIONE SULL'INTERO DATASET
fig3_par = {
    "title": "ETa Predicted (removing dataset outliers) and derived K",
    "ylabel": "Time",
    "grid": True,
    "plot_type": "scatter",
}
fig3, ax3 = plt.subplots(figsize=(15, 8))
et.plot_axis(
    ax3,
    [df.index.get_level_values(1), eta_predicted, "lightgrey"],
    **fig3_par,
    legend="ETa Predicted",
)
et.plot_axis(
    ax3, [y.index, y, "lightblue"], plot_type="scatter", size=50, legend="ETa Measured"
)
et.plot_axis(
    ax3,
    [df.index.get_level_values(1), k_total, "limegreen"],
    plot_type="scatter",
    legend="K Predicted (via ETa)",
)
et.plot_axis(
    ax3,
    [df_measured.index, k_measured, "magenta"],
    plot_type="scatter",
    size=50,
    legend="K Measured",
)
ax3.legend()
