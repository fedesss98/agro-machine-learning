#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:18:50 2022

@author: Federico Amato
Si cerca di ottenere la miglior prima imputazione di tutti i dati di ETa
a partire da quelli misurati.
"""
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import MODULES.et_functions as et


def scale_sets(X_train, y_train, X_test, y_test):
    """
    SCALING: Il train set non deve MAI vedere il test set,
    neanche tramite lo Scaler
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    # Si scala il train set
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Stessa cosa per il target
    scaler.fit(y_train.values.reshape(-1, 1))
    # Si scala il train set
    y_train = scaler.transform(y_train.values.reshape(-1, 1))
    y_test = scaler.transform(y_test.values.reshape(-1, 1))

    return X_train, y_train.ravel(), X_test, y_test.ravel()


def plot_scores(scores):
    fig = plt.figure()
    plt.plot(np.arange(EPOCHS+1), scores[:, 0], c='blue', alpha=0.4)
    plt.plot(np.arange(EPOCHS+1), scores[:, 0], c='green', alpha=0.4)
    plt.hlines(scores[:, 0].max(), 0, EPOCHS+1, ls='--', color='red')
    plt.text(0.5, scores[:, 0].max(), f'Max: {scores[:, 0].max():.4f}')
    plt.show()


def add_days_features(df, day=True, weekday=True, monthday=True, month=True):
    if weekday:
        df.insert(0, 'day_of_week', df.index.weekday)
    if monthday:
        df.insert(0, 'day_of_month', df.index.day)
    if month:
        df.insert(0, 'month', df.index.month)
    if day:
        df.insert(0, 'gregorian_day', df.index.dayofyear)
    return df


DATABASE = '../../CSV/db_villabate_deficit_3.csv'
EPOCHS = 25
RANDOM_STATE = 58
WARM_START = True
ITERATIONS = 2

df = et.make_dataframe(DATABASE, drop_index=True)
# Grafico del DataFrame originale
g = sns.relplot(data=df, x='Day', y='ETa')
plt.xticks(rotation=45)

# PULIZIA
# Si rimuovono le misure di Febbraio 2020, poco significative
df.drop(index=df.loc['2020-02'].index, inplace=True)
# Si inseriscono i contenuti idrici al suolo medi
df.insert(12, 'soil_humidity', df.iloc[:, 0:6].mean(axis=1))
# e si eliminano quelli alle diverse profondità
df.drop(df.columns[:12], axis=1, inplace=True)
# Si eliminano le Deviazioni degli indici ND
df.drop(columns=['Std NDWI', 'Std NDVI'], inplace=True)
# E si rinominano gli indici con lo snake_case
df.rename(columns={'Average NDVI': 'ndvi', 'Average NDWI': 'ndwi'},
          inplace=True)
# Si aggiungono i dati temporali
df = add_days_features(df)

# print('Working Dataframe:\n', df.count())

features = df.iloc[:, 0:-3]
target = df.iloc[:, -3:]
target.plot(subplots=True, figsize=(8, 10))

# Si fa una prima imputazione KNN alle features
imputer = KNNImputer(n_neighbors=5)
features = pd.DataFrame(imputer.fit_transform(features),
                        columns=features.columns, index=features.index)

# print('Features Imputed:\n', features.count())

mlp = MLPRegressor(
    hidden_layer_sizes=(365, 12, 365),
    random_state=RANDOM_STATE,
    max_iter=500,
    warm_start=WARM_START,
    )
rf = RandomForestRegressor(
    n_estimators=365,
    random_state=RANDOM_STATE,
    warm_start=WARM_START
    )

model = mlp

y = target['ETa'].dropna()
X = features.loc[y.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)
train_idx = X_train.index

# SCALING: dopo questo si avranno numpy array
X_train, y_train, X_test, y_test = scale_sets(X_train, y_train, X_test, y_test)

scores = np.zeros((EPOCHS+1, 2))
print("Predicting ETa")
for epoch in tqdm(range(EPOCHS)):
    model.fit(X_train, y_train)
    # model.set_params(n_estimators=100+10*epoch)

    y_predicted = model.predict(X_test)
    r2 = r2_score(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    scores[epoch] = [r2, mse]

imputation_idxs = [idx for idx in features.index if idx not in y.index]
measures_idxs = y.index
X_impute = features.loc[imputation_idxs]
X_impute = StandardScaler().fit_transform(X_impute)
y_imputed = model.predict(X_impute)

# Si ritorna ai DataFrame
target_imputed = pd.DataFrame(
    y_imputed, index=imputation_idxs, columns=['ETa'],
    )
target_imputed.index.name = 'Day'
target_train = pd.DataFrame(y_train, index=train_idx, columns=['ETa'])
target_total = pd.concat([target_imputed, target_train])

features_total = features.loc[target_total.index]
model.fit(features_total.values, target_total.values.ravel())
y_predicted = model.predict(X_test)
r2 = r2_score(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
scores[EPOCHS] = [r2, mse]

print(f'\nMaiVisti Scores with {model}:\n{"R2":7}\t{"RMSE":7}'
      f'\n{r2:.4f}\t{math.sqrt(mse):.4f}')

plot_scores(scores)

target_total['Source'] = [
    'measured' if i in y.index else 'imputed'
    for i in target_total.index]
# Si sovrascrivono le imputazioni corrispondenti a misure

# PLOT ETa Imputato e Misurato
g = sns.relplot(
    data=target_total,
    x='Day', y='ETa',
    style='Source', hue='Source')
plt.xticks(rotation=45)

# IMPUTAZIONE ITERATIVA
# A ogni iterazione si imputa e sostituisce
for _ in range(ITERATIONS):
    y = target_total['ETa']
    X = features.loc[y.index]
    model.set_params(warm_start=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    print("Predicting ETa")
    for _ in tqdm(range(EPOCHS)):
        model.fit(X_train.values, y_train)

    y_predicted = pd.DataFrame(model.predict(X_test.values),
                               index=X_test.index,
                               columns=['ETa'])
    r2 = r2_score(y_test, y_predicted)
    print(f'R2 Score: {r2:.4f}')

    imputation_idxs = [idx for idx in X_test.index if idx not in measures_idxs]
    y.loc[imputation_idxs] = y_predicted.loc[imputation_idxs]

# del model
