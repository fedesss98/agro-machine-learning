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
    plt.title('MLP $R^2$ in training and MaiVisti Test (last score)')
    plt.plot(np.arange(EPOCHS+1), scores[:, 0], c='blue', alpha=0.4)
    plt.plot(np.arange(EPOCHS+1), scores[:, 0], c='green', alpha=0.4)
    plt.hlines(scores[:, 0].max(), 0, EPOCHS+1, ls='--', color='red')
    plt.text(0.5, scores[:, 0].max(), f'Max: {scores[:, 0].max():.4f}')
    plt.show()


DATABASE = '../../CSV/db_villabate_deficit_3.csv'
EPOCHS = 1
RANDOM_STATE = 58  # 58
WARM_START = False
REFIT = True
ITERATIONS = 0

df = et.make_dataframe(DATABASE, drop_index=True)

# PULIZIA
# Si rimuovono le misure di Febbraio 2020, poco significative
df.drop(index=df.loc['2020-02'].index, inplace=True)

# Si inseriscono i contenuti idrici al suolo medi
df.insert(12, 'soil_humidity', df.iloc[:, 0:6].mean(axis=1))
# e si eliminano quelli alle diverse profondità
df.drop(df.columns[0:12], axis=1, inplace=True)
# Si eliminano le Deviazioni degli indici ND
# e le Precipitazioni e le Irrigazioni
df.drop(columns=['Std NDWI', 'Std NDVI', 'I', 'P'], inplace=True)
# E si rinominano gli indici con lo snake_case
df.rename(columns={'Average NDVI': 'ndvi', 'Average NDWI': 'ndwi'},
          inplace=True)
# Si aggiungono i dati temporali
# df.insert(0, 'day_of_week', df.index.weekday)
# df.insert(0, 'day_of_month', df.index.day)
# df.insert(0, 'month', df.index.month)
df.insert(0, 'gregorian_day', df.index.dayofyear)

print('Dataframe:\n', df.count())

features = df.iloc[:, 0:-3]
target = df.iloc[:, -3:]  # ndvi, ndwi, eta
# target.plot(subplots=True, figsize=(8, 10))

# Si fa una prima imputazione KNN alle features
imputer = KNNImputer(n_neighbors=5)
features = pd.DataFrame(imputer.fit_transform(features),
                        columns=features.columns, index=features.index)
# print('Features Imputed:\n', features.count())

mlp = MLPRegressor(
    hidden_layer_sizes=(365, 365),
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


# %% ADDESTRAMENTO
# Si addestra il modello sul set ridotto di ETa misurato
y = target['ETa'].dropna()  # misure di ETa
X = features.loc[y.index]  # features corrispondenti a misure di ETa

# Il 30% di misure di ETa vengono tenute per la validazione in addestramento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)
train_idx = X_train.index
test_idx = X_test.index

# SCALING: dopo questo si avranno numpy array
X_train, y_train, X_test, y_test = scale_sets(X_train, y_train, X_test, y_test)
# Vettore dei punteggi del modello
scores_mlp = np.zeros((EPOCHS+1, 3))
scores_rf = np.zeros((2, 3))
# Si riaddestra il modello diverse volte sullo stesso set
print("Training Model on 75% of Measured data")

for epoch in tqdm(range(EPOCHS)):
    mlp.fit(X_train, y_train)
    # model.set_params(n_estimators=100+10*epoch)

    y_predicted = mlp.predict(X_test)
    r2 = r2_score(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    mbe = et.mean_bias_error(y_test, y_predicted)
    scores_mlp[epoch] = [r2, mse, mbe]

rf.fit(X_train, y_train)
# model.set_params(n_estimators=100+10*epoch)

y_predicted = rf.predict(X_test)
r2 = r2_score(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
mbe = et.mean_bias_error(y_test, y_predicted)
scores_rf[0] = [r2, mse, mbe]

# Le date da imputare sono quelle del database che non corrispondono a misure
imputation_idxs = [idx for idx in features.index if idx not in y.index]
X_impute = features.loc[imputation_idxs]
X_impute = StandardScaler().fit_transform(X_impute)

# %% FIRST IMPUTATION MLP
model = mlp
# Si fa una predizione su queste date, usando il modello addestrato prima
y_imputed = model.predict(X_impute)

# Si ritorna ai DataFrame
# Dataframe con ETa imputato
target_imputed = pd.DataFrame(
    y_imputed, index=imputation_idxs, columns=['ETa'],
    )
target_imputed.index.name = 'Day'
# Dataframe con misure di ETa usate per l'addestramento
target_train = pd.DataFrame(y_train, index=train_idx, columns=['ETa'])
# DataFrame con misure di ETa usate per il test (MaiVisti)
target_test = pd.DataFrame(y_test, index=test_idx, columns=['ETa'])
# DataFrame di ETa visti in addestramento, misurati e imputati
target_visti = pd.concat([target_imputed, target_train])
features_visti = (StandardScaler()
                  .fit_transform(features.loc[target_visti.index]))
# Il modello è addestrato su tutti i dati visti
if REFIT:
    print("Refitting Model on Measured+Imputed data")
    for epoch in tqdm(range(EPOCHS)):
        model.fit(features_visti, target_visti.values.ravel())
print("Predicting MaiVisti")
# E si fa una predizione dei dati MaiVisti
y_predicted = model.predict(X_test)
# Si calcolano i punteggi R2 e MSE
r2 = r2_score(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
mbe = et.mean_bias_error(y_test, y_predicted)
scores_mlp[EPOCHS] = [r2, mse, mbe]

print(f'\nMaiVisti Scores with {model}:\n{"R2":7}\t{"RMSE":7}'
      f'\n{r2:.4f}\t{math.sqrt(mse):.4f}\t{mbe:.4f}')

plot_scores(scores_mlp)

target_total = (pd.concat([target_imputed, target_train, target_test])
                .sort_index())
# Si crea una colonna di etichette per i dati di ETa
target_total['source'] = [
    'Visti' if i in train_idx else 'Mai Visti' if i in test_idx else 'Imputed'
    for i in target_total.index]

g = sns.relplot(
    data=target_total,
    x='Day', y='ETa',
    style='source', hue='source')
plt.xticks(rotation=45)
plt.title('MLP Imputation')
plt.show()

# =============================================================================
# # %% PLOT MATPLOTLIB
# fig, ax = plt.subplots()
# fig.suptitle('MLP Imputation')
# imputed = target_total.loc[target_total['source'] == 'Imputed', 'ETa']
# visti = target_total.loc[target_total['source'] == 'Visti', 'ETa']
# et.plot_axis(ax, [imputed.index, imputed.values],
#              plot_type='scatter',
#              alpha=0.3)
# et.plot_axis(ax, [visti.index, visti.values],
#              plot_type='scatter',
#              alpha=0.3)
# plt.show()
# =============================================================================

# %% FIRST IMPUTATION RF
model = rf
# Col modello addestrato si può procedere alla prima Imputazione

# Si fa una predizione su queste date, usando il modello addestrato prima
y_imputed = model.predict(X_impute)

# Si ritorna ai DataFrame
# Dataframe con ETa imputato
target_imputed = pd.DataFrame(
    y_imputed, index=imputation_idxs, columns=['ETa'],
    )
target_imputed.index.name = 'Day'
# Dataframe con misure di ETa usate per l'addestramento
target_train = pd.DataFrame(y_train, index=train_idx, columns=['ETa'])
# DataFrame con misure di ETa usate per il test (MaiVisti)
target_test = pd.DataFrame(y_test, index=test_idx, columns=['ETa'])
# DataFrame di ETa visti in addestramento, misurati e imputati
target_visti = pd.concat([target_imputed, target_train])
features_visti = (StandardScaler()
                  .fit_transform(features.loc[target_visti.index]))
# Il modello non si riaddestra
# E si fa una predizione dei dati MaiVisti
y_predicted = model.predict(X_test)
# Si calcolano i punteggi R2 e MSE
r2 = r2_score(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
mbe = et.mean_bias_error(y_test, y_predicted)
scores_rf[1] = [r2, mse, mbe]

print(f'\nMaiVisti Scores with {model}:\n{"R2":7}\t{"RMSE":7}'
      f'\n{r2:.4f}\t{math.sqrt(mse):.4f}\t{mbe:.4f}')


target_total = (pd.concat([target_imputed, target_train, target_test])
                .sort_index())
# Si crea una colonna di etichette per i dati di ETa
target_total['source'] = [
    'Visti' if i in train_idx else 'Mai Visti' if i in test_idx else 'Imputed'
    for i in target_total.index]

g = sns.relplot(
    data=target_total,
    x='Day', y='ETa',
    style='source', hue='source',)
plt.xticks(rotation=45)
plt.title('RF Imputation')
plt.show()

# =============================================================================
# # %% PLOT MATPLOTLIB
# fig, ax = plt.subplots()
# fig.suptitle('RF Imputation')
# imputed = target_total.loc[target_total['source'] == 'Imputed', 'ETa']
# visti = target_total.loc[target_total['source'] == 'Visti', 'ETa']
# et.plot_axis(ax, [imputed.index, imputed.values],
#              plot_type='scatter',
#              alpha=0.3)
# et.plot_axis(ax, [visti.index, visti.values],
#              plot_type='scatter',
#              alpha=0.3)
# plt.show()
# =============================================================================

# %% IMPUTAZIONE KNN
# print(f"{'*********':15}\nKNNImputation")
features_target = df.iloc[:, :-3].join(df.iloc[:, -1])
# print(f"Total DataFrame: \n{features_target.count()}")

imputer = KNNImputer(n_neighbors=5, weights='uniform')
features_target = pd.DataFrame(
    imputer.fit_transform(features_target),
    columns=features_target.columns,
    index=features_target.index
    )

# Si separano i DataFrame
# Dataframe con ETa imputato
target = features_target.loc[:, 'ETa']
features = features_target.iloc[:, :-1]
target_visti = pd.concat([target.loc[train_idx], target.loc[imputation_idxs]])
features_visti = pd.concat([features.loc[train_idx],
                            features.loc[imputation_idxs]])
target_maivisti = target.loc[test_idx]
features_maivisti = features.loc[test_idx]
# Il modello è addestrato su tutti i dati visti
model2 = MLPRegressor(
    hidden_layer_sizes=(365, 12, 365),
    random_state=RANDOM_STATE,
    max_iter=500,
    warm_start=WARM_START,
    )
scaler_ft = StandardScaler().fit(features_visti)
scaler_trg = StandardScaler().fit(target_visti.values.reshape(-1, 1))
X_train = scaler_ft.transform(features_visti)
y_train = scaler_trg.transform(target_visti.values.reshape(-1, 1))
X_impute = scaler_ft.transform(features_maivisti)
y_test = scaler_trg.transform(target_maivisti.values.reshape(-1, 1))
model2.fit(X_train, y_train.ravel())
# E si fa una predizione dei dati MaiVisti
y_predicted = model2.predict(X_impute)
# Si calcolano i punteggi R2 e MSE
r2 = r2_score(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
mbe = et.mean_bias_error(y_test, y_predicted)

print(f'\nMaiVisti Scores with KNNImputer and MLP:\n{"R2":7}\t{"RMSE":7}'
      f'\n{r2:.4f}\t{math.sqrt(mse):.4f}\t{mbe:.4f}')

target_total = (pd.concat([target_visti, target_maivisti]).to_frame()
                .sort_index())
target_total['source'] = [
    'Visti' if i in train_idx else 'Mai Visti' if i in test_idx else 'Imputed'
    for i in target_total.index]

g = sns.relplot(
    data=target_total,
    x='Day', y='ETa',
    style='source', hue='source')
plt.xticks(rotation=45)
plt.title('KNN Imputation')
plt.show()

# =============================================================================
# # %% PLOT MATPLOTLIB
# fig, ax = plt.subplots()
# fig.suptitle('KNN Imputation')
# imputed = target_total.loc[target_total['source'] == 'Imputed', 'ETa']
# visti = target_total.loc[target_total['source'] == 'Visti', 'ETa']
# et.plot_axis(ax, [imputed.index, imputed.values],
#              plot_type='scatter',
#              alpha=0.3)
# et.plot_axis(ax, [visti.index, visti.values],
#              plot_type='scatter',
#              alpha=0.3)
# plt.show()
# =============================================================================
