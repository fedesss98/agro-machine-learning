#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:46:54 2022

@author: Federico Amato
Once chosed the best predictor and the best fold, predict all the temporal
series between 2018-2021.

"""
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import MODULES.et_functions as et


ROOT = '../../'
DATABASE = '../../CSV/db_villabate_deficit_6.csv'

SAVE = True
PLOTS = None  # scaled / rescaled / all / None

KFOLDS = 4

EPOCHS = 1
RANDOM_STATE = 12

MODELS_FEATURES = [
        ['Rs', 'U2', 'RHmin', 'RHmax', 'Tmin',
         'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 1
        ['Rs', 'U2', 'RHmax', 'Tmin',
          'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 2
        ['Rs', 'U2', 'RHmax', 'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 3
        ['Rs', 'U2', 'RHmax', 'Tmax', 'SWC', 'NDWI', 'DOY'],  # 4
        ['Rs', 'U2', 'Tmax', 'SWC', 'NDWI', 'DOY'],   # 5
        ['Rs', 'U2', 'Tmax', 'SWC', 'DOY'],  # 6
        ['Rs', 'Tmax', 'SWC', 'DOY'],  # 7
        ['Rs', 'RHmin', 'RHmax', 'Tmin', 'Tmax'],  # 8
        ['ETo', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 9
        ['ETo', 'NDVI', 'NDWI', 'DOY'],  # 10
        ['Rs', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 11
        ['Rs', 'NDVI', 'NDWI', 'DOY'],  # 12
    ]

MLP_PARAMS = {
    'hidden_layer_sizes': (10, 10, 10),
    'max_iter': 10000,
    'random_state': RANDOM_STATE,
    'warm_start': False,
    }

RF_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    # Cost-Complexity Pruning: mettere il più piccolo possibile
    'ccp_alpha': 0.0,
    }

PREDICTORS = {
    'mlp': MLPRegressor(**MLP_PARAMS),
    'rf': RandomForestRegressor(**RF_PARAMS),
    }

# %% FUNCTIONS


def get_eta():
    eta = et.make_dataframe(
        DATABASE,
        columns='ETa',
        start='2018-01-01',
        end='2021-11-30', #2021-03-01
        method='drop',
        drop_index=True,
        )
    return eta


def get_features(columns):
    fts = et.make_dataframe(
        DATABASE,
        date_format='%Y-%m-%d',
        columns=columns,
        start='2018-01-01',
        end='2021-11-30',  #2021-03-01
        method='impute',
        nn=5,
        drop_index=True,
        )
    return fts


def scale_sets(sample, *to_scale, dataframe=False):
    """
    SCALING: Il train set non deve MAI vedere il test set,
    neanche tramite lo Scaler
    """
    scaler = StandardScaler()
    if isinstance(sample, pd.DataFrame) or isinstance(sample, pd.Series):
        if len(sample.shape) < 2:
            sample = sample.values.reshape(-1, 1)
        else:
            sample = sample.values
    elif isinstance(sample, np.ndarray):
        if len(sample.shape) < 2:
            sample = sample.reshape(-1, 1)
    scaler.fit(sample)
    if len(to_scale) == 0:
        scaled_list = scaler.transform(sample)
    else:
        scaled_list = []
        for y in to_scale:
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                if len(y.shape) < 2:
                    y_array = y.values.reshape(-1, 1)
                else:
                    y_array = y.values
                scaled = scaler.transform(y_array)
                if dataframe:
                    scaled = pd.DataFrame(scaled, index=y.index)
                scaled_list.append(scaled)
            elif isinstance(y, np.ndarray):
                if len(y.shape) < 2:
                    y_array = y.reshape(-1, 1)
                else:
                    y_array = y
                scaled = scaler.transform(y_array)
                scaled_list.append(scaled)

    return scaled_list

def rescale_sets(sample, *to_scale, dataframe=False):
    """
    SCALING INVERSO
    """
    scaler = StandardScaler()
    if isinstance(sample, pd.DataFrame) or isinstance(sample, pd.Series):
        if len(sample.shape) < 2:
            sample = sample.values.reshape(-1, 1)
        else:
            sample = sample.values
    elif isinstance(sample, np.ndarray):
        if len(sample.shape) < 2:
            sample = sample.reshape(-1, 1)
    scaler.fit(sample)
    if len(to_scale) == 0:
        rescaled_list = scaler.inverse_transform(sample)
    else:
        rescaled_list = []
        for y in to_scale:
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                if len(y.shape) < 2:
                    y_array = y.values.reshape(-1, 1)
                else:
                    y_array = y.values
                scaled = scaler.inverse_transform(y_array)
                if dataframe:
                    scaled = pd.DataFrame(scaled, index=y.index)
                rescaled_list.append(scaled)
            elif isinstance(y, np.ndarray):
                if len(y.shape) < 2:
                    y_array = y.reshape(-1, 1)
                else:
                    y_array = y
                scaled = scaler.inverse_transform(y_array)
                rescaled_list.append(scaled)

    return rescaled_list


def plot_imputation(x, y, eta=None, **kwargs):
    suptitle = (kwargs.get('suptitle') if 'suptitle' in kwargs
                else 'ETa Imputation')
    title = kwargs.get('title') if 'title' in kwargs else None
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(suptitle, fontsize=18, weight='bold')
    et.plot_axis(ax, grid=True, date_ticks=6,
                 title=title, xlabel='Time', ylabel='ETa',)

    if eta is not None:
        et.plot_axis(ax, [eta.index, eta, 'blue', 0.2],
                     plot_type='scatter', legend='ETa Measured')

    et.plot_axis(ax, [x[0], y[0], 'limegreen'],
                 alpha=0.4,
                 plot_type='scatter', legend='ETa Test')
    et.plot_axis(ax, [x[1], y[1], 'red'],
                 plot_type='scatter', legend='ETa Prediction')
    ax.legend()
    plt.show()


def plot_prediction(eta, x, y):
    fig, ax = plt.subplots()
    ax.plot(eta)
    ax.scatter(x, y, label='Prediction')


def get_folds_indexes(eta_idx, seed=6475):
    # e si mescolano in modo random gli indici (date) di ETa
    RNG = np.random.default_rng(seed=seed)
    RNG.shuffle(eta_idx)
    # Il set di ETa viene diviso in KFOLDS intervalli
    # ogni intervallo è lungo 1/KFOLDS della lunghezza totale
    chunk = int(len(eta_idx)/KFOLDS)
    # Si esegue il programma prendendo di volta in volta come indici (date) dei
    # MaiVisti uno di questi KFOLDS intervalli
    idx_test_list = [[] for k in range(KFOLDS)]
    idx_train_list = [[] for k in range(KFOLDS)]
    for k in range(KFOLDS):
        idx_test = eta_idx[k*chunk: (k+1)*chunk]
        idx_test_list[k] = idx_test
        idx_train_list[k] = [idx for idx in eta_idx
                             if idx not in idx_test]

    return idx_train_list, idx_test_list


eta = get_eta()
eta_idx = copy.deepcopy(eta.index.values)
idx_train_list, idx_test_list = get_folds_indexes(eta_idx)

# %% MAIN
for i, columns in enumerate(MODELS_FEATURES):
    features = get_features(columns)

    target = eta.copy()

    for predictor in PREDICTORS:
        k_r2 = np.zeros((KFOLDS))
        for k in range(KFOLDS):
            print(f'\n{"***":<5} MODEL {i+1} '
                  f'// k{k+1} '
                  f'// {predictor} {"***":>5}')
            model = PREDICTORS[predictor]
            idx_train = idx_train_list[k]
            idx_test = idx_test_list[k]

            X_train = features.loc[idx_train]
            y_train = target.loc[idx_train]
            X_test = features.loc[idx_test]
            y_test = target.loc[idx_test]

            # Scaling dei dati
            X_train, X_test = scale_sets(
                X_train, X_train, X_test)
            y_train, y_test = scale_sets(
                y_train, y_train, y_test)

            model.fit(X_train, y_train.ravel())
            k_r2[k] = model.score(X_test, y_test.ravel())

        print("Best Predictor Score:")
        print(f"{k_r2.max():.4f} at fold {k_r2.argmax()}", end='\n\n')
        best_k = k_r2.argmax()
        model = PREDICTORS[predictor]
        idx_train = idx_train_list[best_k]
        idx_predict = [idx for idx in features.index
                       if idx not in idx_train]
        X_train = features.loc[idx_train]
        y_train = target.loc[idx_train]
        X_predict = features.loc[idx_predict]

        # Scaling
        X_train, X_predict = scale_sets(
            X_train, X_train, X_predict)
        y_train = scale_sets(y_train)

        model.fit(X_train, y_train.ravel())

        # %% PREDICTION
        y_predict = model.predict(X_predict)
        y_predict, = rescale_sets(target, y_predict)
        y_predict = pd.DataFrame(
            y_predict,
            columns=['ETa'],
            index=idx_predict,
            )

        target_total = pd.concat(
            [
                y_predict.loc[[i for i in y_predict.index
                               if i not in target.index]],
                target.to_frame(name='ETa')
            ])
        target_total['source'] = [
            'Measured' if i in target.index
            else 'Predicted'
            for i in target_total.index]
        target_total.index.name = 'Day'

        g = sns.relplot(
            data=target_total,
            x='Day',
            y='ETa',
            style='source',
            hue='source')
        plt.xticks(rotation=50)
        plt.suptitle(f'Model {i+1} predictions ({predictor})')
        plt.show()

        if SAVE:
            y_predict.to_csv(
                f'{ROOT}/PAPER/RESULTS/PREDICTIONS/'
                f'eta_total_prediction_kfolds_m{i+1}_{predictor}.csv',
                sep=';'
                )
