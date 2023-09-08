#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 17:26:29 2022

@author: Federico Amato
Train a model on almost all measures of ETa but for the lasts BLIND_DAYS.
Take scores on those lasts one by one.

Predict the complete temporal series of ETa between 2018-2021
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

from sklearn.metrics import mean_squared_error

import MODULES.et_functions as et


ROOT = '../../'
DATABASE = '../../CSV/db_villabate_deficit_6.csv'

SAVE = True
PLOTS = None  # scaled / rescaled / all / None

RANDOM_STATE = 12
BLIND_DAYS = 4

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
    return et.make_dataframe(
        DATABASE,
        columns='ETa',
        start='2018-01-01',
        # end='2021-11-30',
        method='drop',
        drop_index=True,
    )


def get_features():
    return et.make_dataframe(
        DATABASE,
        date_format='%Y-%m-%d',
        start='2018-01-01',
        # end='2021-11-30',
        method='impute',
        nn=5,
        drop_index=True,
    )


def scale_sets(sample, *to_scale, dataframe=False):
    """
    SCALING: Il train set non deve MAI vedere il test set,
    neanche tramite lo Scaler
    """
    scaler = StandardScaler()
    if isinstance(sample, (pd.DataFrame, pd.Series)):
        if len(sample.shape) < 2:
            sample = sample.values.reshape(-1, 1)
        else:
            sample = sample.values
    elif isinstance(sample, np.ndarray):
        if len(sample.shape) < 2:
            sample = sample.reshape(-1, 1)
    scaler.fit(sample)
    if not to_scale:
        scaled_list = scaler.transform(sample)
    else:
        scaled_list = []
        for y in to_scale:
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_array = y.values.reshape(-1, 1) if len(y.shape) < 2 else y.values
                scaled = scaler.transform(y_array)
                if dataframe:
                    scaled = pd.DataFrame(scaled, index=y.index)
                scaled_list.append(scaled)
            elif isinstance(y, np.ndarray):
                y_array = y.reshape(-1, 1) if len(y.shape) < 2 else y
                scaled = scaler.transform(y_array)
                scaled_list.append(scaled)

    return scaled_list


def rescale_sets(sample, *to_scale, dataframe=False):
    """
    SCALING INVERSO
    """
    scaler = StandardScaler()
    if isinstance(sample, (pd.DataFrame, pd.Series)):
        if len(sample.shape) < 2:
            sample = sample.values.reshape(-1, 1)
        else:
            sample = sample.values
    elif isinstance(sample, np.ndarray):
        if len(sample.shape) < 2:
            sample = sample.reshape(-1, 1)
    scaler.fit(sample)
    if not to_scale:
        rescaled_list = scaler.inverse_transform(sample)
    else:
        rescaled_list = []
        for y in to_scale:
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_array = y.values.reshape(-1, 1) if len(y.shape) < 2 else y.values
                scaled = scaler.inverse_transform(y_array)
                if dataframe:
                    scaled = pd.DataFrame(scaled, index=y.index)
                rescaled_list.append(scaled)
            elif isinstance(y, np.ndarray):
                y_array = y.reshape(-1, 1) if len(y.shape) < 2 else y
                scaled = scaler.inverse_transform(y_array)
                rescaled_list.append(scaled)

    return rescaled_list


def plot_imputation(x, y, eta=None, **kwargs):
    suptitle = kwargs.get('suptitle', 'ETa Imputation')
    title = kwargs.get('title', None)
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


def predict_blinds(X, y):
    scores = []
    for d in range(BLIND_DAYS):
        X_test = X[d].reshape(1, -1)
        y_test = y[d]
        # Predict
        y_model = model.predict(X_test)
        scores.append(mean_squared_error(y_test, y_model))
    return np.sqrt(scores)


features = get_features()
eta = get_eta()

eta_idx = copy.deepcopy(eta.index.values)
idx_train = eta_idx[:-BLIND_DAYS]
idx_test = eta_idx[-BLIND_DAYS:]
idx_predict = [idx for idx in features.index if idx not in idx_train]

scores = {
    'mlp': np.zeros((len(MODELS_FEATURES), BLIND_DAYS)),
    'rf': np.zeros((len(MODELS_FEATURES), BLIND_DAYS))
    }

# %% MAIN
for i, columns in enumerate(MODELS_FEATURES):

    target = eta.copy()

    X_train = features.loc[idx_train, columns]
    y_train = target.loc[idx_train]
    X_test = features.loc[idx_test, columns]
    y_test = target.loc[idx_test]
    X_predict = features.loc[idx_predict, columns]

    # Scaling
    X_train, X_test, X_predict = scale_sets(
        X_train, X_train, X_test, X_predict)
    y_train, y_test = scale_sets(y_train, y_train, y_test)

    for predictor in PREDICTORS:
        print(f'\n{"***":<5} MODEL {i+1} '
              f'// {predictor} {"***":>5}')
        model = PREDICTORS[predictor]
        model.fit(X_train, y_train.ravel())

        # %% SCORES ON BLIND DAYS
        rmse = predict_blinds(X_test, y_test)
        scores[predictor][i] = list(rmse)

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
                f'eta_total_prediction_m{i+1}_{predictor}.csv',
                sep=';'
                )

# %% PLOT SCORES
for predictor, pred_scores in scores.items():
    for model in range(len(MODELS_FEATURES)):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(np.arange(1, BLIND_DAYS+1), pred_scores[model],
                '--o', mfc='none', color='black')
        fig.suptitle("Scores on subsequent days (scaled data)")
        ax.set_title(f"Model {model+1} ({predictor})")
        ax.set_ylabel("Mean Squared Error")
        ax.set_xlabel("Days")
        if SAVE:
            plt.savefig(f"{ROOT}PAPER/PLOTS/"
                        f"BLIND_DAYS/scores_m{model+1}_{predictor}")
        plt.show()
