#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:18:50 2022

@author: Federico Amato
Modelli classici di ML applicati al dataset di ETa,
con diverse features in ingresso.
In base al valore di PLOTS stampa o meno i grafici delle predizioni per i dati
scalati o quelli riscalati.
In base al valore di SAVE esporta il csv con i punteggi di test e train sui
dati scalati (con suffisso _scaled) e riscalati.

Stampa i punteggi dei dati scalati
"""
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score

import MODULES.et_functions as et


ROOT = '../../'
DATABASE = '../../CSV/db_villabate_deficit_6.csv'

SAVE = True
PLOTS = 'all'  # scaled / rescaled / all / None

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
        ['Rs','RHmin', 'RHmax', 'Tmin', 'Tmax'],  # 8
        ['ETo', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 9
        ['ETo', 'NDVI', 'NDWI', 'DOY'],  # 10
        ['Rs', 'Tmin', 'Tmax', 'DOY'],  # 11
        ['Rs', 'Tavg', 'RHavg', 'DOY'],  # 12
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
        method='impute',
        nn=5,
        drop_index=True,
        )
    return fts


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


def rescale_sets(eta, *ys):
    scaler = StandardScaler()
    scaler.fit(eta.values.reshape(-1, 1))
    ys_scaled = []
    for y in ys:
        y_scaled = scaler.inverse_transform(y.values.reshape(-1, 1))
        y_scaled = pd.DataFrame(y_scaled, index=y.index, columns=y.columns)
        ys_scaled.append(y_scaled)
    return ys_scaled


def get_linear_models():
    models = dict()
    models['Huber'] = HuberRegressor()
    models['RANSAC'] = RANSACRegressor()
    models['TheilSen'] = TheilSenRegressor()
    return models


# evaluate a model
def evalute_model(X, y, model, name):
    y = y.ravel()
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(
        model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = np.absolute(scores)
    return scores


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
    # x = np.append(x[1], x[2])
    # y = np.append(y[1], y[2])
    et.plot_axis(ax, [x[1], y[1], 'red'],
                 plot_type='scatter', legend='ETa Prediction')
    # Total Prediction
    if len(x) > 2:
        et.plot_axis(ax, [x[2], y[2], 'orange'],
                     plot_type='scatter', legend='ETa Prediction')
    ax.legend()
    plt.show()


def plot_linear(x, y, **kwargs):
    suptitle = (kwargs.get('suptitle') if 'suptitle' in kwargs
                else 'ETa Regression')
    title = kwargs.get('title') if 'title' in kwargs else None
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(suptitle, fontsize=18, weight='bold')
    et.plot_axis(ax, grid=True, title=title,
                 xlabel='ETa Measured', ylabel='ETa Predicted',)
    # Train
    et.plot_axis(ax, [x[0], y[0], 'blue'],
                 alpha=0.1,
                 plot_type='scatter', legend='Fit Predictions')
    # Test
    et.plot_axis(ax, [x[1], y[1], 'red'],
                 plot_type='scatter', legend='Test Predictions')

    models = get_linear_models()
    results = dict()
    print_scores = kwargs.get('verbose') if 'verbose' in kwargs else False
    if print_scores:
        print("Linear Regressions:")
    for name, model in models.items():
        # evaluate the model
        results[name] = evalute_model(
            x[1].reshape(-1, 1), y[1].values.reshape(-1, 1),
            model, name)
        if print_scores:
            # summarize progress
            print(f'>{name} '
                  f'{np.mean(results[name]):.4} '
                  f'({np.std(results[name]):.4})')

    models['RANSAC'].fit(x[1].reshape(-1, 1), y[1].values.reshape(-1, 1))
    x_linear = np.linspace(min(x[0]), max(x[0]), 100)
    y_linear = models['RANSAC'].predict(x_linear.reshape(-1, 1))
    et.plot_axis(ax, [x_linear, y_linear, 'orange'],
                 legend='Regression',
                 plot_type='line', alpha=1)

    ax.legend()
    plt.show()


def make_scores(train, test):
    scores_dict = {
        'train': {
            'r2':
                r2_score(train[0], train[1]),
            'rmse':
                np.sqrt(mean_squared_error(train[0], train[1])),
            'mbe':
                et.mean_bias_error(train[0], train[1]),
            },
        'test': {
            'r2':
                r2_score(test[0], test[1]),
            'rmse':
                np.sqrt(mean_squared_error(test[0], test[1])),
            'mbe':
                et.mean_bias_error(test[0], test[1]),
            }
        }
    return pd.DataFrame(scores_dict)


eta = get_eta()
# %% CROSS-VALIDATION
# Si prendono gli indici (date) di ETa
eta_idx = copy.deepcopy(eta.index.values)
# e si mescolano in modo random
RNG = np.random.default_rng(seed=6475)
RNG.shuffle(eta_idx)
# Il set di ETa viene diviso in KFOLDS intervalli
# ogni intervallo è lungo 1/KFOLDS della lunghezza totale
chunk = int(len(eta_idx)/KFOLDS)
# Si esegue il programma prendendo di volta in volta come indici (date) dei
# MaiVisti uno di questi KFOLDS intervalli

scores_scld = {}
scores = {}
k_scores_scld = [[0 for i in range(KFOLDS)]
                 for j in range(len(MODELS_FEATURES))]
k_scores = [[0 for i in range(KFOLDS)]
            for j in range(len(MODELS_FEATURES))]

# %% MAIN
for k in range(KFOLDS):
    idx_test = eta_idx[k*chunk: (k+1)*chunk]
    idx_train = [idx for idx in eta_idx if idx not in idx_test]

    for i, columns in enumerate(MODELS_FEATURES):
        features = get_features(columns)

        target = eta.copy()

        for predictor in PREDICTORS:
            print(f'\n{"***":<5} MODEL {i+1} '
                  f'// k{k+1} '
                  f'// {predictor} {"***":>5}')
            idx_predict = [idx for idx in features.index
                           if idx not in idx_train]
            model = PREDICTORS[predictor]

            X_train = features.loc[idx_train]
            y_train = target.loc[idx_train]
            X_test = features.loc[idx_test]
            y_test = target.loc[idx_test]
            X_predict = features.loc[idx_predict]

            pred_scale = StandardScaler().fit(X_train)

            # Scaling dei dati
            X_train, y_train, X_test, y_test = scale_sets(
                X_train, y_train, X_test, y_test)
            X_predict = pred_scale.transform(X_predict)
            # Scaling delle misure
            target = StandardScaler().fit_transform(
                target.values.reshape(-1, 1))
            target = pd.DataFrame(target, index=np.sort(eta_idx))

            model.fit(X_train, y_train)
            y_fit_predict = pd.DataFrame(
                model.predict(X_train),
                columns=['ETa'],
                index=idx_train,
                )
            y_test_predict = pd.DataFrame(
                model.predict(X_test),
                columns=['ETa'],
                index=idx_test,
                )
            y_test_predict.sort_index()
            y_predict = pd.DataFrame(
                model.predict(X_predict),
                columns=['ETa'],
                index=idx_predict,
                )
            if PLOTS == 'scaled' or PLOTS == 'all':
                # Predictions plot
                xs = [idx_train, idx_test, idx_predict]
                ys = [y_fit_predict, y_test_predict, y_predict]
                plot_imputation(xs, ys, target,
                                title=f"Model {i+1} k{k+1} - {predictor}")

                # Linear plot
                xs = [y_train, y_test]
                ys = [y_fit_predict, y_test_predict]
                plot_linear(xs, ys,
                            title=f"Model {i+1} k{k+1} - {predictor}")

            if SAVE:
                y_test_predict.to_csv(
                    f'{ROOT}/PAPER/RESULTS/PREDICTIONS/'
                    f'eta_predictions_m{i+1}_k{k+1}_{predictor}_scaled.csv',
                    sep=';')

            scores_scld[predictor] = make_scores([y_train, y_fit_predict],
                                                 [y_test, y_test_predict],)
            print("Predictor Scores")
            print(scores_scld[predictor])

            # Rescale sets
            y_fit_predict, y_test_predict, y_predict, target = rescale_sets(
                eta, y_fit_predict, y_test_predict, y_predict, target)

            if SAVE:
                # Save predictions (on test set)
                y_test_predict.to_csv(
                    f'{ROOT}/PAPER/RESULTS/PREDICTIONS/'
                    f'eta_predictions_m{i+1}_k{k+1}_{predictor}.csv',
                    sep=';')
                y_predict.to_csv(
                    f'{ROOT}/PAPER/RESULTS/PREDICTIONS/'
                    f'eta_total_predictions_m{i+1}_k{k+1}_{predictor}.csv',
                    sep=';'
                    )

            # Retake original test and training set (unscaled)
            y_train = eta.loc[idx_train]
            y_test = eta.loc[idx_test]

            scores[predictor] = make_scores([y_train, y_fit_predict],
                                            [y_test, y_test_predict],)

            if PLOTS == 'rescaled' or PLOTS == 'all':
                # Predictions plot
                xs = [idx_train, idx_test, idx_predict]
                ys = [y_fit_predict, y_test_predict, y_predict]
                plot_imputation(xs, ys, target,
                                title=f"Model {i+1} k{k+1} - {predictor}")

                # Linear plot
                xs = [eta.loc[idx_train].values,
                      eta.loc[idx_test].values]
                ys = [y_fit_predict, y_test_predict]
                plot_linear(xs, ys,
                            title=f"Model {i+1} k{k+1} - {predictor}")

        k_score = scores_scld['mlp'].join(scores_scld['rf'],
                                          lsuffix='_mlp',
                                          rsuffix='_rf')

        k_scores_scld[i][k] = k_score

        k_score = scores['mlp'].join(scores_scld['rf'],
                                     lsuffix='_mlp',
                                     rsuffix='_rf')

        k_scores[i][k] = k_score

if SAVE:
    for i in range(len(MODELS_FEATURES)):
        m_scores_scld = pd.concat(
            [k_scores_scld[i][k] for k in range(KFOLDS)],
            axis=1,
            keys=range(KFOLDS))
        m_scores_scld.to_csv(f'{ROOT}/PAPER/RESULTS/SCORES/'
                             f'eta_predictions_m{i+1}_scores_scaled.csv',
                             sep=';')
        m_scores = pd.concat(
            [k_scores[i][k] for k in range(KFOLDS)],
            axis=1,
            keys=range(KFOLDS))
        m_scores.to_csv(f'{ROOT}/PAPER/RESULTS/SCORES/'
                        f'eta_predictions_m{i+1}_scores.csv',
                        sep=';')
