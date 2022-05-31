#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 06:03:46 2022

@author: Federico Amato
Classe Imputiterator
Fitta su Features X e imputa il set target
"""
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error

from MODULES.imputiteratorClass import ImputeIterator
from MODULES import et_functions as et

DATABASE = '../../CSV/db_villabate_deficit_6.csv'
SAVE = True

KFOLDS = 10

ITER_LIMIT = 100
INVALID_LIM = 10000

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
        ['Rs', 'Tmin', 'Tmax', 'DOY'],  # 11
        ['Rs', 'Tavg', 'RHavg', 'DOY'],  # 12
    ]

eta = et.make_dataframe(
    DATABASE,
    date_format='%Y-%m-%d',
    columns=['ETa'],
    start='2019-01-01',
    method='drop',
    drop_index=True,
    )

plt.scatter(eta.index, eta.values, color='blue', alpha=0.3)
plt.title('Measured ETa')
plt.show()

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
for n in range(KFOLDS):
    if n != KFOLDS-1:
        idx_maivisti = eta_idx[n*chunk: (n+1)*chunk]
    else:
        idx_maivisti = eta_idx[n*chunk: len(eta_idx)]

# %% ITERATIVE IMPUTATION
    for i, columns in enumerate(MODELS_FEATURES):
        print(f'\n{"***":<10} MODEL {i+1} {"***":>10}')
        features = et.make_dataframe(
            DATABASE,
            date_format='%Y-%m-%d',
            columns=columns,
            start='2018-01-01',
            method='impute',
            nn=5,
            drop_index=True,
            )

        target = eta.copy()

        # %% ITERATIVE IMPUTATION

        it = ImputeIterator(iter_limit=ITER_LIMIT,
                            inv_series_lim=INVALID_LIM,
                            verbose=True,
                            output_freq=50)
        it.fit(features, target)
        # Gli indici dei dati MaiVisti vengono inseriti nell'Imputatore
        imputed = it.impute(target, idx_maivisti=idx_maivisti)

        # Indici delle misure viste (usate per l'imputazione)
        idx_fix = [idx for idx in imputed.index if idx in target.index]
        idx_free = [idx for idx in imputed.index if idx not in target.index]
        # Errore di rescaling
        mse = mean_squared_error(target.loc[idx_fix],
                                 imputed.loc[idx_fix])
        scores = np.array(it.score_mv)
        scores[:, 1] = np.sqrt(scores[:, 1])
        # Score for the internal test
        internal_score = np.append(np.array(it.score_t),
                                   np.array(it.measured_rateo).reshape(-1, 1),
                                   axis=1)
        # Sostituisce la radice del MSE al MSE
        internal_score[:, 1] = np.sqrt(internal_score[:, 1])
        # Punteggi di test
        fit_score = np.array(it.score_fit)
        # Sostituisce la radice del MSE al MSE
        fit_score[:, 1] = np.sqrt(fit_score[:, 1])

        # Si uniscono questi punteggi a quelli del test MaiVisti
        scores = np.append(scores, fit_score, axis=1)
        scores = np.append(scores, internal_score, axis=1)
        # E si crea un DataFrame
        scores = pd.DataFrame(
            scores,
            columns=['R2mv', 'RMSEmv', 'MBEmv',
                     'R2fit', 'RMSEfit',
                     'R2t', 'RMSEt', 'MBEt', 'Rateo']
            )
        if SAVE:
            scores.to_csv(f'../PLOTS/ITERATIVEIMPUTER/SCORES_MV/10FOLDS/'
                          f'scoresmv_{ITER_LIMIT}_model{i+1}_k{n+1}.csv',
                          sep=';',)

        # %% PLOTS
        # Si crea una colonna di etichette per i dati di ETa
        target_total = pd.concat([imputed.to_frame(name='ETa'),
                                  target.loc[idx_maivisti]]).sort_index()
        target_total['source'] = [
            'Misurati' if i in idx_fix
            else 'Mai Visti' if i in idx_maivisti
            else 'Imputed' for i in target_total.index]
        target_total.index.name = 'Day'

        if SAVE:
            target_total.to_csv(f'../../CSV/IMPUTED/10FOLDS/'
                                f'eta_imputed_{ITER_LIMIT}_model{i+1}_k{n+1}',
                                sep=';')

        # Plot ETa vs Time
        sns.relplot(
            height=5, aspect=1.61,
            data=target_total,
            x='Day', y='ETa',
            style='source', hue='source')
        plt.xticks(rotation=90)
        plt.suptitle(f'Model {i+1} - Fold {n+1}')
        plt.title(f'Final Imputation: max $R^2 = {it.max_score:0.4}$')
        plt.show()
