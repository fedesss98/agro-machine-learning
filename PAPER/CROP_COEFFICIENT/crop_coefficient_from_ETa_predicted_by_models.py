#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 06:09:34 2022

@author: Federico Amato
Take ETa predicted and computes Kc by taking the ratio
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


# %% FUNCTIONS

def plot_kc(data, x, y, hue, theoretical=False, **kwargs):
    fig, ax = plt.subplots(figsize=(11, 5))
    # x = data[x].values
    source = data[hue].unique()
    colors = ['black', 'red']
    facecolors = ['black', 'none']
    for i in range(2):
        x_plot = data.query(f"source == '{source[i]}'")[x]
        y_plot = data.query(f"source == '{source[i]}'")[y]
        c = colors[i]
        face = facecolors[i]
        ax.scatter(x_plot, y_plot,
                   color=c, facecolors=face, marker='o', s=10,
                   label=source[i])
    if theoretical:
        ax = plot_trapezoidal(ax)
    ax.legend(loc='upper left')
    title = 'Kc Measured and Predicted'
    title = kwargs.get('title') if 'title' in kwargs else title
    xlabel = kwargs.get('xlabel') if 'xlabel' in kwargs else x
    ylabel = kwargs.get('ylabel') if 'ylabel' in kwargs else y
    if 'ylim' in kwargs:
        ylim = kwargs.get('ylim')
        ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', labelrotation=45)

    return fig, ax


def get_trapezoidal():
    data = '../../CSV/KC/Trapezoidal_Kc.csv'
    kc = pd.read_csv(data,
                     sep=';',
                     decimal=',',
                     header=[1],
                     index_col=0,
                     parse_dates=True,
                     infer_datetime_format=True, dayfirst=True
                     )
    return kc


def plot_trapezoidal(ax, **kwargs):
    kc = get_trapezoidal()
    x = kc.index
    allen = kc.iloc[:, 0]
    rallo = kc.iloc[:, 1]

    ax.plot(x, allen, 'r--', label=allen.name, linewidth=2)
    ax.plot(x, rallo, 'r-', label=rallo.name, linewidth=2)
    return ax

# %% CONSTANTS


ROOT = '../../'
DATABASE = '../../CSV/db_villabate_deficit_6.csv'

MODELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
PREDICTORS = ['mlp', 'rf']

PLOT_ETA = False
SAVE = True


# %% MAIN
eta = et.make_dataframe(
    DATABASE,
    date_format='%Y-%m-%d',
    columns=['ETa'],
    start='2018-01-01',
    method='drop',
    drop_index=True,
    )

eto = et.make_dataframe(
    DATABASE,
    date_format='%Y-%m-%d',
    columns=['ETo'],
    start='2018-01-01',
    method='impute',
    nn=5,
    drop_index=True,
    )

for m in MODELS:
    for predictor in PREDICTORS:
        eta_predict = pd.read_csv(
            f'{ROOT}PAPER/RESULTS/PREDICTIONS/'
            f'eta_total_prediction_m{m}_{predictor}.csv',
            sep=';',
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True)
        # Remove predictions where there are measures
        eta_predict = eta_predict.loc[[idx for idx in eto.index
                                       if idx not in eta.index]]
        print(f"Model {m} shapes:"
              f"\nETa: {eta.shape}"
              f"\nETo: {eto.shape}"
              f"\nETa Predicted: {eta_predict.shape}")

        # Make total ETa DataFrame (measures + predictions)
        total_eta = pd.concat([eta_predict, eta])
        # Computes total Kc (from measures and predictions)
        total_kc = total_eta['ETa'] / eto['ETo']
        total_kc = total_kc.to_frame('Kc')

        # %% PLOT ETa
        if PLOT_ETA:
            total_eta['source'] = [
                'Measured' if i in eta.index
                else 'Predicted'
                for i in total_eta.index]
            total_eta.index.name = 'Day'

            g = sns.relplot(
                data=total_eta,
                x="Day",
                y="ETa",
                hue="source",
                style="source",
                height=5,
                aspect=1.61,
                ).set(title=f"ETa - Model {m}")
            plt.show()

        # %% PLOT KC
        total_kc['source'] = [
            'Measured' if i in eta.index
            else 'Predicted'
            for i in total_kc.index]
        total_kc.index.name = 'Day'

        plot_kc(total_kc.reset_index(),
                x="Day",
                y="Kc",
                hue="source",
                title=f"Kc Measured and Predicted for Model {m}",
                theoretical=True)
        plt.show()

        # Outliers detection
        outliers = et.remove_outliers(total_kc.loc[eta.index, 'Kc'],
                                      total_kc['Kc'],
                                      'IsolationForest',
                                      origin='Series',
                                      verbose=True)
        polish_kc = total_kc[~outliers]
        plot_kc(polish_kc.reset_index(),
                x="Day",
                y="Kc",
                hue="source",
                title=f"Kc Polished for Model {m}",
                theoretical=False,
                ylim=(0.25, 1.5))
        plt.show()

        # %% SAVE
        if SAVE:
            total_kc.to_csv(f"{ROOT}PAPER/RESULTS/CROPCOEFFICIENT_PREDICTIONS/"
                            f"kc_prediction_m{m}_{predictor}.csv",
                            sep=';')
