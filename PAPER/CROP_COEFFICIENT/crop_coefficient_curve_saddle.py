#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:30:41 2021

@author: Federico Amato
Il programma calcola il rapporto ETa/ET0 durante un anno
per risalire ai coefficienti caratteristici che legano le due
grandezze.
L'andamento è calcolato su due anni in modo da poter fare un confronto.
Dal programma si può anche ricercare la stagione di crescita
riconoscendola nell'andamento temporale del coefficiente.
Segue quindi una decomposizione stagionale su base settimanale,
mensile o annua.

Come si possono stimare i coefficienti Ks Kcb e Ke singolarmente?
vedi TABLE 10 da:
https://www.fao.org/3/X0490E/x0490e0a.htm#chapter%205%20%20%20introduction%20to%20crop%20evapotranspiration%20(etc)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import et_functions as et

def plot_both(ax1, ax2, days, trend, seasonal):
    et.plot_axis(ax1, [days, trend], grid=True, title='Trend')
    et.plot_axis(ax2, [days, seasonal], grid=True, title='Seasonal')

def plot_just_one(ax1, days, y, title):
    et.plot_axis(ax1, [days, y], grid=True, title=title)

database = '../CSV/db_villabate_deficit.csv'
dataframe_par = {
    'columns': [
        'θ 10',
        'θ 20',
        'θ 30',
        'θ 40',
        'θ 50',
        'θ 60',
        'Potenziali PD',
        'Sap_Flow',
        'U 2m',
        'Rshw dwn',
        'RHmin',
        'RHmax',
        'Tmin',
        'Tmax',
        'ET0',
        'ETa',
        ],
    'method': 'impute',
    'nn': 5,
    }

YEARS = ['2018','2019','2020']

TO_PLOT = [
        'K predicted',
        'K measured',
        'Potenziali PD',
        'Sap Flow',
    ]

# Si crea il DataFrame complessivo con i dati della coltivazione
df = et.make_dataframe(database, **dataframe_par)
saddle_df = pd.DataFrame()

for year in YEARS:
    yearly_data = et.filter_dates(df, f'{year}-03-15', f'{year}-10-01')
    saddle_df = pd.concat([saddle_df, yearly_data])
k = saddle_df['ETa'] / saddle_df['ET0']

eta_measured = et.make_dataframe(database, columns=['ET0','ETa'])
k_measured = eta_measured['ETa'] / eta_measured['ET0']

fig, ax = plt.subplots(figsize=(10,6))
plot_par = {
    'title': 'ETa/ET0 from 2018 to 2020 between March (15/03) and October (01/10)',
    'xlabel': 'Time',
    'ylabel': 'ETa/ET0',
    'grid': True,
    'date_ticks': 3,
    }

days = k.index.get_level_values(1)
eta_days = eta_measured.index
y_min = 0
y_max = 1.5
for y in TO_PLOT:
    if y == 'K predicted':
        et.plot_axis(ax, [days, k], plot_type='scatter', legend='Total K', **plot_par)
    elif y == 'K measured':
        et.plot_axis(ax, [eta_days, k_measured, "orange"], plot_type='scatter', legend='K measured')
    elif y == 'Potenziali PD':
        et.plot_axis(ax, [days, saddle_df['Potenziali PD']], plot_type='scatter', legend='PreDown')
        y_min = -1
    elif y == 'Sap Flow':
        et.plot_axis(ax, [days, saddle_df['Sap_Flow']], plot_type='scatter', legend='SapFlow')
        y_max = 3
ax.set_ylim(y_min,y_max)
ax.legend()
