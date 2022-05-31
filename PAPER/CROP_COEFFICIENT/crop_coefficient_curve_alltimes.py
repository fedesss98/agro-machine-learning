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

import matplotlib.pyplot as plt

import et_functions as et

from statsmodels.tsa.seasonal import seasonal_decompose

def decompose_data(df, period=365, trend=True, seasonal=True):
    decomposition = seasonal_decompose(df, model='additive', period=period)
    trend_series = decomposition.trend
    seasonal_series = decomposition.seasonal

    n=2 if trend and seasonal else 1
    fig2, axs = plt.subplots(n, figsize=(10,6))
    plt.suptitle(f'Data decomposition in periods of {period}-days')
    if n==2:
        plot_both(axs[0], axs[1], days, trend_series, seasonal_series)
    elif trend:
        plot_just_one(axs, days, trend_series, 'Trend')
    else:
        plot_just_one(axs, days, seasonal_series, 'Seasonal')

    return decomposition

def plot_both(ax1, ax2, days, trend, seasonal):
    et.plot_axis(ax1, [days, trend], grid=True, title='Trend')
    et.plot_axis(ax2, [days, seasonal], grid=True, title='Seasonal')

def plot_just_one(ax1, days, y, title):
    et.plot_axis(ax1, [days, y], grid=True, title=title)

database = '../../CSV/db_villabate_deficit.csv'
dataframe_par = {
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
    'method': 'impute',
    'nn': 5,
    }

START ='2017/07/12'
END ='2018/01/01'

# Si crea il DataFrame complessivo con i dati della coltivazione
df = et.make_dataframe(database, **dataframe_par)
eta_measured = et.make_dataframe(database, columns=['ET0','ETa'])
data_pre = et.filter_dates(df, START, END)
data_deficit = et.filter_dates(df, END, None)

k_deficit = data_deficit['ETa'] / data_deficit['ET0']
k_pre = data_pre['ETa'] / data_pre['ET0']
k_measured = eta_measured['ETa'] / eta_measured['ET0']

fig, ax = plt.subplots(figsize=(10,6))
plot_par = {
    'title': 'ETa/ET0 in time',
    'xlabel': 'Time' ,
    'ylabel': 'ETa/ET0',
    'grid': True,
    'date_ticks': 3,
    }

days_pre = data_pre.index.get_level_values(1)
days = data_deficit.index.get_level_values(1)
eta_days = eta_measured.index
et.plot_axis(ax, [days, k_deficit], plot_type='scatter', legend='Total K', **plot_par)
et.plot_axis(ax, [days_pre, k_pre, "green"], plot_type='scatter', legend='K pre-deficit')
et.plot_axis(ax, [eta_days, k_measured, "orange"], plot_type='scatter', legend='K measured')
ax.legend()

# ANALISI STAGIONALE
MONTHLY = 30
YEARLY = 365
WEEKLY = 7

week_data = decompose_data(k_deficit, period=WEEKLY, seasonal=False)
month_data = decompose_data(k_deficit, period=MONTHLY)
year_data = decompose_data(k_deficit, period=YEARLY)
