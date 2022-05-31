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

Come si possono stimare i coefficienti Ks Kcb e Ke singolarmente?
vedi TABLE 10 da:
https://www.fao.org/3/X0490E/x0490e0a.htm#chapter%205%20%20%20introduction%20to%20crop%20evapotranspiration%20(etc)

"""

import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt

from MODULES import imputiterator as ii
from MODULES import et_functions as et

from sklearn.preprocessing import StandardScaler

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

START1YEAR ='2019/01/01'
END1YEAR ='2019/12/31'

START2YEAR ='2020/01/01'
END2YEAR ='2020/11/04'


# Si crea il DataFrame complessivo con i dati della coltivazione
df = et.make_dataframe(database, **dataframe_par)

# Si crea il database del primo anno
gs1year = et.make_dataframe(database, **dataframe_par, start=START1YEAR, end=END1YEAR)
# La stagione di crescita parte da zero
gs1year['Season_Days'] = [i-gs1year.index.get_level_values(0)[0] for i in gs1year.index.get_level_values(0)]

gs2year = et.make_dataframe(database, **dataframe_par, start=START2YEAR, end=END2YEAR)
gs2year['Season_Days'] = [i-gs2year.index.get_level_values(0)[0] for i in gs2year.index.get_level_values(0)]

df['K'] = df['ETa'] / df['ET0']

fig, axs = plt.subplots(3, figsize=(6,8))

total_time = df.index.get_level_values(1).tolist()
eta = df['ETa'].values
et.plot_axis(axs[0], [total_time,eta,'green'], plot_type='scatter', title = 'Full ETa Set Imputed (5neighbors)', date_ticks = 4, formatter='%Y-%b')

eta = StandardScaler().fit_transform(gs1year['ETa'].values.reshape(-1,1))
x = gs1year.index.get_level_values(1)
x_tot = gs1year.index.get_level_values(0)
k2019 = StandardScaler().fit_transform(df.loc[x_tot, 'K'].values.reshape(-1,1))
et.plot_axis(axs[1], [x, eta], plot_type='scatter', title = 'Eta in Growing Season 2019', date_ticks = 1, formatter = '%b')
et.plot_axis(axs[1], [x, k2019, 'red'], plot_type='line', title = 'Eta in Growing Season 2019')

eta = StandardScaler().fit_transform(gs2year['ETa'].values.reshape(-1,1))
x = gs2year.index.get_level_values(1)
x_tot = gs2year.index.get_level_values(0)
k2020 = StandardScaler().fit_transform(df.loc[x_tot, 'K'].values.reshape(-1,1))
et.plot_axis(axs[2], [x, eta], plot_type='scatter', title = 'Eta in Growing Season 2020', date_ticks = 1, formatter = '%b')
et.plot_axis(axs[2], [x, k2020, 'red'], plot_type='line', title = 'Eta in Growing Season 2020')
fig.tight_layout()

fig2, ax = plt.subplots(figsize=(8,6))
et.plot_axis(ax, [x, k2019],[x, k2020], title = 'Coefficient K during Growing Season 2019', date_ticks = 1, formatter = '%b')
