#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:38:08 2022

@author: Federico Amato
Come si comporta la nostra imputazione iterativa rispetto a
una interpolazione lineare dei dati degli indici NDWI e NDVI?
"""

import pandas as pd
import numpy as np

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from MODULES import et_functions as et

database = '../../CSV/NDVI/deficit_imputed_NDVI_NDWI.csv'
index_measures = '../../NDVI/db_deficit_ndvi_ndwi.csv'

FIG_PARAMS = {'plot_type':'scatter',
              'date_ticks': 4,}
FIGFONT = {'weight':'bold', 'fontname':'Helvetica'}

df = et.make_dataframe(database, drop_index=True)
df_measured = et.make_dataframe(index_measures,
                                drop_index=True,
                                columns=['NDVI','NDWI'],
                                date_format='%d/%m/%y')

#%% NDVI
day = df.index
day_measures = df_measured.index
ndvi_measured = df_measured.loc[:,'NDVI']
ndvi_ml = df.loc[:,'NDVI-ML']
ndvi_il = df.loc[:,'NDVI-IL']

R2_il_ml = r2_score(ndvi_il,ndvi_ml)
MSE_il_ml = mean_squared_error(ndvi_il,ndvi_ml)

fig,ax = plt.subplots()
fig.suptitle('NDVI Imputation and Interpolation', **FIGFONT)
et.plot_axis(ax, [day, ndvi_il, 'limegreen'], **FIG_PARAMS,
             legend='NDVI interpolated')
et.plot_axis(ax, [day, ndvi_ml], plot_type='scatter',
             legend='NDVI imputed')
et.plot_axis(ax, [day_measures, ndvi_measured, 'red'], plot_type='scatter',
             legend='NDVI measured')
fig.text(0.135,0.78, f'Distances between methods:\n$R^2$ = {R2_il_ml:.3f} - MSE = {MSE_il_ml:.3f}',
         bbox = {'facecolor': 'white'})
ax.set_ylim((0.35,1.15))
ax.legend()

#%% LINEAR NDVI

fig,ax = plt.subplots()
fig.suptitle('NDVI Imputation and Interpolation', **FIGFONT)
et.plot_axis(ax, [ndvi_il,ndvi_ml, 'orange'], plot_type='scatter',grid=True,)
et.plot_axis(ax, [ndvi_measured,ndvi_measured, 'red'], plot_type='scatter',legend='NDVI measured')
fig.text(0.18,0.89, f'Distances between methods: $R^2$ = {R2_il_ml:.3f} - MSE = {MSE_il_ml:.3f}',)
ax.set_xlabel('Interpolated')
ax.set_ylabel('Imputed')
ax.legend()

#%% NDWI
day_measures = df_measured.index
ndwi_measured = df_measured.loc[:,'NDWI']
ndwi_ml = df.loc[:,'NDWI-ML']
ndwi_il = df.loc[:,'NDWI-IL']

R2_il_ml = r2_score(ndwi_il,ndwi_ml)
MSE_il_ml = mean_squared_error(ndwi_il,ndwi_ml)

fig,ax = plt.subplots()
fig.suptitle('NDWI Imputation and Interpolation', **FIGFONT)
et.plot_axis(ax, [day, ndwi_il, 'limegreen'], **FIG_PARAMS,
             legend='NDWI interpolated')
et.plot_axis(ax, [day, ndwi_ml], plot_type='scatter',
             legend='NDWI imputed')
et.plot_axis(ax, [day_measures, ndwi_measured, 'red'], plot_type='scatter',
             legend='NDWI measured')
fig.text(0.135,0.78, f'Distances between methods:\n$R^2$ = {R2_il_ml:.3f} - MSE = {MSE_il_ml:.3f}',
         bbox = {'facecolor': 'white'})
ax.set_ylim((0.,0.8))
ax.legend()

#%% LINEAR NDWI

fig,ax = plt.subplots()
fig.suptitle('NDWI Imputation and Interpolation', **FIGFONT)
et.plot_axis(ax, [ndwi_il,ndwi_ml, 'blue'], plot_type='scatter',grid=True,)
et.plot_axis(ax, [ndwi_measured,ndwi_measured, 'red'], plot_type='scatter',legend='NDWI measured')
fig.text(0.18,0.89, f'Distances between methods: $R^2$ = {R2_il_ml:.3f} - MSE = {MSE_il_ml:.3f}',)
ax.set_xlabel('Interpolated')
ax.set_ylabel('Imputed')
ax.legend()
