#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:25:38 2022

@author: Federico Amato
Quali features sono maggiormente correlate all'ETa?
1) Si usa la Scattermatrix per una valutazione visiva;
2) Si usano algoritmi di Feature Importance da Scikit learn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

import seaborn as sns

from MODULES import et_functions as et

def plot_scattermatrix(df):
    sm = pd.plotting.scatter_matrix(df, alpha=0.6, figsize=(10, 10), diagonal='kde')

    # Ruota le etichette
    [s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
    [s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
    
     # Imposta la dimensione delle label
    [s.xaxis.label.set_fontsize(13) for s in sm.reshape(-1)]
    [s.yaxis.label.set_fontsize(13) for s in sm.reshape(-1)]

    # Sposta le etichette per non farle sovrapporre alla figura
    [s.get_yaxis().set_label_coords(-0.7,0.35) for s in sm.reshape(-1)]

    # Nasconde i ticks
    [s.set_xticks(()) for s in sm.reshape(-1)]
    [s.set_yticks(()) for s in sm.reshape(-1)]

    plt.show()


database = '../../CSV/db_villabate_deficit_3.csv'
database_es = '../../CSV/tensione_vapore_saturo.csv'
database_ndvi = '../../NDVI/deficit_ndvi.csv'
database_ndwi = '../../NDVI/deficit_NDWI.csv'
dataframe_par = {
    'columns': [
        'U 2',
        'Rshw dwn',
        'RHmin',
        'RHmax',
        'Tmin',
        'Tmax',
        'ETo',
        ],
    'method': 'drop',
    'nn': 4,
    'drop_index': True,
    'date_format': '%d/%m/%Y',
    }

# Si raccoglie l'intero set di dati della coltivazione
df = et.make_dataframe(database, **dataframe_par)
# Estrae i dati di tensione di vapore saturo
df_es = et.make_dataframe(database_es, drop_index=True)
# Estraei dati di ETa
df_eta = et.make_dataframe(database, columns='ETa', method='drop', drop_index=True)
# Estrae i dati di NDVI e NDWI
df_ndvi = et.make_dataframe(database_ndvi, columns='NDVI', method='drop', date_format='%d/%m/%y', drop_index=True)
df_ndwi = et.make_dataframe(database_ndwi, columns='NDWI', method='drop', drop_index=True)


#soil water contents 

dataframe_SWC = {
    'columns': [
        'θ 10',
        'θ 20',
        'θ 30',
        'θ 40',
        'θ 50',
        ],
    'method': 'drop',
    'nn': 4,
    'drop_index': True,
    'date_format': '%d/%m/%Y',
    }
df_SWC = et.make_dataframe(database, **dataframe_SWC)
# Contenuto idrico al suolo medio
# nella media è stato esculo θ60 su consiglio del prof provenzano
mean_SWC = df_SWC.mean(axis=1).to_frame()
mean_SWC = mean_SWC.rename(columns={0 : "θ mean"})

# Unisce i dataframes
df_total = df.join([mean_SWC,df_ndvi, df_ndwi,df_eta], how='inner')

#%% FEATURE IMPORTANCE
model = GradientBoostingRegressor(random_state=0)
X = df_total.dropna().iloc[:,0:-1]
y = df_total.dropna().iloc[:,-1]
model.fit(X,y)
importances = model.feature_importances_
features = df_total.iloc[:,:-1].columns
df_fi = pd.Series(importances, index=features,name='Importance').sort_values(ascending=False)
print(f'{"Feature":{8}} -  Importance Score\n____________________________')
for feature, importance in df_fi.iteritems():
    print(f'{feature:{11}} {importance:.3}')

plt.barh(df_fi.index, df_fi.values)
plt.title('Feature Importance in ETa predictions')
plt.tight_layout()

#%% HEATMAP
# calcolo del coefficiente di correlazione.
corr_df = df_total.corr(method='pearson')
plt.figure(figsize=(9, 6))
sns.heatmap(corr_df, annot=True, center=0, cmap='RdBu')
plt.suptitle('Correlation Coefficient', fontsize=13, y=0.92)
plt.xticks(fontsize = 13, rotation = 45)
plt.yticks(fontsize = 13)
plt.show()

#%% SCATTERMATRIX
# Si crea la matrice di correlazione tra le Features
matrix = plot_scattermatrix(df_total)

# PLOTS
fig = plt.figure(figsize = (16,9))
plt.plot(df_SWC)
plt.plot(mean_SWC)
plt.legend(['θ 10', 'θ 20', 'θ 30', 'θ 40', 'θ 50', 'θ mean', ])
plt.title('Soil Water [deficit]')
plt.show()
