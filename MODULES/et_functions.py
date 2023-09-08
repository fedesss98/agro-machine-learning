#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:50:38 2021

@author: Federico Amato
Funzioni Utili per le analisi sui dati di Evapotraspirazione
"""
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.dates as mdates

from sklearn.impute import KNNImputer
# Per la ricerca di outliers
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def remove_outliers(df_fit, df, model,
                    origin='DataFrame', verbose=True, **kwargs):
    """
    Utilizza i metodi di outliers detection in Scikit-Learn.
    Parameters
    ----------
    df_fit : {array-like, sparse matrix} of shape (n_samples, n_features)
        DataFrame di valori misurati su cui fittare il predittore.
    df : {array-like, sparse matrix} of shape (n_samples, n_features)
        DataFrame in cui cercare ed eliminare gli outliers.
    model : string
        Modello da usare:
            - 'IsolationForest';
            - 'LOF';
            - 'SVM';
    origin : string
        Origine dei dati, in modo da prefottarmattarli correttamente.
    verbose : bool
        Print number of outliers.
    **kwargs : dictionary
        Argomenti aggiuntivi da usare nel predittore:
            - 'contamination' (IsolationForest) : float in (0 , 0.5];
            - 'nu' (OneClassSVM) : float in (0 , 1];
    Returns
    -------
    inliners : list
        Lista di indici in cui sono presenti inliners (True) e outliers (False)

    """
    if verbose:
        print('\nDetecting Outliers...')
        print(f'Original data shape: {df.shape}')
    if origin == 'Series':
        df_fit = df_fit.values.reshape(-1,1)
        df = df.values.reshape(-1,1)
    if model == 'IsolationForest':
        cntm = kwargs.get('contamination', 0.1)
        model = IsolationForest(contamination = cntm)
    if model == 'LOF':
        model = LocalOutlierFactor()
    if model == 'SVM':
        nu = kwargs.get('nu', 0.1)
        model = OneClassSVM(nu = nu)
    model.fit(df_fit)
    indexes = model.predict(df)
    outliers = indexes == -1
    if verbose:
        print(f'{len([i for i in outliers if i])} Outliers detected.\n')
    return outliers


def filter_dates(df, start, end):
    start_ind = df.xs(level='Day', key=start).index[0] if start else 0
    end_ind = df.xs(level='Day', key=end).index[0] if end else df.index[-1][0]
    df = df.loc[start_ind: end_ind]
    return df


def extract_dataset(path, date_format):
    def dateparse(x): return dt.strptime(x, date_format)
    db = pd.read_csv(path, sep=';', decimal=',', parse_dates=[
                     'Day'], date_parser=dateparse)
    db.index = [db.index.tolist(), db['Day']]
    try:
        db['P'] = db['P'].fillna(0)
        db['I'] = db['I'].fillna(0)
    except:
        print('Error with Precipitations and Irrigation data')
    return db


def make_dataframe(data, method=None, date_format='%Y-%m-%d', **kwargs):
    """
    data: string,
    percorso da cui estrarre il database;
    method = string
    - 'drop' per buttare i le righe con features vuote
    - 'impute' per l'imputazione KNN con 'nn' vicini
    - 'itimpute' per l'imputazione iterativa
    **kwargs :
        columns: list,
        per selezionare solo alcune Colonne del database originale
        nn: int,
        per i vicini del KNNImputer;
        target: string,
        per l'Imputazione Iterativa;
        drop_index: bool,
        per tenere o meno la colonna con gli indici originali

    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        db = data
    else:
        db = extract_dataset(data, date_format)
    try:
        columns = kwargs['columns']
    except:
        columns = [col for col in db.columns if col != 'Day']
    start = kwargs.get('start', None)
    end = kwargs.get('end', None)
    df = filter_dates(db.loc[:, columns], start, end)
    drop = kwargs.get('drop_index', False)
    if method == 'drop':
        print('Dropping rows...')
        df = df.dropna().reset_index(level=0, drop=drop)
    elif method == 'impute':
        print('Automatic Imputation')
        k = kwargs['nn']
        imputer = KNNImputer(n_neighbors=k, weights="distance")
        df = pd.DataFrame(imputer.fit_transform(df),
                          columns=columns, index=df.index)
        if drop:
            df = df.reset_index(level=0, drop=True)
    elif drop:
        df = df.reset_index(level=0, drop=True)

    return df


def plot_axis(ax, *axis, plot_type='line', **kwargs):
    """
    Add plot to axe.

    Parameters
    ----------
    ax : Matplotlib Axe
        DESCRIPTION.
    *axis : list
        lista [x,y] da plottare nell'Axe.
    plot_type : string, optional
        DESCRIPTION. The default is 'line'.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """
    for axe in axis:
        x = axe[0]
        y = axe[1]
        color = axe[2] if len(axe) == 3 else None
        size = kwargs.get('size', 20)
        alpha = kwargs.get('alpha', 0.5)
        if plot_type == 'scatter':
            plot = ax.scatter(x, y, alpha=alpha, c=color, s=size)
        elif plot_type == 'line':
            plot, = ax.plot(x, y, alpha=alpha, c=color)
    if 'grid' in kwargs:
        if kwargs.get('grid'):
            ax.grid()
        else:
            ax.grid(visible=False)
    if 'title' in kwargs:
        ax.set_title(kwargs.get('title'))
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs.get('xlabel'))
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs.get('ylabel'))
    if 'legend' in kwargs:
        plot.set_label(kwargs.get('legend'))
    if 'date_ticks' in kwargs:
        ticks = kwargs.get('date_ticks')
        formatter = kwargs.get('formatter', '%Y-%m')
        make_ticks_dates(ax, ticks, formatter)
    return ax


def make_ticks_dates(ax, months, formatter):
    month_list = list(range(1, 13, months))
    ax.xaxis.set_major_locator(mdates.MonthLocator(month_list))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True)

    ax.xaxis.set_major_formatter(mdates.DateFormatter(formatter))
    # Rotates and right-aligns the x labels so they don't crowd each other.
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=35, horizontalalignment='right')


def mean_bias_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_true - y_pred)
    return diff.mean()
