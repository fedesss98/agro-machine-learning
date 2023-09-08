#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 06:09:34 2022

@author: Federico Amato
Take ETa predicted and computes Kc by taking the ratio ETa/ET0.
This gives Kc*Ks.
Ks threshold is at average soil humidity < 0.21, for bigger values this is 1.
A filter is implemented to get rid of days where SWC < 0.21 and consider only
pure values of Kc (i.e. not Kc*Ks)

Predictions in folder:
PAPER/RESULTS/PREDICTIONS/
are generated by the code:
PAPER/ETA_NEWFEATURES/ETa_total_prediction.py

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.dates as mdates

import MODULES.et_functions as et

from statsmodels.tsa.seasonal import seasonal_decompose

# %% CONSTANTS


ROOT = '../../'
DATABASE = '../../CSV/db_villabate_deficit_6.csv'

END_DATE = None

MODELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
PREDICTORS = ['mlp', 'rf']

PLOT_ETA = False
SAVE = False

num_outliers = [{'mlp': 0, 'rf': 0} for _ in MODELS]


# %% FUNCTIONS
def get_eta_predicted():
    database = (f'{ROOT}PAPER/RESULTS/PREDICTIONS/'
                f'eta_total_prediction_m{m}_{predictor}.csv')
    return pd.read_csv(
        database,
        sep=';',
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )


def get_et_measures(database):
    eta = et.make_dataframe(
        database,
        date_format='%Y-%m-%d',
        columns=['ETa'],
        start='2018-01-01',
        # end='2021-11-30',
        method='drop',
        drop_index=True,
        )

    eto = et.make_dataframe(
        database,
        date_format='%Y-%m-%d',
        columns=['ETo'],
        start='2018-01-01',
        # end='2021-11-30',
        method='impute',
        nn=5,
        drop_index=True,
        )
    return eta, eto


def plot_kc(data, x, y, hue, theoretical=False, swc=False, **kwargs):
    title = 'Kc Measured and Predicted'
    title = kwargs.get('title', title)
    xlabel = kwargs.get('xlabel', x)
    ylabel = kwargs.get('ylabel', '$K_c$')
    if swc:
        fig, (ax, axp) = plt.subplots(2, 1, sharex=True,
                                      figsize=(20, 14),
                                      gridspec_kw={'height_ratios': [3, 1]},
                                      tight_layout=True
                                      )
        swc_df = get_swc(DATABASE)
        axp.plot(swc_df)
        # Threshold
        axp.axhline(y=0.21, ls='--', c='darkblue')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        axp.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        axp.tick_params(axis='x', labelrotation=90)
        axp.set_xlabel(xlabel, fontsize=34)
        axp.set_ylabel('SWC', fontsize=34)
        axp.tick_params(axis='both', labelsize=30)
    else:
        fig, ax = plt.subplots(figsize=(14, 11))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel(xlabel, fontsize=34)
    ax.grid(ls=(0, (10, 15)))
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
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
    ax.legend(loc='upper center', fontsize=24, ncol=2)
    if 'ylim' in kwargs:
        ylim = kwargs.get('ylim')
        ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_ylabel(ylabel, fontsize=34)
    ax.tick_params(axis='both', labelsize=30)

    axs = (ax, axp) if swc else ax
    return fig, axs


def get_trapezoidal():
    database = '../../CSV/KC/Trapezoidal_Kc.csv'
    return pd.read_csv(
        database,
        sep=';',
        decimal=',',
        header=[1],
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
        dayfirst=True,
    )


def get_swc(database):
    return et.make_dataframe(
        database,
        date_format='%Y-%m-%d',
        columns=['SWC'],
        start='2018-01-01',
        # end='2021-11-30',
        method='impute',
        nn=5,
        drop_index=True,
    )


# def filter_kc(kc):
#     swc = get_swc(DATABASE)
#     kc = kc.loc[swc['SWC'] > 0] #0.21 conviene filtrare alla fine
#     return kc

def filter_kc2(kc2):
    swc = get_swc(DATABASE)
    kc2 = kc2.loc[swc['SWC'] > 0.20] #0.21 conviene filtrare alla fine
    return kc2

def plot_trapezoidal(ax, **kwargs):
    kc = get_trapezoidal()
    x = kc.index
    allen = kc.iloc[:, 0]
    rallo = kc.iloc[:, 1]

    ax.plot(x, allen, 'b--', label=allen.name, linewidth=3)
    ax.plot(x, rallo, 'b-', label=rallo.name, linewidth=3)
    return ax


def plot_relplot(df, title):
    g = sns.relplot(
        data=df,
        x="Day",
        y="ETa",
        hue="source",
        style="source",
        height=5,
        aspect=1.61,
        ).set(title=title)
    plt.show()
    return g


def save_plot(fig, figname):
    fname = f'{ROOT}/PAPER/PLOTS/CROP_COEFFICIENT/{figname}'
    fig.savefig(f'{fname}.png', bbox_inches="tight")
    fig.savefig(f'{fname}.pdf', bbox_inches="tight")
    fig.savefig(f'{fname}.eps', bbox_inches="tight")


def make_total_df(df, idx_measured):
    df['source'] = [
        'From measurements' if i in idx_measured
        else 'From predictions'
        for i in df.index]
    df.index.name = 'Day'
    return df


# %% MAIN
# Get ETa and ET0 measured
eta, eto = get_et_measures(DATABASE)

for i, m in enumerate(MODELS):
    for predictor in PREDICTORS:
        eta_predict = get_eta_predicted()
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
        
        # result = seasonal_decompose(total_kc, model='additive', period=365)
        # result._trend.plot()
        # result._seasonal.plot()
        # plt.show()    
       
        # %% PLOT ETa
        if PLOT_ETA:
            total_eta = make_total_df(total_eta, eta.index)
            plot_relplot(total_eta, title=f"ETa - Model {m}")

        # %% PLOT KC

        total_kc = make_total_df(total_kc, eta.index)
        # First remove days when SWC < 0.2, when there is water stress Ks
        # (ETa / ETo = Kc*Ks)
        #filtrare i dati dopo la scomposizione stagionale
        # total_kc = filter_kc(total_kc)
        
        # result = seasonal_decompose(total_kc['Kc'], model='additive', period=365)
        # result._trend.plot()
        # result._seasonal.plot()
        # # Kc2 = result._seasonal +  result._trend.mean()
        # # Kc2.plot()
        # plt.show()  
        
        # Remaining measures indexes
        idxs = [idx for idx in total_kc.index if idx in eta.index]
        fig, _ = plot_kc(total_kc.reset_index(),
                         x="Day",
                         y="Kc",
                         hue="source",
                         title=f"Kc Measured and Predicted for Model {m}",
                         theoretical=True,
                         ylim=(0, 3.0),
                         xlabel=None)
        if SAVE:
            figname = f"kc_predict_m{m}_{predictor}"
            save_plot(fig, figname)
        plt.show()

        # Outliers detection
        outliers = et.remove_outliers(total_kc.loc[idxs, 'Kc'],
                                      total_kc['Kc'],
                                      'IsolationForest',
                                      origin='Series',
                                      contamination=0.01,
                                      verbose=True)
        num_outliers[i][predictor] = np.count_nonzero(outliers)/len(total_kc)
        polish_kc = total_kc[~outliers]
 
        
###############################################################################        
        result = seasonal_decompose(polish_kc['Kc'], model='additive', period=365)
        result.trend.plot()
        result.seasonal.plot()
        mean_seas = result._seasonal.mean()
        Kc2 = result._seasonal +  result.trend.mean()
        # Kc3.plot()
        # plt.show()
        
        total_kc2 = make_total_df(Kc2,eta.index)
        total_kc2 = total_kc2.to_frame()
        total_kc2 = total_kc2[:-1]
        total_kc2 ['source'] = total_kc['source']
###############################################################################     
        total_kc2.rename(columns = {'seasonal':'Kc2'}, inplace = True)
        total_kc2 = filter_kc2(total_kc2)
      
        # Plot
        fig, _ = plot_kc(polish_kc.reset_index(),
                         x="Day",
                         y="Kc",
                         hue="source",
                         title=f"Kc Polished for Model {m}",
                         theoretical=True,
                         swc=False,
                         ylim=(0.2, 1.4),
                         xlabel=None)
         # Plot
        fig, _ = plot_kc( total_kc2.reset_index(),
                         x="Day",
                         y="Kc2",
                         hue="source",
                         title=f"Kc Polished for Model {m}",
                         theoretical=True,
                         swc=False,
                         ylim=(0.2, 1.4),
                         xlabel=None)
        
        if SAVE:
            figname = f"kc_predict_polish_m{m}_{predictor}"
            save_plot(fig, figname)

        plt.show()

        # %% SAVE
        if SAVE:
            total_kc.to_csv(f"{ROOT}PAPER/RESULTS/CROPCOEFFICIENT_PREDICTIONS/"
                            f"kc_prediction_m{m}_{predictor}.csv",
                            sep=';')

print("Fraction of outliers removed:")
print(pd.DataFrame(num_outliers))
