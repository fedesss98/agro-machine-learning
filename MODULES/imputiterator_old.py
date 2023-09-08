#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 21:14:39 2021

@author: Federico Amato
Funzioni per l'Imputazione Iterativa.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neural_network import MLPRegressor

def make_sets(X, y, target, split_len):
    randindex = X.index.get_level_values(0).tolist()
    np.random.shuffle(randindex)
    train_index = randindex[:split_len]
    test_index = randindex[split_len:]

    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    return [train_index, test_index], [X_train, y_train], [X_test, y_test]

def model_fit(X_train, y_train, X_test, y_test):
    model.set_params(**model_params)

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    return model, score

def complete_set(df, target, real_data):
    train_cols = [cols for cols in df.columns if cols != target]

    df[f'{target}_completed'] = model.predict(df.loc[:,train_cols]) # previsione di ETa usando tutto il set di dati

    df[f'{target}_completed'] = df[target].fillna(df[f'{target}_completed'])

def iteration_loop(df, target, n=100):
    loop = 0
    best_score = 0
    while (df[target].count() < len(df.index)):
        loop = loop + 1
        train_cols = [
            cols
            for cols in df.columns
            if cols not in [target, f'{target}_completed']
        ]
        X = df.loc[:,train_cols] # features fino a ET0 compresa
        y = df.loc[:,f'{target}_completed'] # Target Misurato completato con quello Predetto
        train_len = len(df.index)-n

        indexes, train_sets, test_sets = make_sets(X,y, target, train_len)
        model.set_params(**model_params)
        model.fit(train_sets[0], train_sets[1])
        score = model.score(test_sets[0], test_sets[1])

        if score > best_score: best_score = score

        predicted = pd.Series(model.predict(test_sets[0]), index=indexes[1], name=f'{target}_predicted')
        df[target].fillna(predicted, inplace=True)
        df.loc[df[target].notnull(),f'{target}_completed'] = df[target] # rimpiazza le righe di ETa Completed con i nuovi ETa predetti

        print(f'Iteration {loop}:')
        print(f'{target} count: ', df[target].count())
        print(f'Model Score: {score}')

    return best_score, df

def print_output(score, iterative_score):
    output = "Iterative Imputation Score > Original prediction Score"
    print(output) if iterative_score > score else print(output.replace('>', '<'))
    print(f'First Prediction Score: {score}')
    print(f'Iterative Imputation Best Score: {iterative_score}')

def plot_imputation(ax, target, df, target_measured, **fig_params):
    x1 = df.index.get_level_values('Day')
    y1 = df[target]
    x2 = target_measured.index.get_level_values('Day')
    y2 = target_measured[target]
    et0 = df['ET0']

    ax.scatter(x1, y1, c="red", alpha=0.3, s=20, label="Imputed")
    ax.scatter(x2, y2, c="blue", alpha=0.3, s=10, label="Measured")
    ax.plot(x1, et0, c="black", alpha=0.3, label="ET0")
    ax.set_title(f'{target} and ET0')
    ax.legend()

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    # Rotates and right-aligns the x labels so they don't crowd each other.
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=35, horizontalalignment='right')


def iterative_imputation(df, target, target_measured):
    print(f'Starting imputation of {target}\n')
    train_len = int(imputation_params['train_length']*len(target_measured.index))

    train_cols = [col for col in df.columns if col != target]
    X = target_measured[train_cols]
    y = target_measured[target]
    indexes, train_sets, test_sets = make_sets(X, y, target, train_len)

    model.set_params(**model_params)
    model.fit(train_sets[0], train_sets[1])
    score = model.score(test_sets[0], test_sets[1])

    complete_set(df, target, target_measured)

    fig, ax = plt.subplots()
    plot_imputation(ax, target, df, target_measured, **imputation_params['fig_params'])
    plt.show()
    iterative_score, df = iteration_loop(df, target,)

    print_output(score, iterative_score)

    fig, ax = plt.subplots()
    plot_imputation(ax, target, df, target_measured, **imputation_params['fig_params'])
    plt.show()

    return df.iloc[:,:-1] # restituisce il dataset originale con la colonna Target completamente imputata

model = MLPRegressor()
model_params = {
        'max_iter': 1000, # max iterations
        'activation': 'relu' ,# activation
        'alpha': 0.01,
        'hidden_layer_sizes': (100,100,100), # hidden layer sizes
        'learning_rate': 'constant', # learning rate
        'solver': 'adam', # solver
    }

imputation_params = {
    'train_length': 4/5,
    'n_to_impute': 100,
    'fig_params': {'figsize': (10,10)},
    }

# # PROGRAMAM STANDALONE
# dateparse = lambda x: dt.strptime(x, '%d/%m/%Y')
# db_d = pd.read_csv('../CSV/db_villabate_deficit.csv', sep=';', decimal=',', parse_dates=['Day'], date_parser=dateparse)

# # SELEZIONE DATI
# # Features complete con abbastanza misure
# cols = ['RHmin', 'RHmax', 'θ 10', 'θ 20', 'θ 30', 'θ 40', 'θ 50', 'θ 60',
#         'U 2m', 'Rshw dwn', 'Tmin', 'Tmax', 'ET0', 'ETa']
# target = 'ETa'

# db_d.index = daysindex = [db_d.index.tolist(), db_d['Day']]
# newindex = pd.MultiIndex.from_arrays(daysindex, names=['Index', 'Day'])

# target_measured = db_d.loc[:, cols].dropna()
# df = db_d.loc[:, cols].dropna(subset = cols[:-1]) # (1153,14) rimuove righe con nan in qualsiasi colonna tranne ETa

# iterative_imputation(df, target, target_measured, **imputation_params)
