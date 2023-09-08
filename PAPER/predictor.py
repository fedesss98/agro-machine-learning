#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 11:43:03 2021

@author: Federico Amato

Applicazione dei modelli di Machine Learning per predire l'Evapotraspirazione
di Riferimento ET0.
Le features utili sono già state selezionate in diversi modi.

"""
import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.impute import KNNImputer

from MODULES import imputiterator as ii

# Modelli
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from datetime import datetime as dt


def clear_log(LOG, n=-1):
    try:
        csv = pd.read_csv(LOG, sep=';')
    except:
        print("LOG File not found...")
    if n<0:
        csv.drop(csv.index, inplace=True)
    else:
        csv.drop(csv.tail(n).index, inplace=True)

    csv.reset_index(drop=True, inplace=True)
    csv.to_csv(LOG, sep=';', index=False)

    # print("Lines in Log file after cleaning:")
    # print(csv)


def write_log(inp, outp, results_df, LOG=False):
    print("\nTotal Analys:\n")
    print(results_df)
    if LOG:
        results_df.insert(1, 'Columns', str(inp['COLUMNS']))
        results_df.insert(2, 'Target', inp['TARGET'])
        results_df.insert(9, 'Imputation Method', inp['METHOD'][0])
        results_df.insert(10, 'PreProcessing', inp['PROCESSING'][0])
        model_params_list = make_list_from_params(results_df['Model'])
        results_df.insert(11, 'Parameters', model_params_list)
        results_df.reset_index(drop=True, inplace=True)

        try:
            csv = pd.read_csv(LOG, sep=';')
            results_df = csv.append(results_df, ignore_index=True)
            # print("Lines in Log file:")
            # print(results_df)
        except:
            print(f"LOG File not found...\nCreating the new LOG file: '{LOG}'")

        results_df.to_csv(LOG, sep=';', index=False)

def make_list_from_params(models):
    return [model_params[model] for model in models]


def extract_dataset(path):
    def dateparse(x): return dt.strptime(x, '%d/%m/%Y')
    db = pd.read_csv(path, sep=';', decimal=',', parse_dates=[
                     'Day'], date_parser=dateparse)
    db.index = [db.index.tolist(), db['Day']]
    try:
        db['P'] = db['P'].fillna(0)
        db['I'] = db['I'].fillna(0)
    except:
        pass
    return db

def make_dataframe(path, columns, method=['drop'], *args):
    db = extract_dataset(path)
    df = db.loc[:,columns]
    if method[0] == 'drop':
        print('Dropping')
        df = df.dropna().reset_index(drop=True)
    elif method[0] == 'impute':
        print('Automatic Imputation')
        k = method[1]
        imputer = KNNImputer(n_neighbors=k, weights="uniform")
        df = pd.DataFrame(imputer.fit_transform(df), columns=columns)
    elif method[0] == 'iterative_impute':
        print('Iterative Imputation')
        target = args[0]
        target_measured = df.dropna()
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
        cols = [col for col in columns if col != target]
        df[cols] = imputer.fit_transform(df[cols])
        df = ii.iterative_imputation(df, target, target_measured)
    else:
        raise Exception("Imputer not valid.")
    return df


def process_data(df, method, *df2):
    if method[0] == 'scale':
        dp = scale_data(df, method[1])
    elif(method[0] == 'rescale'):
        dp = rescale_data(df, method[1], df2)
    else:
        raise Exception("process_data() isn't ready to do other than scale")
    return dp


def average_feature(df, features):
    df = df.loc[:, features].mean(axis=1)
    return df


def rescale_data(data_to_scale, scaler, data_to_fit):
    scaler.fit(np.array(data_to_fit).reshape(-1, 1))
    return scaler.inverse_transform(data_to_scale)


def scale_data(data_to_scale, scaler):
    if not isinstance(data_to_scale, (pd.DataFrame, pd.core.series.Series)):
        raise Exception(
            "Function scale_data() expect a DataFrame as third argument")
    else:
        data_scaled = pd.DataFrame(scaler.fit_transform(data_to_scale),
                                   columns=in_params['COLUMNS'])
    return data_scaled


def make_filename(path):
    filename = path.replace(
        '../CSV/', '').replace('db_villabate_', '').replace('.csv', '')
    filename = ''.join((out_params['PATH'],
                        filename,
                        out_params['FILE_SUFFIX'],
                        '.', out_params['FILE_PROPS']['format']))
    return filename

def init_model(model='mlp'):
    if (model == 'MLP'):
        model = MLPRegressor(**model_params['MLP'])
    elif (model == 'LR'):
        model = LinearRegression(**model_params['LR'])
    elif (model == 'SVM'):
        model = SVR(**model_params['SVM'])
    elif (model == 'RF'):
        model = RandomForestRegressor(**model_params['RF'])
    return model

def predict_target(df, model):
    random_indexes = df.index.tolist()
    np.random.shuffle(random_indexes)
    train_indexes = random_indexes[:700]
    test_indexes = random_indexes[700:]

    cols = [col for col in in_params['COLUMNS'] if col != in_params['TARGET']]

    X_train = df.loc[train_indexes, cols].values
    y_train = df.loc[train_indexes, in_params['TARGET']].values
    X_test = df.loc[test_indexes, cols].values
    y_test = df.loc[test_indexes, in_params['TARGET']].values

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    prediction = model.predict(df.iloc[:,:-1].values)

    return prediction, score

def print_warning(path):
    filename = make_filename(path)

    if os.path.exists(filename):
        message = '\n'.join((f'Sicuro di voler sovrascrivere il file {filename}?',
                             'Premere Invio per continuare o inserire un imput per annullare:'))

    else:
        message = '\n'.join((f'Sicuro di voler salvare il nuovo file {filename}?',
                                 'Premere Invio per continuare o inserire un imput per annullare:'))
    return not (input(message))

def print_model_output(X, y, score, filename, model, out_params):
    reg = LinearRegression(fit_intercept=False)
    X= X.values.reshape(-1,1)
    m = reg.fit(X,y).coef_[0] # result 0
    R2 = reg.score(X,y) # result 1
    MSE = mean_squared_error(X,y) # result 2
    MXE = max_error(X, y) # result 3

    fig = plot_regression(X, y, [m, R2, MSE, MXE], model, filename, **out_params)
    plt.show(fig)
    plt.close(fig)

    print(f"{model} score for Deficit Field: {score}")
    print(f" - mean squared error: {MSE}")

    return R2, MSE, MXE

def plot_regression(X, y, results, model, path, **out_params):

    title = path.replace('../CSV/','').replace('db_villabate_','').replace('.csv','')
    textstr = '\n'.join((
        f'Score of Linear Regression with "{model}":',
        f'$R^2$ = {results[1]:.3f}',
        'Model Results on Test Set:',
        f'MSE = {results[2]:.3f}',
        f'MXE = {results[3]:.3f}',
        ))
    props = out_params['TEXT_PROPS']

    fig, ax = plt.subplots(figsize=out_params['FIGSIZE'])

    ax.scatter(X, y, alpha = 0.3, c=out_params['PLOT_COLOR'])
    ax.plot(np.linspace(X.min(), X.max()), results[0]*np.linspace(X.min(), X.max()),
             color = out_params['LINE_COLOR'],)
    ax.grid()

    ax.set_xlabel(f"Observed {in_params['TARGET']}")
    ax.set_ylabel(f"Simulated {in_params['TARGET']}")
    ax.text(0.05,0.7, textstr, transform=ax.transAxes, bbox = props)
    ax.set_title(title + out_params['FILE_SUFFIX'])

    if out_params['SAVE_CHECK']:
        save_plot(path, fig)

    return fig

def save_plot(path, fig):
    filename = make_filename(path)
    fig.savefig(filename, **out_params['FILE_PROPS'])

    return fig

def start_prediction(PATHS, out_params, **in_params):
    results_columns = ['Path', 'Model', 'Score', 'Mean_Squared_Error', 'Max_Error', 'R2']
    results = pd.DataFrame(columns = results_columns)
    for path in PATHS:
        print(f'Analyzing {path}')

        path_results = main(path, out_params, **in_params)

        print(f'\nFinal results for {path}:\n')
        print(path_results)
        print('\n___________________\n')

        path_results.insert(0, 'Path', path)
        filename = make_filename(path) if out_params['SAVE_CHECK'] else None
        path_results.insert(6, 'FigName', filename)

        results = results.append(path_results)

    return results

def main(path, out_params, **in_params):
    if CHECK_SAVE and SAVE:
        out_params['SAVE_CHECK'] = print_warning(path)
    else:
        out_params['SAVE_CHECK'] = False

    df = make_dataframe(path, in_params['COLUMNS'], in_params['METHOD'], in_params['TARGET'])

    result_columns = ['Model', 'Score', 'Mean_Squared_Error', 'Max_Error', 'R2']
    results = pd.DataFrame(columns = result_columns)

    if in_params['PROCESSING']:
        dp = process_data(df, in_params['PROCESSING'])

        for model in in_params['MODEL']:
            predictor = init_model(model)
            predicted, score = predict_target(dp, predictor)
            # Scaling Inverso
            method = ['rescale', StandardScaler()]
            out_params['FILE_SUFFIX'] = '_rescaled'

            X = df[in_params['TARGET']].values.reshape(-1,1)
            y = predicted.reshape(-1,1)
            y = process_data(y, method, X)
            r2, mse, mxe = print_model_output(df[in_params['TARGET']], y, score, path, model, out_params)
            model_results = pd.DataFrame([[model, score, mse, mxe, r2]],
                                         columns = result_columns)
            results = results.append(model_results)

        results.sort_values(by=['Score'], inplace=True, ascending=False, ignore_index=True)


    else:
        for model in in_params['MODEL']:
            predictor = init_model(in_params['MODEL'])
            predicted, score = predict_target(df, predictor)
            r2, mse, mxe = print_model_output(df[in_params['TARGET']], predicted, score, path, model, out_params)
            model_results = pd.DataFrame([[model, score, mse, mxe, r2]],
                                         columns = result_columns)
            results = results.append(model_results)

        results.sort_values(by=['Score'], inplace=True, ignore_index=True)


    return results

# Percorsi da cui estrarre i file da analizzare
PATHS = [
        '../CSV/db_villabate_deficit.csv',
        # '../CSV/db_villabate_full.csv',
        # '../CSV/deficit_seasonal_trends.csv',
        # '../CSV/deficit_seasonal_seasonal.csv',
        # '../CSV/deficit_seasonal_noise.csv',
        # '../CSV/full_seasonal_trends.csv',
        # '../CSV/full_seasonal_seasonal.csv',
        # '../CSV/full_seasonal_noise.csv',
        ]

# Parametri in input per calibrare il programma
in_params = {
        "TARGET": "ETa", # Colonna del dataset da predire
        "COLUMNS": [
            'RHmin',
            'RHmax',
            'θ 10',
            'θ 20',
            'θ 30',
            'θ 40',
            'θ 50',
            'θ 60',
            'U 2m',
            'Rshw dwn',
            'Tmin',
            'Tmax',
            'ET0',
            'ETa'
            ], # Colonne da estrarre dal file .csv
        "METHOD": ['iterative_impute',], # {['drop'], ['impute', n_neighbors], ['iterative_impute']}
        "PROCESSING": ['scale', StandardScaler()], # {False, ['scale', scaler], ['avg', [cols_to_average], [cols2_to_average],...]}
        "MODEL": ['SVM', 'RF', 'LR', 'MLP'], # {['MLP], ['LR'], ['SVM'], ['RF'], o lista di questi}
    }

out_params = {
    'FIGSIZE': (10,8),
    'PLOT_COLOR': 'seagreen',
    'LINE_COLOR': 'limegreen',
    'TEXT_PROPS': {
        'facecolor': 'white',
        'edgecolor': 'lightgray',
        },
    'TEXT_SIZE': 14,
    'SAVEFIG': False,
    'PATH': './PLOTS/',
    'TITLE': 'deficit',
    'FILE_SUFFIX': '',
    'FILE_PROPS': {
        'dpi': 180,
        'format': 'png', # {'eps','jpg','jpeg','pdf','pgf','png','ps''raw','rgba','svg','svgz','tif','tiff'}
        'bbox_inches': 'tight',
        # 'transparent': False, # PostScript backend does not support transparency
        },
    'SAVE_CHECK': True,
    }


model_params = {
    'MLP': {
        'hidden_layer_sizes': (100,100,100), # tuple
        'activation': 'relu', #{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        'solver': 'adam', # {‘lbfgs’, ‘sgd’, ‘adam’}
        'alpha': 0.01, # float; L2 penalty
        'learning_rate': 'constant', # {‘constant’, ‘invscaling’, ‘adaptive’}
        'max_iter': 1000, # int
        'shuffle': False,
        'verbose': False,
        },

    'SVM': {
        'kernel': 'rbf', # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
        'degree': 3,
        'gamma': 'scale',
        'tol': 1e-3, # float; tolerance for stopping criterion
        'verbose': False,
        'max_iter': -1, # int, or -1 for no limit
        },

    'LR': {
        'fit_intercept': True,
        'positive': False,
        },

    'RF': {
        'n_estimators': 100,
        # 'criterion': 'absolute_error', # {“squared_error”, “absolute_error”, “poisson”}
        'max_depth': None, # int
        'min_samples_split': 2, # int or float, minimum number of samples required to split an internal node
        'verbose': 0, # int
        },
    }

SAVE = False     # Salva il plot in output
CHECK_SAVE = True   # Verifica di voler salvare o sovrascrivere il plot in output
# LOG = './LOG/log.csv' # Nome del file di log o False
LOG = False

results = start_prediction(PATHS, out_params, **in_params)
write_log(in_params, out_params, results, LOG)

# Se vuoi eliminare le ultime n righe del file di log de-commenta la riga sotto:
# clear_log(LOG, 3) # n: int o -1 (default) per eliminare tutto il file
