#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 06:03:46 2022

@author: Federico Amato
Classe Imputiterator
Fitta su Features X e imputa il set target
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import MODULES.et_functions as et
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class ImputeIterator:

    # Random Generator for reproducible results
    rng = np.random.default_rng(seed=14209)
    mlp_random_state = 4536
    rf_random_state = 256

    # GRID SEARCH
    grid_params = {
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam', 'sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.1],
        }
    # Grid-Search Regressor
    rgr = GridSearchCV(MLPRegressor(random_state=432,
                                    hidden_layer_sizes=(100,100,100),
                                    max_iter=10000), grid_params)

    mlp_params = {
        'hidden_layer_sizes': (50,100,100,50),
        'activation': 'relu',  # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
        'solver': 'adam',  # ‘lbfgs’, ‘sgd’, ‘adam’
        'max_iter': 10000,
        'alpha': 0.1,
        'learning_rate': 'constant',
        'random_state': mlp_random_state,
        }

    model = MLPRegressor(**mlp_params)
    model_final = model
    # model_final = RandomForestRegressor(random_state=rf_random_state,)
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    # Percentuale di dati usati per l'Addestramento dell'Imputazione
    train_fraction = 0.8

    # Array dei punteggi ottenuti durante le Iterazioni
    score_t = []  # Punteggio del Test (step 1)
    score_v = []  # Punteggio della Valutazione (step 2)
    score_mv = []  # Punteggio su un set di Target mai visto (step 4)
    measured_rateo = []  # Percentuale dei punti misurati nei Test
    # Parametri di uscita

    def __init__(self, model=model,
                 train_fraction=train_fraction,
                 invalid_series_limit = 10,
                 iter_limit = 15,
                 output_freq = 10,
                 verbose = False):
        self.model = model
        self.train_fraction = train_fraction
        self.fitted = False
        self.n_invalid_series = 0
        self.n_imputations = 0
        self.invalid_series_limit = invalid_series_limit
        self.iter_limit = iter_limit
        self.output_freq = output_freq
        self.verbose = verbose

        # Placeholders per parametri che si genereranno dopo
        self.idx_free = None
        self.idx_fix = None
        self.idx_visti = None
        self.idx_maivisti = None
        self.X = None
        self.y = None
        self.target_name = None
        self.target = None
        self.features_visti = None
        self.target_visti = None
        print("Benvenuti nell'ImputeIterator")

    def description(self):
        desc = ('Classe Imputiterator\n'
                'Fitta su Features X e imputa il set target.\n'
                '_________\nParametri:\n'
                f'Invalid Series Limit = {self.invalid_series_limit}\n'
                f'Iterations Limit: {self.iter_limit}\n'
                f'Train Fraction: {self.train_fraction}')

        return desc

    def fit(self, X):
        self.fitted = True
        self.X = X

    def impute(self, target, maivisti_size=0.25):
        if self.fitted:
            # Si trasforma la Serie in DataFrame
            if isinstance(target, pd.Series):
                target = target.to_frame()
            self.target = target
            self.target_name = target.columns[0]
            self.maivisti_size = int(len(self.target)*maivisti_size)
            self.max_score = 0

            self.main()
            self.print_ending()
        else:
            raise Exception('You need to fit the Imputer first!')

        return self.y

    def fit_impute(self, X, target, maivisti_size=0.25):
        self.fitted = True
        self.X = X
        self.target = target
        self.target_name = target.columns[0]
        self.maivisti_size = int(len(self.target)*maivisti_size)
        self.max_score = 0

        self.main()
        self.print_ending()
        print(f'{self.n_imputations} Imputations done.\nBest Validation Score: {self.max_score}')
        return self.y

    def __step0(self, idx_free, idx_fix):
        """
        È una prima Validazione:
        Dai dati imputati si cercano di predire quelli misurati.
        Parameters
        ----------
        idx_free: list
            Lista di indici delle Imputazioni di Target
        idx_fix: list
            Lista di indici delle misure vere di Target

        Returns
        -------
        score0 : float
            Punteggio 0 di riferimento.

        """
        # Dei dati visti si prendono tutti quelli Imputati
        # (con indice idx_free)
        # e si usano per l'Addestramento col model.fit(X,y)
        X_train = self.features_visti.loc[idx_free]
        y_train = self.y.loc[idx_free]
        self.model.fit(X_train, y_train)
        # e si fa quindi una Predizione su quelli Misurati
        # (con indice idx_fix)
        X_predict = self.features_visti.loc[idx_fix]
        y_predict = self.model.predict(X_predict)
        # Si calcola lo Score 0 tra Previsti e Misurati
        y_measured = self.y.loc[idx_fix]
        score0 = r2_score(y_measured, y_predict)
        if self.verbose: print(f'Score 0: {score0}')
        return score0

    def __step1(self, idx_free):
        """
        IMPUTAZIONE
        Si fa un Addestramento di un Regressore su una frazione del set
        Completo.
        Si fa una Predizione sul resto del set.
        Si sovrascrivono questi dati Predetti a quelli precedentemente
        Imputati,
        senza toccare quelli Misurati.
        Si ottiene una Serie temporanea di Target, da Validare.
        Parameters
        ----------
        idx_free: list,
            Indice dei dati Imputati che si possono sovrascrivere
        Returns
        -------
        temp : Series
            Serie con nuovi valori imputati dalla Predizione e i valori vecchi,
            Imputati e Misurati.
        test_score : float
            Punteggio della predizione sui dati Misti
            (Misurati e Imputati al passo prima).

        """
        idx_train = self.rng.choice(
            self.features_visti.index,
            size=int(len(self.features_visti.index)*self.train_fraction),
            replace=False)
        idx_predict = [idx for idx in self.features_visti.index
                       if idx not in idx_train]
        # Quanti punti effettivamente misurati stanno nel set di Test?
        predict_in_fix = [idx for idx in idx_predict if idx not in idx_free]
        measured_rateo = len(predict_in_fix)/len(idx_predict)

        X_train = self.features_visti.loc[idx_train]
        y_train = self.y.loc[idx_train]
        X_predict = self.features_visti.loc[idx_predict]
        y_test = self.y.loc[idx_predict]

        self.model.fit(X_train, y_train)
        y_predict = pd.Series(
            self.model.predict(X_predict),
            index=idx_predict, name='Target Imputed')
        # Si calcola il Test Score
        test_score = r2_score(y_test, y_predict)

        # Si filtrano le Predizioni di Target non coincidenti a misure
        y_new = y_predict.filter(items=idx_free)
        # Questi dati imputati devono essere validati.
        # Si usa un set temporaneo di Target imputati e validati,
        temp = self.y
        # completato da questi dati appena imputati
        temp.loc[y_new.index] = y_new

        return temp, test_score, measured_rateo

    def __step2(self, temp, idx_free, idx_fix):
        """
        VALIDAZIONE
        Dai dati imputati si cercano di predire quelli misurati.
        Stavolta i dati appena imputati si trovano in un DataFrame temporaneo.
        Parameters
        ----------
        X : TYPE
            Matrice di features da cui fare predizioni.
        temp : TYPE
            Serie di dati Target imputati e misurati.
        target : TYPE
            Serie di dati Target misurati su cui calcolare il punteggio.

        Returns
        -------
        valid_iteration : bool
            Risultato della Validazione, passata o non passata.

        """
        X_train = self.features_visti.loc[idx_free]
        y_train = temp.loc[idx_free]
        self.model.fit(X_train, y_train)
        X_predict = self.features_visti.loc[idx_fix]
        y_predict = self.model.predict(X_predict)
        # Si calcola lo Score di Validazione
        y_measured = self.y.loc[idx_fix]
        validation_score = r2_score(y_measured, y_predict)
        self.score_v.append(validation_score)
        # Qui comincia la validazione, dove il punteggio è confrontato
        # con quello massimo. Se lo supera, l'iterazione è valida
        valid_iteration = False
        if validation_score > self.max_score:
            valid_iteration = True
            self.max_score = validation_score

        return valid_iteration


    def __step3(self, valid, temp):
        """
        INSERIMENTO
        Se la validazione è buona si possono inserire i Target imputati,
        altrimenti si ignora l'iterazione.
        L'inserimento avviene se il punteggio della nuova iterazione è
        maggiore del precedente.

        Parameters
        ----------
        score : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        temp : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        valid : bool
            Vero se l'imputazione è valida, Falso altimenti.

        """
        if valid:
            self.y = temp
            self.n_invalid_series = 0
            self.n_imputations = self.n_imputations + 1
        else:
            self.n_invalid_series = self.n_invalid_series + 1


    def __step4(self):
        """
        TEST MAIVISTI
        Ultimo test:

        Returns
        -------
        score : TYPE
            DESCRIPTION.
        mse : TYPE
            DESCRIPTION.

        """
        self.model_final.fit(self.features_visti, self.y)
        y_predict_final = self.model_final.predict(self.features_maivisti)

        # Plottare la predizione
        # pd.DataFrame(y_predict_final.ravel(), index=self.target_maivisti.index, columns=['ETa']).reset_index().plot(x='Day',y='ETa', kind='scatter', rot=30)

        score = r2_score(self.target_maivisti, y_predict_final)
        mse = mean_squared_error(self.target_maivisti, y_predict_final)
        return score, mse

    def print_header(self):
        if self.verbose: print(f'\nSCORES:\ni'
                               f'\t{"Test":>14}'
                               f'\t{"Validation":>15}'
                               f'\t{"Final( RMSE )":>15}'
                               f'\t{"Inserted":>}')

    def print_output(self, i, valid_iter, maivisti_mse):
        inserted = u'\u2713' if valid_iter else 'X'
        if self.verbose:
            print(f'{i:<}\t{self.score_t[-1]:>10.4}'
                  f'({self.measured_rateo[-1]:.0%})'
                  f'\t{self.score_v[-1]:>15.4}'
                  f'\t{self.score_mv[-1]:>7.2}({math.sqrt(maivisti_mse):^6.2})'
                  f'\t{inserted:>8}')
        elif not i % self.output_freq:
            print(f'Iteration {i}: {inserted}.')

    def print_ending(self):
        print(f'{self.n_imputations} Imputations done.\n'
              f'Best Validation Score: {self.max_score}\n'
              f'Best MaiVisti Score: {max(self.score_mv)}')

    @staticmethod
    def scale(df):
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df),
                          index=df.index,
                          columns=df.columns)
        return df

    @staticmethod
    def scale_series(s, name='Serie'):
        scaler = StandardScaler()
        s = pd.Series(
            scaler.fit_transform(s.values.reshape(-1, 1)).reshape(len(s)),
            index=s.index, name=name
            )
        return s

    # %% MAIN PROGRAM

    def main(self):
        features_names = [cols for cols in self.X.columns
                          if cols != self.target_name]
        # Si blocca un certo numero di misure di Target
        # da non usare nell'Imputazione,
        # per usarle poi come controllo nel Test Maivisti
        idx_maivisti = self.rng.choice(self.target.index,
                                       size=self.maivisti_size,
                                       replace=False)
        self.target_maivisti = self.target.loc[idx_maivisti]
        # Si prendono le date di queste misure bloccate
        # E si tirano via di conseguenza le features legate a questi Target
        self.features_maivisti = self.X.loc[idx_maivisti, features_names]
        # Il resto del Target potrà essere usato (visto) per l'Imputazione
        idx_visti = [idx for idx in self.target.index
                     if idx not in idx_maivisti]
        self.target_visti = self.target.loc[idx_visti]

        # Si prendono quindi tutte le rimanenti date
        idx_visti = [idx for idx in self.X.index if idx not in idx_maivisti]
        # E si prendono le features legate a queste
        self.features_visti = self.X.loc[idx_visti, features_names]
        # E anche i Target (Misurati e Imputati in creazione)
        self.y = self.X.loc[idx_visti, self.target_name]

        # Si fa uno Scaling per avere previsioni migliori
        self.features_visti = self.scale(self.features_visti)
        self.features_maivisti = self.scale(self.features_maivisti)
        self.target_maivisti = self.scale(self.target_maivisti)
        self.target_visti = self.scale(self.target_visti)
        self.y = self.scale_series(self.y, 'Target Imputed')

        # Date con misure del Target: non dovranno essere toccate
        idx_fix = self.target_visti.index
        # Le restanti date sono libere di essere sovrascritte nell'Imputazione
        idx_free = [idx for idx in self.features_visti.index
                    if idx not in idx_fix]

        # STEP 0 - PRIMO SCORE
        score0 = self.__step0(idx_free, idx_fix)
        self.score_v.append(score0)

        # Stampa Output iniziale
        self.print_header()

        # CICLO DI IMPUTAZIONE
        # Variabili di controllo del ciclo
        i = 0  # Cicli eseguiti

        while (self.n_invalid_series < self.invalid_series_limit
               and i < self.iter_limit):
            # STEP 1 - IMPUTAZIONE
            temp, test_score, rateo = self.__step1(idx_free)
            self.score_t.append(test_score)
            self.measured_rateo.append(rateo)
            # STEP 2 - VALIDAZIONE
            valid_iter = self.__step2(temp, idx_free, idx_fix)

            # STEP 3 - INSERIMENTO
            self.__step3(valid_iter, temp)

            # STEP 4 - FINAL SCORE
            # Punteggio su una frazione di Target mai vista
            maivisti_score, maivisti_mse = self.__step4()
            self.score_mv.append(maivisti_score)

            # Output finale con i tre punteggi
            self.print_output(i, valid_iter, maivisti_mse)
            # Continua con le iterazioni
            i = i + 1

        y = self.y.values.reshape(-1, 1)
        y = StandardScaler().fit(self.target).inverse_transform(y).ravel()
        self.y = pd.Series(y, name='Target Imputed', index=idx_visti)

    def plot_imputation(self, **kwargs):
        # Si riprendono le misure non scalate del Target
        target_visti = self.target.loc[self.target_visti.index]
        target_maivisti = self.target.loc[self.target_maivisti.index]
        # E i soli dati Imputati
        y_imputed = self.y[~self.y.index.isin(self.target.index)]

        fig, ax = plt.subplots()
        fig.suptitle(f'Iterative imputation of {self.target_name} '
                     f'for {self.iter_limit} iterations')
        # idx_imputed = self.y.join(self.target)
        et.plot_axis(
            ax, [y_imputed.index, y_imputed.values],
            plot_type='scatter', alpha=0.4)
        et.plot_axis(
            ax, [target_maivisti.index, target_maivisti.values, 'red'],
            legend='MaiVisti', plot_type='scatter')
        et.plot_axis(
            ax, [target_visti.index, target_visti.values, 'green'],
            legend='Visti', plot_type='scatter')
