#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 06:03:46 2022

@author: Federico Amato
Classe Imputiterator
Fitta su Features X e imputa il set target

Risultati con il random_state del MLP
(sia MLP_RAND_STATE che first_mlp('random_state') )
ottenuti dal programma ETa_iterativeimputation_models.py
usando solo il primo Modello, con paramteri:
    KFOLDS = 2
    ITER_LIMIT = 1000
    INVALID_LIM= 7000

Rand - Best MaiVisti [iterazione 0 - imputazione iterativa]
58: [0.8735 - 0.8078]
980: [0.8805 - 0.7922]
12: [0.8698 - 0.8241]

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

import MODULES.et_functions as et
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class ImputeIterator:

    # Random Generator for reproducible results
    RNG = np.random.default_rng(seed=6475)
    MLP_RAND_STATE = 12
    RF_RAND_STATE = 256
    EPOCHS = 0

    first_mlp = MLPRegressor(
        hidden_layer_sizes=(10, 10, 10),  # (365, 365)
        random_state=12,
        max_iter=10000,  # 500
        warm_start=False,
        )

    mlp_params = {
        "hidden_layer_sizes": (10, 10, 10),  # (100,100,100)
        "activation": "relu",  # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
        "solver": "adam",  # ‘lbfgs’, ‘sgd’, ‘adam’
        "max_iter": 10000,
        "alpha": 0.0001,
        "learning_rate": "constant",
        # "learning_rate_init": 0.0002,
        # 'warm_start': True,
        "random_state": MLP_RAND_STATE,
    }

    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    # Percentuale di dati usati per l'Addestramento dell'Imputazione
    train_fraction = 0.8

    def __init__(
        self,
        model=None,
        train_fraction=train_fraction,
        inv_series_lim=10,
        iter_limit=15,
        output_freq=10,
        verbose=False,
    ):
        self.train_fraction = train_fraction
        self.fitted = False
        self.n_inv_series = 0
        self.n_imputations = 0
        self.inv_series_lim = inv_series_lim
        self.iter_limit = iter_limit
        self.output_freq = output_freq
        self.verbose = verbose

        # Placeholders per parametri che si genereranno dopo
        self.idx_free = None
        self.idx_fix = None
        self.idx_visti = None
        self.idx_maivisti = None
        self.idx_predict=None
        self.X = None
        self.y = None
        self.target_name = None
        self.target = None
        self.fts_visti = None
        self.target_visti = None

        # first_model = RandomForestRegressor(n_estimators=365,random_state=58)
        self.first_model = self.first_mlp
        if model is None:
            self.model = self.first_mlp
        else:
            self.model = model
        self.model_final = model
        self.model_test = RandomForestRegressor(
            random_state=self.RF_RAND_STATE,)

        # Array dei punteggi ottenuti durante le Iterazioni
        self.score_fit = []  # Punteggio in addestramento
        self.score_t = np.zeros((1, 3))  # Punteggio del Test (step 1)
        self.score_v = []  # Punteggio della Valutazione (step 2)
        self.score_mv = []  # Punteggio su un set di Target mai visto (step 4)
        self.measured_rateo = np.zeros((1, 1))  # Punti misurati nei Test

        print("Benvenuti nell'ImputeIterator")

    def description(self):
        desc = (
            "Classe Imputiterator\n"
            "Fitta su Features X e imputa il set target.\n"
            "_________\nParametri:\n"
            f"Invalid Series Limit = {self.inv_series_lim}\n"
            f"Iterations Limit: {self.iter_limit}\n"
            f"Train Fraction: {self.train_fraction}"
        )

        return desc

    def fit(self, features, target, model=None):
        self.fitted = True
        self.X = features
        # Si trasforma la Serie in DataFrame
        if isinstance(target, pd.Series):
            target = target.to_frame()
        self.target = target
        test_size = 0.25
        if model is None:
            model = self.first_model
        X_train, X_test, y_train, y_test = train_test_split(
            features.loc[target.index], target,
            test_size=test_size, random_state=58,
        )

        # SCALING: dopo questo si avranno numpy array
        X_train, y_train, X_test, y_test = self.scale_sets(
            X_train, y_train, X_test, y_test)
        # Addestramento
        print(f"Training Model on {1-test_size:.0%} of Measured data")
        model.fit(X_train, y_train)
        y_fit = model.predict(X_train)
        r2_fit = r2_score(y_train, y_fit)
        mse_fit = mean_squared_error(y_train, y_fit)
        self.score_fit.append([r2_fit, mse_fit])
        self.first_model = model

    def impute(self, target,
               maivisti_size=0.25, idx_maivisti=None, idx_predict=None):
        if self.fitted:
            self.target_name = target.columns[0]
            self.maivisti_size = int(len(self.target) * maivisti_size)
            self.idx_maivisti = idx_maivisti
            self.idx_predict = idx_predict
            self.max_score = 0
            self.iterative_imputation()
            self.__print_ending()
        else:
            raise Exception("You need to fit the Imputer first!")

        return self.y

    def fit_imputed(self, features, target, maivisti_size=0.25):
        self.fitted = False
        self.X = features
        self.target = target
        self.target_name = target.columns[0]
        self.maivisti_size = int(len(self.target) * maivisti_size)
        self.max_score = 0

        self.iterative_imputation()
        self.__print_ending()

        return self.y

    def first_imputation(self, df_to_impute, idx_free, idx_fix, idx_maivisti):
        if self.fitted:
            refit = False
            y_imputed = self.first_model.predict(df_to_impute.values)
            y_imputed = pd.DataFrame(y_imputed,
                                     columns=['ETa'],
                                     index=idx_free
                                     )
        else:
            refit = True
            fts_and_trgt = pd.concat([self.fts_visti, self.target_visti],
                                     axis=1)
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            fts_and_trgt = pd.DataFrame(imputer.fit_transform(fts_and_trgt),
                                        columns=fts_and_trgt.columns,
                                        index=fts_and_trgt.index
                                        )
            y_imputed = fts_and_trgt.loc[idx_free, 'ETa'].to_frame()

        self.y = pd.concat([y_imputed, self.target_visti], axis=0)
        score0, mse0, mbe0 = self.test_on_maivisti(self.first_model,
                                                   self.fts_visti, self.y,
                                                   refit=refit)
        self.plot_target(idx_fix, idx_free, idx_maivisti,
                         title=f'First Imputation: $R^2$ = {score0}')
        self.score_mv.append([score0, mse0, mbe0])
        # self.max_score = score0

        # Stampa Output iniziale
        self.__print_header()

    def __impute_anew(self, idx_free):
        """
        IMPUTAZIONE
        Si addestra un Regressore su una frazione del set Completo.
        Si fa una Predizione sul resto del set.
        Si sovrascrivono i dati Predetti a quelli precedentemente Imputati,
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
        measured_rateo : float
            Percentuale di dati Misurati nel set di Test

        """
        # TODO prenderli dalle misure di Target ottenute:
        # Al posto di prendere l'index di train si prende quello di test,
        # che deve contenere solo dati di Target Misurato.
        if self.idx_predict is not None:
            predict_len = int(
                len(self.fts_visti.index) * (1-self.train_fraction)
                )
            idx_predict_imputed = self.RNG.choice(
                idx_free,
                size=int(predict_len/2),
                replace=False,
            )
            idx_predict_measured = self.idx_predict[:int(predict_len/2)]

            idx_predict = np.append(idx_predict_imputed, idx_predict_measured)
            idx_train = [
                idx for idx in self.fts_visti.index if idx not in idx_predict
                ]
        else:
            idx_train = self.RNG.choice(
                self.fts_visti.index,
                size=int(len(self.fts_visti.index) * self.train_fraction),
                replace=False,
            )
            idx_predict = [
                idx for idx in self.fts_visti.index if idx not in idx_train
                ]
        # Quanti punti effettivamente misurati stanno nel set di Test?
        predicted_measures = [
            idx for idx in idx_predict if idx not in idx_free
            ]
        measured_rateo = len(predicted_measures) / len(idx_predict)

        X_train = self.fts_visti.loc[idx_train]
        y_train = self.y.loc[idx_train]
        X_predict = self.fts_visti.loc[idx_predict]
        y_test = self.y.loc[idx_predict]

        y_predict, test_score = self.internal_test([X_train, y_train],
                                                   [X_predict, y_test],
                                                   predicted_measures)

        # Si filtrano le Predizioni di Target non coincidenti a misure
        y_new = y_predict.filter(items=idx_free)
        # Questi dati imputati devono essere validati.
        # Si usa un set temporaneo di Target imputati e validati,
        temp = self.y.copy()
        # completato da questi dati appena imputati
        temp.loc[y_new.index] = y_new.to_frame()

        return temp, test_score, measured_rateo

    def internal_test(self, train, test, subset=None):
        """
        Prova l'imputazione con un test sui dati visti.
        Se viene fornito un subset, calcola i punteggi solo in quel subset.
        """
        X_train, y_train = train
        X_test, y_test = test
        self.model.fit(X_train.to_numpy(), y_train.values.ravel())
        y_fit = self.model.predict(X_train.to_numpy())
        r2_fit = r2_score(y_train, y_fit)
        mse_fit = mean_squared_error(y_train, y_fit)
        self.score_fit.append([r2_fit, mse_fit])
        y_predict = self.model.predict(X_test.to_numpy())
        y_predict = pd.Series(y_predict,
                              index=X_test.index, name="Target Imputed")
        if subset is not None:
            y_test_measures = y_test.loc[subset]
            y_predict_measures = y_predict.loc[subset]
            test_score = r2_score(y_test_measures, y_predict_measures)
            test_mse = mean_squared_error(y_test, y_predict)
            test_mbe = et.mean_bias_error(y_test, y_predict)
            test_score = [[test_score, test_mse, test_mbe]]
        else:
            test_score = r2_score(y_test, y_predict)
            test_mse = mean_squared_error(y_test, y_predict)
            test_mbe = et.mean_bias_error(y_test, y_predict)
            test_score = [[test_score, test_mse, test_mbe]]
        return y_predict, test_score

    def __check_valid_iteration(self, y_temp):
        """
        VALIDAZIONE
        Esegue un test MaiVisti sul DataFrame temporaneo di dati imputati.

        Parameters
        ----------
        y_temp : DataFrane
            DataFrame temporaneo di dati imputati.

        Returns
        -------
        valid_iteration : bool
            Indica se l'iterazione che ha prodotto y_temp è valida o meno.

        """
        valid_iteration = False
        score, mse, mbe = self.test_on_maivisti(self.model,
                                                self.fts_visti, y_temp)
        self.score_mv.append([score, mse, mbe])
        if score > self.max_score:
            valid_iteration = True
            self.max_score = score
        return valid_iteration

    def __insert_or_not(self, temp, valid):
        """
        INSERIMENTO
        Se la validazione è buona si possono inserire i Target imputati,
        altrimenti si ignora l'iterazione.
        L'inserimento avviene se il punteggio della nuova iterazione è
        maggiore del precedente.

        Parameters
        ----------

        temp : DataFrame
            DESCRIPTION.
        valid : bool
            DESCRIPTION.

        """
        if valid:
            self.y = temp
            self.n_inv_series = 0
            self.n_imputations = self.n_imputations + 1
        else:
            self.n_inv_series = self.n_inv_series + 1

    def test_on_maivisti(self, model, X_train, y_train, refit=True):
        """
        VALIDAZIONE
        Il modello si addestra su tutti i dati visti e imputati e cerca di
        predire quelli MaiVisti.

        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        X_train : TYPE
            DESCRIPTION.
        y_train : TYPE
            DESCRIPTION.

        Returns
        -------
        score : TYPE
            DESCRIPTION.
        mse : TYPE
            DESCRIPTION.
        mbe : TYPE
            DESCRIPTION.

        """
        # Addestra sui dati Visti
        if refit:
            model.set_params(warm_start=False)

            model.fit(X_train.to_numpy(), y_train.values.ravel())
            # print('fit score maivisti')
            # self.score_fit.append(
            #     model.score(X_train.to_numpy(), y_train.values.ravel()))
        # Predice i dati MaiVisti
        X_predict = self.fts_maivisti
        y_predicted = model.predict(X_predict.values)
        # Calcola i punteggi
        score = r2_score(self.target_maivisti, y_predicted)
        mse = mean_squared_error(self.target_maivisti, y_predicted)
        mbe = et.mean_bias_error(self.target_maivisti, y_predicted)
        return score, mse, mbe

    # %% MAIN PROGRAM

    def iterative_imputation(self):
        # Si blocca un certo numero di misure di Target
        # da non usare nell'Imputazione,
        # per usarle poi come controllo nel Test Maivisti
        if self.idx_maivisti is not None:
            idx_maivisti = self.idx_maivisti
        else:
            idx_maivisti = self.RNG.choice(
                self.target.index, size=self.maivisti_size, replace=False
            )
            self.idx_maivisti = idx_maivisti
        # print('MaiVisti dates:\n'
        #       f'{list(pd.to_datetime(idx_maivisti).dayofyear)}')
        target_maivisti = self.target.loc[idx_maivisti]
        # Si prendono le date di queste misure bloccate
        # E si tirano via di conseguenza le features legate a questi Target
        fts_maivisti = self.X.loc[idx_maivisti]
        # Il resto del Target potrà essere usato (visto) per l'Imputazione
        # Queste date con misure del Target non dovranno essere toccate
        idx_fix = [idx for idx in self.target.index if idx not in idx_maivisti]
        target_visti = self.target.loc[idx_fix]
        # Si prendono quindi tutte le rimanenti date
        idx_visti = [idx for idx in self.X.index if idx not in idx_maivisti]
        # E si prendono le features legate a queste
        fts_visti = self.X.loc[idx_visti]
        # E anche i Target (Misurati e Imputati in creazione)
        # y = self.X.loc[idx_visti, self.target_name]
        # Le restanti date sono libere di essere sovrascritte nell'Imputazione
        # E sono tutte quelle Viste che non corrispondono a Misure
        idx_free = [idx for idx in fts_visti.index if idx not in idx_fix]
        # Su queste si dovranno imputare i dati
        fts_impute = fts_visti.loc[idx_free]

        # Si fa uno Scaling per avere previsioni migliori
        self.fts_visti = self.scale(fts_visti)
        self.fts_maivisti = self.scale(fts_maivisti)
        self.target_maivisti = self.scale(target_maivisti)
        self.target_visti = self.scale(target_visti)
        fts_impute = self.scale(fts_impute)
        # self.y = self.scale_series(y, "Target Imputed")

        # STEP 0 - PRIMA IMPUTAZIONE
        # Si predice (imputazione) sui dati Visti e non Misurati
        # e si trasforma in DataFrame
        self.first_imputation(fts_impute, idx_free, idx_fix, idx_maivisti)

        # CICLO DI IMPUTAZIONE
        # Variabili di controllo del ciclo
        i = 0  # Cicli eseguiti

        while self.n_inv_series < self.inv_series_lim and i < self.iter_limit:
            # STEP 1 - IMPUTAZIONE
            temp, test_score, rateo = self.__impute_anew(idx_free)
            self.score_t = np.append(self.score_t, test_score, axis=0)
            self.measured_rateo = np.append(self.measured_rateo, rateo)
            # STEP 2 - VALIDAZIONE
            valid_iter = self.__check_valid_iteration(temp)
            # if valid_iter:
            #     self.plot_target(idx_fix, idx_free, idx_maivisti,
            #                      title=f'Imputation {i}: $R^2$ = {test_score}')
            # STEP 3 - INSERIMENTO
            self.__insert_or_not(temp, valid_iter)

            # Output finale con i tre punteggi
            if not i % self.output_freq or valid_iter:
                self.__print_output(i, valid_iter)

            # Continua con le iterazioni
            i = i + 1
        mse_scores = np.array(self.score_mv)[:, 1]
        mbe_scores = np.array(self.score_mv)[:, 2]
        self.min_mse = min(mse_scores[1:])
        self.min_mbe = min(mbe_scores[1:], key=abs)
        # RESCALING
        self.y.sort_index(inplace=True)
        y = self.y.values.reshape(-1, 1)
        y = StandardScaler().fit(self.target).inverse_transform(y).ravel()
        self.y = pd.Series(y, name="Target Imputed", index=idx_visti)


    def plot_target(self, idx_fix, idx_free, idx_maivisti, **kwargs):
        target_total = pd.concat([self.y, self.target_maivisti]).sort_index()
        # Si crea una colonna di etichette per i dati di ETa
        target_total['source'] = [
            'Misurati' if i in idx_fix else 'Mai Visti' if i in idx_maivisti
            else 'Imputed' for i in target_total.index]
        target_total.index.name = 'Day'

        sns.relplot(
            height=5, aspect=1.61,
            data=target_total.sort_values('source'),
            x='Day', y='ETa',
            style='source', hue='source')
        plt.xticks(rotation=90)
        plt.title(kwargs.get('title') if 'title' in kwargs else 'Imputation')
        plt.show()

    def plot_imputation(self, **kwargs):
        # Si riprendono le misure non scalate del Target
        target_visti = self.target.loc[self.target_visti.index]
        target_maivisti = self.target.loc[self.target_maivisti.index]
        # E i soli dati Imputati
        y_imputed = self.y[~self.y.index.isin(self.target.index)]

        fig, ax = plt.subplots()
        fig.suptitle(
            f"Iterative imputation of {self.target_name} "
            f"for {self.iter_limit} iterations"
        )
        # idx_imputed = self.y.join(self.target)
        et.plot_axis(
            ax, [y_imputed.index, y_imputed.values],
            legend="Imputed",
            plot_type="scatter", alpha=0.4, date_ticks=4,
        )
        et.plot_axis(
            ax,
            [target_maivisti.index, target_maivisti.values, "red"],
            legend="MaiVisti",
            plot_type="scatter",
        )
        et.plot_axis(
            ax,
            [target_visti.index, target_visti.values, "green"],
            legend="Visti",
            plot_type="scatter",
        )
        ax.legend()

    def __print_header(self):
        print(f'First MaiVisti Scores:\n'
              f'{"R2":7}\t{"RMSE":7}\t{"MBE":7}\n'
              f'{self.score_mv[0][0]:0.4}\t'
              f'{math.sqrt(self.score_mv[0][1]):0.4}\t'
              f'{self.score_mv[0][2]:0.4}')
        if self.verbose:
            print(
                f"\nSCORES:\ni"
                f'\t{"Internal Test R2":>17}'
                f'\t{"MaiVisti R2":>15}'
                f'\t{"MaiVisti RMSE":>15}'
                f'\t{"MaiVisti MBE":>15}'
                f'\t{"Inserted":>8}'
            )

    def __print_output(self, i, valid_iter):
        inserted = "\u2713" if valid_iter else "X"
        if self.verbose:
            print(
                f'{i:<}\t{self.score_t[-1, 0]:>13.4}'
                f'({self.measured_rateo[-1]:.0%})'
                f'\t{self.score_mv[-1][0]:>11.4}'
                f'\t\t{math.sqrt(self.score_mv[-1][1]):>11.4}'
                f'\t\t{self.score_mv[-1][2]:>11.4}'
                f'\t{inserted:>8}'
                )
        elif valid_iter:
            print(
                f'{i:<}\t{self.score_t[-1, 0]:>13.4}'
                f'({self.measured_rateo[-1]:.0%})'
                f'\t{self.score_mv[-1][0]:>11.4}'
                f'\t\t{math.sqrt(self.score_mv[-1][1]):>11.4}'
                f'\t\t{self.score_mv[-1][2]:>11.4}'
                f'\t{inserted:>8}'
                )

    def __print_ending(self):
        print(
            f"{int(self.n_imputations)} Imputations done.\n"
            f"Best Internal R2: {self.score_t.max(axis=0)[0]:0.4f}\n"
            f"Best MaiVisti R2: {self.max_score:0.4f}\n"
            f"Minimum RMSE: {math.sqrt(self.min_mse):0.4f}\n"
            f"Minimum MBE: {self.min_mbe:0.4}"
        )

    @staticmethod
    def scale(df):
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df),
                          index=df.index, columns=df.columns)
        return df

    @staticmethod
    def scale_series(s, name="Serie"):
        scaler = StandardScaler()
        s = pd.Series(
            scaler.fit_transform(s.values.reshape(-1, 1)).reshape(len(s)),
            index=s.index,
            name=name,
        )
        return s

    @staticmethod
    def scale_sets(X_train, y_train, X_test, y_test):
        """
        SCALING: Il train set non deve MAI vedere il test set,
        neanche tramite lo Scaler
        """
        scaler = StandardScaler()
        scaler.fit(X_train)
        # Si scala il train set
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # Stessa cosa per il target
        scaler.fit(y_train.values.reshape(-1, 1))
        # Si scala il train set
        y_train = scaler.transform(y_train.values.reshape(-1, 1))
        y_test = scaler.transform(y_test.values.reshape(-1, 1))

        return X_train, y_train.ravel(), X_test, y_test.ravel()


if __name__ == '__main__':

    import copy

    DATABASE = '../CSV/db_villabate_deficit_6.csv'
    SAVE = True

    ITER_LIMIT = 1000
    INVALID_LIM = 10000
    MV_FRACTION = 0.2
    FIT_FRACTION = 0.8

    COLUMNS = [
        'Rs', 'U2', 'RHmin', 'RHmax',
        'Tmin', 'Tmax', 'SWC',
        'NDVI', 'NDWI', 'DOY'
        ]

    features = et.make_dataframe(
        DATABASE,
        date_format='%Y-%m-%d',
        columns=COLUMNS,
        start='2018-01-01',
        method='impute',
        nn=5,
        drop_index=True,
        )

    eta = et.make_dataframe(
        DATABASE,
        date_format='%Y-%m-%d',
        columns=['ETa'],
        start='2018-01-01',
        method='drop',
        drop_index=True,
        )

    # Si prendono gli indici (date) di ETa
    eta_idx = copy.deepcopy(eta.index.values)
    # e si mescolano in modo random
    RNG = np.random.default_rng(seed=6475)
    RNG.shuffle(eta_idx)
    num_maivisti = int(MV_FRACTION*len(eta_idx))
    num_predict = int(FIT_FRACTION*len(eta_idx))
    idx_maivisti = eta_idx[: num_maivisti]
    idx_predict = eta_idx[-num_predict:]
    # make Timestamps
    idx_predict = [pd.Timestamp(idx) for idx in idx_predict]

    it = ImputeIterator(iter_limit=ITER_LIMIT,
                        inv_series_lim=INVALID_LIM,
                        verbose=True,
                        output_freq=20)
    it.fit(features, eta)
    # Gli indici dei dati MaiVisti vengono inseriti nell'Imputatore
    imputed = it.impute(eta,
                        idx_maivisti=idx_maivisti, idx_predict=idx_predict)

    # Indici delle misure viste (usate per l'imputazione)
    idx_fix = [idx for idx in imputed.index if idx in eta.index]
    idx_free = [idx for idx in imputed.index if idx not in eta.index]

    scores = np.array(it.score_mv)
    scores[:, 1] = np.sqrt(scores[:, 1])
    # Score for the internal test
    internal_score = np.append(np.array(it.score_t),
                               np.array(it.measured_rateo).reshape(-1, 1),
                               axis=1)
    # Sostituisce la radice del MSE al MSE
    internal_score[:, 1] = np.sqrt(internal_score[:, 1])
    # Punteggi di test
    fit_score = np.array(it.score_fit)
    # Sostituisce la radice del MSE al MSE
    fit_score[:, 1] = np.sqrt(fit_score[:, 1])

    # Si uniscono questi punteggi a quelli del test MaiVisti
    scores = np.append(scores, fit_score, axis=1)
    scores = np.append(scores, internal_score, axis=1)
    # E si crea un DataFrame
    scores = pd.DataFrame(
        scores,
        columns=['R2mv', 'RMSEmv', 'MBEmv',
                 'R2fit', 'RMSEfit',
                 'R2t', 'RMSEt', 'MBEt', 'Rateo']
        )
    if SAVE:
        scores.to_csv(f'../PAPER/PLOTS/ITERATIVEIMPUTER/SCORES_MV/SAME_FIT/'
                      f'scoresmv_{ITER_LIMIT}_model1.csv',
                      sep=';',)

    # %% PLOTS
    # Si crea una colonna di etichette per i dati di ETa
    eta_total = pd.concat([imputed.to_frame(name='ETa'),
                           eta.loc[idx_maivisti]]).sort_index()
    eta_total['source'] = [
        'Predictors' if i in idx_predict
        else 'Misurati' if i in idx_fix
        else 'Mai Visti' if i in idx_maivisti
        else 'Imputed' for i in eta_total.index]
    eta_total.index.name = 'Day'

    # Plot ETa vs Time
    sns.relplot(
        height=5, aspect=1.61,
        data=eta_total,
        x='Day', y='ETa',
        style='source', hue='source')
    plt.xticks(rotation=90)
    plt.title(f'Final Imputation: max $R^2 = {it.max_score:0.4}$')
    plt.show()
