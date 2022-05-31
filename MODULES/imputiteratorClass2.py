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
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class ImputeIterator:

    # Random Generator for reproducible results
    RNG = np.random.default_rng(seed=6475)
    MLP_RAND_STATE = 4536
    RF_RAND_STATE = 256

    # GRID SEARCH
    grid_params = {
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["adam", "sgd", "adam"],
        "alpha": [0.0001, 0.001, 0.1],
    }
    # Grid-Search Regressor
    rgr = GridSearchCV(
        MLPRegressor(
            random_state=432,
            hidden_layer_sizes=(100, 100, 100),
            max_iter=10000
        ),
        grid_params,
    )

    mlp_params = {
        "hidden_layer_sizes": (365, 120, 365),
        "activation": "relu",  # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
        "solver": "sgd",  # ‘lbfgs’, ‘sgd’, ‘adam’
        "max_iter": 10000,
        "alpha": 0.1,
        "learning_rate": "adaptive",
        "learning_rate_init": 0.0002,
        # 'warm_start': True,
        "random_state": MLP_RAND_STATE,
    }

    first_model = RandomForestRegressor(n_estimators=365, random_state=4)
    model = MLPRegressor(**mlp_params)
    model_final = model
    model_test = RandomForestRegressor(random_state=RF_RAND_STATE,)
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    # Percentuale di dati usati per l'Addestramento dell'Imputazione
    train_fraction = 0.8

    # Array dei punteggi ottenuti durante le Iterazioni
    score_t = []  # Punteggio del Test (step 1)
    score_v = []  # Punteggio della Valutazione (step 2)
    score_mv = []  # Punteggio su un set di Target mai visto (step 4)
    measured_rateo = []  # Percentuale dei punti misurati nei Test
    # Parametri di uscita

    def __init__(
        self,
        model=model,
        train_fraction=train_fraction,
        inv_series_lim=10,
        iter_limit=15,
        output_freq=10,
        verbose=False,
    ):
        self.model = model
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
        self.X = None
        self.y = None
        self.target_name = None
        self.target = None
        self.fts_visti = None
        self.target_visti = None
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

    def fit(self, features, target, model=first_model):
        self.fitted = True
        self.X = features
        self.target = target
        test_size = 0.25
        X_train, X_test, y_train, y_test = train_test_split(
            features.loc[target.index], target,
            test_size=0.25, random_state=4,
        )

        # SCALING: dopo questo si avranno numpy array
        X_train, y_train, X_test, y_test = self.scale_sets(
            X_train, y_train, X_test, y_test)
        # Addestramento
        print(f"Training Model on {1-test_size:%} of Measured data")
        model.fit(X_train, y_train)
        self.first_model = model

    def impute(self, target, maivisti_size=0.25):
        if self.fitted:
            # Si trasforma la Serie in DataFrame
            if isinstance(target, pd.Series):
                target = target.to_frame()

            self.target_name = target.columns[0]
            self.maivisti_size = int(len(self.target) * maivisti_size)
            self.max_score = 0
            self.iterative_imputation()
            self.__print_ending()
        else:
            raise Exception("You need to fit the Imputer first!")

        return self.y

    def fit_impute(self, X, target, maivisti_size=0.25):
        self.fitted = True
        self.X = X
        self.target = target
        self.target_name = target.columns[0]
        self.maivisti_size = int(len(self.target) * maivisti_size)
        self.max_score = 0

        self.iterative_imputation()
        self.__print_ending()
        print(
            f'{self.n_imputations} Imputations done.\n'
            f'Best Validation Score: {self.max_score}'
        )
        return self.y

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

        idx_train = self.RNG.choice(
            self.fts_visti.index,
            size=int(len(self.fts_visti.index) * self.train_fraction),
            replace=False,
        )
        idx_predict = [idx for idx in self.fts_visti.index
                       if idx not in idx_train]
        # Quanti punti effettivamente misurati stanno nel set di Test?
        predicted_measures = [idx for idx in idx_predict
                              if idx not in idx_free]
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
        temp = self.y
        # completato da questi dati appena imputati
        temp.loc[y_new.index] = y_new

        return temp, test_score, measured_rateo

    def internal_test(self, train, test, subset=None):
        X_train, y_train = train
        X_test, y_test = test
        self.model.fit(X_train, y_train)
        y_predict = self.model.predict(X_test)
        y_predict = pd.Series(y_predict,
                              index=X_test.index, name="Target Imputed")
        if subset is not None:
            y_test_measures = y_test.loc[subset]
            y_predict_measures = y_predict.loc[subset]
            test_score = r2_score(y_test_measures, y_predict_measures)
        else:
            test_score = r2_score(y_test, y_predict)
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

    def test_on_maivisti(self, model, X_train, y_train):
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
        model.fit(X_train, y_train)
        # Predice i dati MaiVisti
        X_predict = self.fts_maivisti
        y_predicted = model.predict(X_predict)
        # Calcola i punteggi
        score = r2_score(self.target_maivisti, y_predicted)
        mse = mean_squared_error(self.target_maivisti, y_predicted)
        mbe = et.mean_bias_error(self.target_maivisti, y_predicted)
        return score, mse, mbe

    # %% MAIN PROGRAM

    def iterative_imputation(self):
        fts_names = [col for col in self.X.columns if col != self.target_name]
        # Si blocca un certo numero di misure di Target
        # da non usare nell'Imputazione,
        # per usarle poi come controllo nel Test Maivisti
        idx_maivisti = self.RNG.choice(
            self.target.index, size=self.maivisti_size, replace=False
        )
        target_maivisti = self.target.loc[idx_maivisti]
        # Si prendono le date di queste misure bloccate
        # E si tirano via di conseguenza le features legate a questi Target
        fts_maivisti = self.X.loc[idx_maivisti, fts_names]
        # Il resto del Target potrà essere usato (visto) per l'Imputazione
        idx_visti = [idx for idx in self.target.index
                     if idx not in idx_maivisti]
        target_visti = self.target.loc[idx_visti]

        # Si prendono quindi tutte le rimanenti date
        idx_visti = [idx for idx in self.X.index if idx not in idx_maivisti]
        # E si prendono le features legate a queste
        fts_visti = self.X.loc[idx_visti, fts_names]
        # E anche i Target (Misurati e Imputati in creazione)
        y = self.X.loc[idx_visti, self.target_name]

        # Si fa uno Scaling per avere previsioni migliori
        self.fts_visti = self.scale(fts_visti)
        self.fts_maivisti = self.scale(fts_maivisti)
        self.target_maivisti = self.scale(target_maivisti)
        self.target_visti = self.scale(target_visti)
        self.y = self.scale_series(y, "Target Imputed")

        # Date con misure del Target: non dovranno essere toccate
        idx_fix = self.target_visti.index
        # Le restanti date sono libere di essere sovrascritte nell'Imputazione
        idx_free = [idx for idx in self.fts_visti.index if idx not in idx_fix]

        # STEP 0 - PRIMO SCORE
        score0, mse0, mbe0 = self.test_on_maivisti(self.model,
                                                   self.fts_visti, self.y)
        self.score_mv.append([score0, mse0, mbe0])
        print(self.score_mv[-1])
        self.max_score = score0

        # Stampa Output iniziale
        self.__print_header()

        # CICLO DI IMPUTAZIONE
        # Variabili di controllo del ciclo
        i = 0  # Cicli eseguiti

        while self.n_inv_series < self.inv_series_lim and i < self.iter_limit:
            # STEP 1 - IMPUTAZIONE
            temp, test_score, rateo = self.__impute_anew(idx_free)
            self.score_t.append(test_score)
            self.measured_rateo.append(rateo)
            # STEP 2 - VALIDAZIONE
            valid_iter = self.__check_valid_iteration(temp)

            # STEP 3 - INSERIMENTO
            self.__insert_or_not(temp, valid_iter)

            # Output finale con i tre punteggi
            self.__print_output(i, valid_iter)
            # Continua con le iterazioni
            i = i + 1
        mse_scores = np.array(self.score_mv)[:, 1]
        mbe_scores = np.array(self.score_mv)[:, 2]
        self.min_mse = min(mse_scores, key=abs)
        self.min_mbe = min(mbe_scores, key=abs)
        y = self.y.values.reshape(-1, 1)
        y = StandardScaler().fit(self.target).inverse_transform(y).ravel()
        self.y = pd.Series(y, name="Target Imputed", index=idx_visti)

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
                f'{i:<}\t{self.score_t[-1]:>13.4}'
                f'({self.measured_rateo[-1]:.0%})'
                f'\t{self.score_mv[-1][0]:>11.4}'
                f'\t\t{math.sqrt(self.score_mv[-1][1]):>11.4}'
                f'\t\t{self.score_mv[-1][2]:>11.4}'
                f'\t{inserted:>8}'
            )
        elif not i % self.output_freq:
            print(f"Iteration {i}: {inserted}.")

    def __print_ending(self):
        print(
            f"{self.n_imputations} Imputations done.\n"
            f"Best MaiVisti R2: {self.max_score:0.4}\n"
            f"Minimum RMSE: {math.sqrt(self.min_mse):0.4}\n"
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

    # Si estrae il DataFrame completo delle Features
    features = et.make_dataframe(
        '../CSV/db_villabate_deficit.csv',
        columns=['θ 10', 'θ 20', 'θ 30', 'θ 40', 'θ 50', 'θ 60', 'U 2m',
                 'Rshw dwn', 'RHmin', 'RHmax', 'Tmin', 'Tmax', 'ET0'],
        start='2018-01-01',
        drop_index=True
        )

    eta = et.make_dataframe('../CSV/db_villabate_deficit.csv',
                            columns=['ETa'],
                            start='2018-01-01',
                            method='drop',
                            drop_index=True
                            )

    # Iterazioni massime per l'Imputazione Iterativa:
    iter_limit = 5
    inv_series_lim = 50

    # Si uniscono Features e Target
    X = features.join(eta)
    # Si fa una prima imputazione di X (righe di Features o Target mancanti)
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    X = pd.DataFrame(imputer.fit_transform(X),
                     columns=X.columns, index=X.index)

    # Oggetto ImputeIterator per l'Imputazione Iterativa
    it = ImputeIterator(iter_limit=iter_limit,
                        inv_series_lim=inv_series_lim,
                        verbose=True)

    # Si fitta l'Imputatore sul set di Features e Target già imputato
    it.fit(X)
    # Si imputa il Target a partire da quello misurato
    print('*** ETa Imputation ***\n'
          f'Features used:\n{list(X.columns)}')
    eta_imputed = it.impute(eta)
    eta_imputed.name = 'ETa Imputed'

    it.plot_imputation()
