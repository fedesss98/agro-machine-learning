# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 18:45:03 2021

@author: Federico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


reg = LinearRegression(fit_intercept=False)
linreg = LinearRegression()
# ESTRAZIONE DATI
db_d = pd.read_csv('db_deficit_ndvi_ndwi.csv', sep=';', decimal=',')

# SELEZIONE DATI
# Features complete con abbastanza misure
cols = ['RHmin', 'RHmax', 'θ 10', 'θ 20', 'θ 30', 'θ 40', 'θ 50', 'θ 60',
        'U 2m', 'Tmin', 'Tmax', 'ET0', 'NDWI']

cols2 = ['RHmin', 'RHmax', 'θ 10', 'θ 20', 'θ 30', 'θ 40', 'θ 50', 'θ 60',
        'U 2m', 'Tmin', 'Tmax', 'ET0']

x_d = db_d.loc[:, cols2]
y_d = db_d.loc[:, 'NDWI']

print("FULL: count of NULL values before  imputation\n")
print(x_d.isnull().sum())
#imputer dei dati considerati come input
imputer = KNNImputer(n_neighbors=5)
x_d = pd.DataFrame(imputer.fit_transform(x_d),columns = cols2)

print("FULL: count of NULL values after knn imputation\n")
print(x_d.isnull().sum())

x_d= pd.concat([x_d, y_d], axis=1)


NDWI_measured = x_d.loc[:, cols].dropna() # rimuove le righe con nan in qualsiasi colonna
d = x_d.loc[:, cols].dropna(subset = cols[:-1]) # rimuove righe con nan in qualsiasi colonna tranne NDWI
NDWI_index = NDWI_measured.index # lista degi indici delle righe con misure di NDWI
d_index = d.index

# SCALING
NDWI_measured_s = pd.DataFrame(prep.StandardScaler().fit_transform(NDWI_measured), columns=cols, index=NDWI_measured.index)
d_s = pd.DataFrame(prep.StandardScaler().fit_transform(x_d), columns=cols, index=d.index)

# MULTI-LAYER PERCEPTRON
# il set è diviso in due sottoinsiemi con dati random.
# uno è usato per l'addestramento, l'latro per la validazione.
mi = 1000 # max iterations
act = 'relu' # activation
alpha = 0.001
hls = (200,100,200) # hidden layer sizes
lear = 'constant' # learning rate
slvr = 'adam' # solver
mlp = MLPRegressor(activation = act, alpha=alpha, hidden_layer_sizes=hls, solver = slvr, learning_rate = lear, max_iter = mi)


# DEFICIT SCALED
NDWI_randindx = NDWI_index.tolist()
np.random.shuffle(NDWI_randindx)
train_len = int(4/5*len(NDWI_index))
a = NDWI_randindx[:train_len]
b = NDWI_randindx[train_len:]

X_train = np.array(NDWI_measured_s.loc[a, :'ET0']) # estrazione di 4/5 dei dati
y_train = np.array(NDWI_measured_s.loc[a, 'NDWI']) 

X_test = np.array(NDWI_measured_s.loc[b, :'ET0']) # rimanente 1/5 dei dati
y_test = np.array(NDWI_measured_s.loc[b, 'NDWI'])

mlp.fit(X_train, y_train)
linreg.fit(X_train, y_train)

score = mlp.score(X_test, y_test)

d_s['NDWI_completed'] = mlp.predict(d_s.loc[:,:'ET0']) # previsione di NDWI usando tutto il set di dati
mse = mean_squared_error(NDWI_measured_s['NDWI'], d_s['NDWI_completed'].loc[NDWI_index])
print("Punteggio del mlp nella predizione di NDWI: ", score)
print("Errore quadratico medio tra NDWI predetto e misurato: ", mse)

# Before fill
plt.scatter(NDWI_measured_s.index, NDWI_measured_s['NDWI'], c="red", alpha=0.3, s=40, label="NDWI measured")
plt.scatter(NDWI_measured_s.index, d_s['NDWI_completed'].loc[NDWI_index], c="blue", alpha=0.3, s=20, label="NDWI predicted")
plt.title("NDWI measuread and Predicted")
plt.legend()
plt.show()

# Filling 
d_s['NDWI_completed'] = d_s['NDWI'].fillna(d_s['NDWI_completed'])
mse = mean_squared_error(NDWI_measured_s['NDWI'], d_s['NDWI_completed'].loc[NDWI_index])

# After fill
plt.scatter(NDWI_index, NDWI_measured_s['NDWI'], c="red", alpha=0.3, s=70, label="NDWI measured")
plt.scatter(NDWI_index, d_s['NDWI_completed'].loc[NDWI_index], c="blue", alpha=0.3, s=20, label="NDWI predicted")
plt.title("NDWI measured and Predicted (after .fillna())")
plt.legend()
plt.show()

NDWI_completed = d_s['NDWI_completed'] # per confronti nei grafici

loop = 0
while(d_s['NDWI'].count() < 939):
    loop += 1
    print("Iteration", loop)
    # IMPUTATION
    n_NDWI_toPredict = 94
    X = d_s.loc[:,:'ET0'] # features fino a ET0 compresa
    y = d_s.loc[:,'NDWI_completed'] # PD Misurata completata con PD Predetta e PD Imputata
    
    randindx = d_index.tolist()
    np.random.shuffle(randindx)
    train_len = len(d_index)-n_NDWI_toPredict
    test_index = randindx[train_len:] # n_PD_toPredict indici
    train_index = randindx[:train_len] #  restanti degli indici
    
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    
    mlp.fit(X_train, y_train)
    score = mlp.score(X_test, y_test)
    print("Punteggio del mlp:", score)
    NDWI_predicted = pd.Series(mlp.predict(X_test), index=test_index, name="NDWI_predicted")
    mlp_mse = mean_squared_error (y_test, NDWI_predicted)
    
    d_s['NDWI'].fillna(NDWI_predicted, inplace=True)
    print("NDWI count:", d_s['NDWI'].count())
    d_s.loc[d_s['NDWI'].notnull(),'NDWI_completed'] = d_s['NDWI'] # rimpiazza le righe di NDWI Completed con i nuovi NDWI predetti
    
    # plt.scatter(d_s.index, NDWI_completed, c="yellow", alpha=0.2, s=70, label="NDWI completed")
    # plt.scatter(NDWI_index, NDWI_measured_s['NDWI'], c="red", alpha=0.4, s=70, label="NDWI measured")
    # plt.scatter(d_s.index, d_s['NDWI'], c="blue", alpha=0.4, s=20, label="NDWI predicted ({})".format(d_s['NDWI'].count()))
    # plt.title("NDWI Measured and Predicted ({} iterations)".format(loop))
    # plt.legend()
    # plt.show()
    
# TEST 
# Rispetto alle predizione fatte con i valori di PD misurati come si comporta l'Imputazione? 
mlp.fit(NDWI_measured_s.loc[:,:'ET0'], NDWI_measured_s.loc[:,'NDWI'])
randrange = [index for index in d_index.tolist() if index not in NDWI_index.tolist()] 
randrange = np.random.choice(randrange,100) # 100 indici random che non coincidono con quelli di PD misurato
y_imputed = d_s.loc[randrange,'NDWI']
y_predicted = mlp.predict(d_s.loc[randrange, :'ET0']) # Predetta a partire dagli NDWI Misurati
test_score = mean_squared_error(y_predicted, y_imputed)
plt.scatter(randrange, y_predicted, c="red", alpha=0.4, s=70, label="NDWI Predicted")
plt.scatter(randrange, y_imputed, c="blue", alpha=0.4, s=20, label="NDWI Imputed")
plt.title("NDWI Predicted via NDWI Measured vs NDWI Imputed in a random range")
plt.legend()
plt.show()

plt.scatter(d_index, d_s['NDWI_completed'], c="red", alpha=0.2, s=10, label="NDWI Imputed")
plt.scatter(NDWI_index, NDWI_measured['NDWI'], c="blue", alpha=0.3, s=2, label="NDWI Measured")
# plt.scatter(empty_index, PD_imputed, c="green", alpha=0.2, s=15, label="PD SciKitImputed")
plt.plot(d_index, d_s['ET0'], c="black", alpha=0.3, label="ET0")
plt.title("NDWI Imputed and ET0  scaled")
plt.legend()
plt.show()

print("MSE di alcuni valori random di NDWI Imputati contro quelli Predetti dagli NDWI Misurati:", test_score)



################################# FULL #######################################
db_f = pd.read_csv('db_full_ndvi_ndwi.csv', sep=';', decimal=',')

x_f = db_f.loc[:, cols2]
y_f = db_f.loc[:, 'NDWI']

print("Full: count of NULL values before  imputation\n")
print(x_f.isnull().sum())
#imputer dei dati considerati come input
imputer = KNNImputer(n_neighbors=5)
x_f = pd.DataFrame(imputer.fit_transform(x_f),columns = cols2)

print("Full: count of NULL values after knn imputation\n")
print(x_f.isnull().sum())

x_f= pd.concat([x_f, y_f], axis=1)

NDWI_measured_f = x_f.loc[:, cols].dropna() # rimuove le righe con nan in qualsiasi colonna
f = x_f.loc[:, cols].dropna(subset = cols[:-1]) # rimuove righe con nan in qualsiasi colonna tranne NDWI
NDWI_index_f = NDWI_measured_f.index # lista degi indici delle righe con misure di NDWI
f_index = f.index

# SCALING
NDWI_measured_s_f = pd.DataFrame(prep.StandardScaler().fit_transform(NDWI_measured_f), columns=cols, index=NDWI_measured_f.index)
f_s = pd.DataFrame(prep.StandardScaler().fit_transform(x_f), columns=cols, index=f.index)

NDWI_randindx_f = NDWI_index_f.tolist()
np.random.shuffle(NDWI_randindx_f)
train_len = int(4/5*len(NDWI_index_f))
a = NDWI_randindx_f[:train_len]
b = NDWI_randindx_f[train_len:]

X_train = np.array(NDWI_measured_s_f.loc[a, :'ET0']) # estrazione di 4/5 dei dati
y_train = np.array(NDWI_measured_s_f.loc[a, 'NDWI']) 

X_test = np.array(NDWI_measured_s_f.loc[b, :'ET0']) # rimanente 1/5 dei dati
y_test = np.array(NDWI_measured_s_f.loc[b, 'NDWI'])

mlp.fit(X_train, y_train)
linreg.fit(X_train, y_train)

score = mlp.score(X_test, y_test)

f_s['NDWI_completed'] = mlp.predict(f_s.loc[:,:'ET0']) # previsione di NDWI usando tutto il set di dati
mse = mean_squared_error(NDWI_measured_s_f['NDWI'], f_s['NDWI_completed'].loc[NDWI_index_f])
print("FULL - Punteggio del mlp nella predizione di NDWI: ", score)
print("FULL - Errore quadratico medio tra NDWI predetto e misurato: ", mse)

# Before fill
plt.scatter(NDWI_measured_s_f.index, NDWI_measured_s_f['NDWI'], c="red", alpha=0.3, s=40, label="NDWI measured")
plt.scatter(NDWI_measured_s_f.index, f_s['NDWI_completed'].loc[NDWI_index_f], c="blue", alpha=0.3, s=20, label="NDWI predicted")
plt.title("NDWI measuread and Predicted - Before fill - FULL")
plt.legend()
plt.show()

# Filling 
f_s['NDWI_completed'] = f_s['NDWI'].fillna(f_s['NDWI_completed'])
mse = mean_squared_error(NDWI_measured_s_f['NDWI'], f_s['NDWI_completed'].loc[NDWI_index_f])

# Filling 
d_s['NDWI_completed'] = d_s['NDWI'].fillna(d_s['NDWI_completed'])
mse = mean_squared_error(NDWI_measured_s['NDWI'], d_s['NDWI_completed'].loc[NDWI_index])

# After fill
plt.scatter(NDWI_index_f, NDWI_measured_s_f['NDWI'], c="red", alpha=0.3, s=70, label="NDWI measured")
plt.scatter(NDWI_index_f, f_s['NDWI_completed'].loc[NDWI_index_f], c="blue", alpha=0.3, s=20, label="NDWI predicted")
plt.title("NDWI measured and Predicted (after .fillna()) - FULL")
plt.legend()
plt.show()

NDWI_completed_f = f_s['NDWI_completed'] # per confronti nei grafici

loop = 0
while(f_s['NDWI'].count() < 939):
    loop += 1
    print("Iteration", loop)
    # IMPUTATION
    n_NDWI_toPredict = 94
    X = f_s.loc[:,:'ET0'] # features fino a ET0 compresa
    y = f_s.loc[:,'NDWI_completed'] # NDWI Misurata completata con NDWI Predetta e NDWI Imputata
    
    randindx = f_index.tolist()
    np.random.shuffle(randindx)
    train_len = len(f_index)-n_NDWI_toPredict
    test_index = randindx[train_len:] # n_NDWI_toPredict indici
    train_index = randindx[:train_len] #  restanti degli indici
    
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    
    mlp.fit(X_train, y_train)
    score = mlp.score(X_test, y_test)
    print("Punteggio del mlp:", score)
    NDWI_predicted_f = pd.Series(mlp.predict(X_test), index=test_index, name="NDWI_predicted")
    mlp_mse = mean_squared_error (y_test, NDWI_predicted_f)
    
    f_s['NDWI'].fillna(NDWI_predicted_f, inplace=True)
    print("NDWI count:", f_s['NDWI'].count())
    f_s.loc[f_s['NDWI'].notnull(),'NDWI_completed'] = f_s['NDWI'] # rimpiazza le righe di NDWI Completed con i nuovi NDWI predetti
    
    # plt.scatter(f_s.index, NDWI_completed_f, c="yellow", alpha=0.2, s=70, label="NDWI completed")
    # plt.scatter(NDWI_index_f, NDWI_measured_s_f['NDWI'], c="red", alpha=0.4, s=70, label="NDWI measured")
    # plt.scatter(d_s.index, f_s['NDWI'], c="blue", alpha=0.4, s=20, label="NDWI predicted ({})".format(d_s['NDWI'].count()))
    # plt.title("FULL _ NDWI Measured and Predicted ({} iterations)".format(loop))
    # plt.legend()
    # plt.show()

    
# TEST Full
# Rispetto alle predizione fatte con i valori di NDWI misurati come si comporta l'Imputazione? 
mlp.fit(NDWI_measured_s_f.loc[:,:'ET0'], NDWI_measured_s_f.loc[:,'NDWI'])
randrange = [index for index in f_index.tolist() if index not in NDWI_index_f.tolist()] 
randrange = np.random.choice(randrange,100) # 100 indici random che non coincidono con quelli di NDWI misurato
y_imputed = f_s.loc[randrange,'NDWI']
y_predicted = mlp.predict(f_s.loc[randrange, :'ET0']) # Predetta a partire dagli NDWI Misurati
test_score = mean_squared_error(y_predicted, y_imputed)
# plt.scatter(randrange, y_predicted, c="red", alpha=0.4, s=70, label="NDWI Predicted")
# plt.scatter(randrange, y_imputed, c="blue", alpha=0.4, s=20, label="NDWI Imputed")
# plt.title("NDWI Predicted via NDWI Measured vs NDWI Imputed in a random range")
# plt.legend()
# plt.show()

plt.scatter(d_index, f_s['NDWI_completed'], c="red", alpha=0.2, s=10, label="NDWI Imputed")
plt.scatter(NDWI_index_f, NDWI_measured_f['NDWI'], c="blue", alpha=0.3, s=2, label="NDWI Measured")
# plt.scatter(empty_index, NDWI_imputed, c="green", alpha=0.2, s=15, label="NDWI SciKitImputed")
plt.plot(d_index, f_s['ET0'], c="black", alpha=0.3, label="ET0")
plt.title("NDWI Imputed and ET0 - FULL")
plt.legend()
plt.show()

print("MSE di alcuni valori random di NDWI Imputati contro quelli Predetti dagli NDWI Misurati:", test_score)