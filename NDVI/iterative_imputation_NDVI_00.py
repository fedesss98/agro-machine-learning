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
        'U 2m', 'Rshw dwn', 'Tmin', 'Tmax', 'ET0', 'NDVI']

cols2 = ['RHmin', 'RHmax', 'θ 10', 'θ 20', 'θ 30', 'θ 40', 'θ 50', 'θ 60',
        'U 2m', 'Rshw dwn', 'Tmin', 'Tmax', 'ET0']

x_d = db_d.loc[:, cols2]
y_d = db_d.loc[:, 'NDVI']

print("count of NULL values before  imputation\n")
print(x_d.isnull().sum())
#imputer dei dati considerati come input
imputer = KNNImputer(n_neighbors=5)
x_d = pd.DataFrame(imputer.fit_transform(x_d),columns = cols2)

print("count of NULL values after knn imputation\n")
print(x_d.isnull().sum())

x_d= pd.concat([x_d, y_d], axis=1)

NDVI_measured = x_d.loc[:, cols].dropna() # rimuove le righe con nan in qualsiasi colonna
d = x_d.loc[:, cols].dropna(subset = cols[:-1]) # rimuove righe con nan in qualsiasi colonna tranne NDVI
NDVI_index = NDVI_measured.index # lista degi indici delle righe con misure di NDVI
d_index = d.index

# SCALING
NDVI_measured_s = pd.DataFrame(prep.StandardScaler().fit_transform(NDVI_measured), columns=cols, index=NDVI_measured.index)
d_s = pd.DataFrame(prep.StandardScaler().fit_transform(x_d), columns=cols, index=d.index)

# MULTI-LAYER PERCEPTRON
# il set è diviso in due sottoinsiemi con dati random.
# uno è usato per l'addestramento, l'latro per la validazione.
mi = 1000 # max iterations
act = 'relu' # activation
alpha = 0.001
hls = (300,100,200) # hidden layer sizes
lear = 'constant' # learning rate
slvr = 'adam' # solver
mlp = MLPRegressor(activation = act, alpha=alpha, hidden_layer_sizes=hls, solver = slvr, learning_rate = lear, max_iter = mi)


######################################## DEFICIT #######################
NDVI_randindx = NDVI_index.tolist()
np.random.shuffle(NDVI_randindx)
train_len = int(4/5*len(NDVI_index))
a = NDVI_randindx[:train_len]
b = NDVI_randindx[train_len:]

X_train = np.array(NDVI_measured_s.loc[a, :'ET0']) # estrazione di 4/5 dei dati
y_train = np.array(NDVI_measured_s.loc[a, 'NDVI']) 

X_test = np.array(NDVI_measured_s.loc[b, :'ET0']) # rimanente 1/5 dei dati
y_test = np.array(NDVI_measured_s.loc[b, 'NDVI'])

mlp.fit(X_train, y_train)
linreg.fit(X_train, y_train)

score = mlp.score(X_test, y_test)

d_s['NDVI_completed'] = mlp.predict(d_s.loc[:,:'ET0']) # previsione di NDVI usando tutto il set di dati
mse = mean_squared_error(NDVI_measured_s['NDVI'], d_s['NDVI_completed'].loc[NDVI_index])
print("Punteggio del mlp nella predizione di NDVI: ", score)
print("Errore quadratico medio tra NDVI predetto e misurato: ", mse)

# Before fill
plt.scatter(NDVI_measured_s.index, NDVI_measured_s['NDVI'], c="red", alpha=0.3, s=40, label="NDVI measured")
plt.scatter(NDVI_measured_s.index, d_s['NDVI_completed'].loc[NDVI_index], c="blue", alpha=0.3, s=20, label="NDVI predicted")
plt.title("NDVI measuread and Predicted - Before fill - Deficit")
plt.legend()
plt.show()

# Filling 
d_s['NDVI_completed'] = d_s['NDVI'].fillna(d_s['NDVI_completed'])
mse = mean_squared_error(NDVI_measured_s['NDVI'], d_s['NDVI_completed'].loc[NDVI_index])

# After fill
plt.scatter(NDVI_index, NDVI_measured_s['NDVI'], c="red", alpha=0.3, s=70, label="NDVI measured")
plt.scatter(NDVI_index, d_s['NDVI_completed'].loc[NDVI_index], c="blue", alpha=0.3, s=20, label="NDVI predicted")
plt.title("NDVI measured and Predicted (after .fillna()) - Deficit")
plt.legend()
plt.show()

NDVI_completed = d_s['NDVI_completed'] # per confronti nei grafici

loop = 0
while(d_s['NDVI'].count() < 939):
    loop += 1
    print("Iteration", loop)
    # IMPUTATION
    n_NDVI_toPredict = 94
    X = d_s.loc[:,:'ET0'] # features fino a ET0 compresa
    y = d_s.loc[:,'NDVI_completed'] # NDVI Misurata completata con NDVI Predetta e NDVI Imputata
    
    randindx = d_index.tolist()
    np.random.shuffle(randindx)
    train_len = len(d_index)-n_NDVI_toPredict
    test_index = randindx[train_len:] # n_NDVI_toPredict indici
    train_index = randindx[:train_len] #  restanti degli indici
    
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    
    mlp.fit(X_train, y_train)
    score = mlp.score(X_test, y_test)
    print("Punteggio del mlp:", score)
    NDVI_predicted = pd.Series(mlp.predict(X_test), index=test_index, name="NDVI_predicted")
    mlp_mse = mean_squared_error (y_test, NDVI_predicted)
    
    d_s['NDVI'].fillna(NDVI_predicted, inplace=True)
    print("NDVI count:", d_s['NDVI'].count())
    d_s.loc[d_s['NDVI'].notnull(),'NDVI_completed'] = d_s['NDVI'] # rimpiazza le righe di NDVI Completed con i nuovi NDVI predetti
    
    # plt.scatter(d_s.index, NDVI_completed, c="yellow", alpha=0.2, s=70, label="NDVI completed")
    # plt.scatter(NDVI_index, NDVI_measured_s['NDVI'], c="red", alpha=0.4, s=70, label="NDVI measured")
    # plt.scatter(d_s.index, d_s['NDVI'], c="blue", alpha=0.4, s=20, label="NDVI predicted ({})".format(d_s['NDVI'].count()))
    # plt.title("NDVI Measured and Predicted ({} iterations)".format(loop))
    # plt.legend()
    # plt.show()
    
# TEST 
# Rispetto alle predizione fatte con i valori di NDVI misurati come si comporta l'Imputazione? 
mlp.fit(NDVI_measured_s.loc[:,:'ET0'], NDVI_measured_s.loc[:,'NDVI'])
randrange = [index for index in d_index.tolist() if index not in NDVI_index.tolist()] 
randrange = np.random.choice(randrange,100) # 100 indici random che non coincidono con quelli di NDVI misurato
y_imputed = d_s.loc[randrange,'NDVI']
y_predicted = mlp.predict(d_s.loc[randrange, :'ET0']) # Predetta a partire dagli NDVI Misurati
test_score = mean_squared_error(y_predicted, y_imputed)
# plt.scatter(randrange, y_predicted, c="red", alpha=0.4, s=70, label="NDVI Predicted")
# plt.scatter(randrange, y_imputed, c="blue", alpha=0.4, s=20, label="NDVI Imputed")
# plt.title("NDVI Predicted via NDVI Measured vs NDVI Imputed in a random range")
# plt.legend()
# plt.show()

plt.scatter(d_index, d_s['NDVI_completed'], c="red", alpha=0.2, s=10, label="NDVI Imputed")
plt.scatter(NDVI_index, NDVI_measured['NDVI'], c="blue", alpha=0.3, s=2, label="NDVI Measured")
# plt.scatter(empty_index, NDVI_imputed, c="green", alpha=0.2, s=15, label="NDVI SciKitImputed")
plt.plot(d_index, d_s['ET0'], c="black", alpha=0.3, label="ET0")
plt.title("NDVI Imputed and ET0 Deficit")
plt.legend()
plt.show()

print("MSE di alcuni valori random di NDVI Imputati contro quelli Predetti dagli NDVI Misurati:", test_score)

################################ FULL ######################################

db_f = pd.read_csv('db_full_ndvi_ndwi.csv', sep=';', decimal=',')

x_f = db_f.loc[:, cols2]
y_f = db_f.loc[:, 'NDVI']

print("Full: count of NULL values before  imputation\n")
print(x_f.isnull().sum())
#imputer dei dati considerati come input
imputer = KNNImputer(n_neighbors=5)
x_f = pd.DataFrame(imputer.fit_transform(x_f),columns = cols2)

print("Full: count of NULL values after knn imputation\n")
print(x_f.isnull().sum())

x_f= pd.concat([x_f, y_f], axis=1)

NDVI_measured_f = x_f.loc[:, cols].dropna() # rimuove le righe con nan in qualsiasi colonna
f = x_f.loc[:, cols].dropna(subset = cols[:-1]) # rimuove righe con nan in qualsiasi colonna tranne NDVI
NDVI_index_f = NDVI_measured_f.index # lista degi indici delle righe con misure di NDVI
f_index = f.index

# SCALING
NDVI_measured_s_f = pd.DataFrame(prep.StandardScaler().fit_transform(NDVI_measured_f), columns=cols, index=NDVI_measured_f.index)
f_s = pd.DataFrame(prep.StandardScaler().fit_transform(x_f), columns=cols, index=f.index)

NDVI_randindx_f = NDVI_index_f.tolist()
np.random.shuffle(NDVI_randindx_f)
train_len = int(4/5*len(NDVI_index_f))
a = NDVI_randindx_f[:train_len]
b = NDVI_randindx_f[train_len:]

X_train = np.array(NDVI_measured_s_f.loc[a, :'ET0']) # estrazione di 4/5 dei dati
y_train = np.array(NDVI_measured_s_f.loc[a, 'NDVI']) 

X_test = np.array(NDVI_measured_s_f.loc[b, :'ET0']) # rimanente 1/5 dei dati
y_test = np.array(NDVI_measured_s_f.loc[b, 'NDVI'])

mlp.fit(X_train, y_train)
linreg.fit(X_train, y_train)

score = mlp.score(X_test, y_test)

f_s['NDVI_completed'] = mlp.predict(f_s.loc[:,:'ET0']) # previsione di NDVI usando tutto il set di dati
mse = mean_squared_error(NDVI_measured_s_f['NDVI'], f_s['NDVI_completed'].loc[NDVI_index_f])
print("FULL - Punteggio del mlp nella predizione di NDVI: ", score)
print("FULL - Errore quadratico medio tra NDVI predetto e misurato: ", mse)

# Before fill
plt.scatter(NDVI_measured_s_f.index, NDVI_measured_s_f['NDVI'], c="red", alpha=0.3, s=40, label="NDVI measured")
plt.scatter(NDVI_measured_s_f.index, f_s['NDVI_completed'].loc[NDVI_index_f], c="blue", alpha=0.3, s=20, label="NDVI predicted")
plt.title("NDVI measuread and Predicted - Before fill - FULL")
plt.legend()
plt.show()

# Filling 
f_s['NDVI_completed'] = f_s['NDVI'].fillna(f_s['NDVI_completed'])
mse = mean_squared_error(NDVI_measured_s_f['NDVI'], f_s['NDVI_completed'].loc[NDVI_index_f])

# Filling 
d_s['NDVI_completed'] = d_s['NDVI'].fillna(d_s['NDVI_completed'])
mse = mean_squared_error(NDVI_measured_s['NDVI'], d_s['NDVI_completed'].loc[NDVI_index])

# After fill
plt.scatter(NDVI_index_f, NDVI_measured_s_f['NDVI'], c="red", alpha=0.3, s=70, label="NDVI measured")
plt.scatter(NDVI_index_f, f_s['NDVI_completed'].loc[NDVI_index_f], c="blue", alpha=0.3, s=20, label="NDVI predicted")
plt.title("NDVI measured and Predicted (after .fillna()) - FULL")
plt.legend()
plt.show()

NDVI_completed_f = f_s['NDVI_completed'] # per confronti nei grafici

loop = 0
while(f_s['NDVI'].count() < 939):
    loop += 1
    print("Iteration", loop)
    # IMPUTATION
    n_NDVI_toPredict = 94
    X = f_s.loc[:,:'ET0'] # features fino a ET0 compresa
    y = f_s.loc[:,'NDVI_completed'] # NDVI Misurata completata con NDVI Predetta e NDVI Imputata
    
    randindx = f_index.tolist()
    np.random.shuffle(randindx)
    train_len = len(f_index)-n_NDVI_toPredict
    test_index = randindx[train_len:] # n_NDVI_toPredict indici
    train_index = randindx[:train_len] #  restanti degli indici
    
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    
    mlp.fit(X_train, y_train)
    score = mlp.score(X_test, y_test)
    print("Punteggio del mlp:", score)
    NDVI_predicted_f = pd.Series(mlp.predict(X_test), index=test_index, name="NDVI_predicted")
    mlp_mse = mean_squared_error (y_test, NDVI_predicted_f)
    
    f_s['NDVI'].fillna(NDVI_predicted_f, inplace=True)
    print("NDVI count:", f_s['NDVI'].count())
    f_s.loc[f_s['NDVI'].notnull(),'NDVI_completed'] = f_s['NDVI'] # rimpiazza le righe di NDVI Completed con i nuovi NDVI predetti
    
    # plt.scatter(f_s.index, NDVI_completed_f, c="yellow", alpha=0.2, s=70, label="NDVI completed")
    # plt.scatter(NDVI_index_f, NDVI_measured_s_f['NDVI'], c="red", alpha=0.4, s=70, label="NDVI measured")
    # plt.scatter(d_s.index, f_s['NDVI'], c="blue", alpha=0.4, s=20, label="NDVI predicted ({})".format(d_s['NDVI'].count()))
    # plt.title("FULL _ NDVI Measured and Predicted ({} iterations)".format(loop))
    # plt.legend()
    # plt.show()

# NDVI_fit = NDVI_measured_f['NDVI']
  
# TEST Full
# Rispetto alle predizione fatte con i valori di NDVI misurati come si comporta l'Imputazione? 
mlp.fit(NDVI_measured_s_f.loc[:,:'ET0'], NDVI_measured_s_f.loc[:,'NDVI'])
randrange = [index for index in f_index.tolist() if index not in NDVI_index_f.tolist()] 
randrange = np.random.choice(randrange,100) # 100 indici random che non coincidono con quelli di NDVI misurato
y_imputed = f_s.loc[randrange,'NDVI']
y_predicted = mlp.predict(f_s.loc[randrange, :'ET0']) # Predetta a partire dagli NDVI Misurati
test_score = mean_squared_error(y_predicted, y_imputed)
# plt.scatter(randrange, y_predicted, c="red", alpha=0.4, s=70, label="NDVI Predicted")
# plt.scatter(randrange, y_imputed, c="blue", alpha=0.4, s=20, label="NDVI Imputed")
# plt.title("NDVI Predicted via NDVI Measured vs NDVI Imputed in a random range")
# plt.legend()
# plt.show()



scaler = StandardScaler().fit(f)  
f_s2 = f_s.loc[:, cols]
# NDVI_completed_inverse_f = f_s['NDVI_completed']


inversione = pd.DataFrame(scaler.inverse_transform(f_s2),columns=cols, index=f.index)

plt.scatter(db_f['Day'], inversione['NDVI'], c="red", alpha=0.2, s=10, label="NDVI Imputed")
plt.scatter(NDVI_index_f, NDVI_measured_f['NDVI'], c="blue", alpha=0.3, s=2, label="NDVI Measured")
# plt.scatter(empty_index, NDVI_imputed, c="green", alpha=0.2, s=15, label="NDVI SciKitImputed")
# plt.plot(d_index, f_s['ET0'], c="black", alpha=0.3, label="ET0")
plt.title("NDVI Imputed and ET0 - FULL")
plt.legend()
plt.show()

print("MSE di alcuni valori random di NDVI Imputati contro quelli Predetti dagli NDVI Misurati:", test_score)