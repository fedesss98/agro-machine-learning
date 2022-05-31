# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:04:07 2021

@author: Antonino
"""

from sklearn.preprocessing import StandardScaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print(scaler.fit(data))

# print(scaler.mean_)
X= scaler.transform(data)
print(scaler.transform(data))


inversione = scaler.inverse_transform(X, copy=None)

print(inversione)