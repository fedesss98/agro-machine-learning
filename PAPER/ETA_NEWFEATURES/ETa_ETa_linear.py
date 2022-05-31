#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:27:32 2022

@author: Federico Amato
"""
import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import r2_score, mean_squared_error

from MODULES.imputiteratorClass import ImputeIterator
from MODULES import et_functions as et


ITERATIONS = 100
KFOLDS = 4
DATABASE = '../../CSV/IMPUTED/eta_imputed_{ITER_LIMIT}_{KFOLDS}k'

eta = et.make_dataframe(
    DATABASE,
    date_format='%Y-%m-%d',
    start='2018-01-01',
    method='drop',
    drop_index=True,
    )
