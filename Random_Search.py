# Pandas is used for data manipulation
import pandas as pd
import numpy as np
import sklearn.preprocessing as prep

from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Read in data as a dataframe
features = pd.read_csv('CSV/db_villabate_deficit_6.csv',sep=';', decimal=',')
# Convert to numpy arrays
RANDOM_STATE = 42

# features = features.drop('Eta', axis = 1)
cols = ['Rs', 'U2', 'RHmin', 'RHmax', 'Tmin',
         'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY','ETa']
features = features.loc[:, cols].dropna().reset_index(drop=True)
# features = pd.DataFrame(prep.StandardScaler().fit(features).transform(features), columns=cols)
target = features['ETa']
features = np.array(features)
target = np.array(target)

# Training and Testing Sets
from sklearn.model_selection import train_test_split
train_features, test_features, train_target, test_target = train_test_split(features, target, 
                                                                            test_size = 0.25, random_state = RANDOM_STATE)

important_feature_names = ['Rs', 'U2', 'RHmin', 'RHmax', 'Tmin',
         'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY', 'ETo']

rf = RandomForestRegressor(random_state = RANDOM_STATE)

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2', None]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor(random_state = RANDOM_STATE)
# Random search of parameters, using 4 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='neg_mean_absolute_error', 
                              cv = 4, verbose=2, random_state=RANDOM_STATE, n_jobs=-1,
                              return_train_score=True)

# Fit the random search model
rf_random.fit(train_features, train_target);
print(rf_random.best_params_)
# print(rf_random.cv_results_)

def evaluate(model, test_features, test_target):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_target)
    mape = 100 * np.mean(errors / test_target)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f}'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
# Evaluate the Default Model
base_model = RandomForestRegressor(n_estimators = 10, random_state = RANDOM_STATE)
base_model.fit(train_features, train_target)
base_accuracy = evaluate(base_model, test_features, test_target)
# Evaluate the Best Random Search Model
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_target)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [50,60, 70, 80, 90],
    'max_features': ['auto'],
    'min_samples_leaf': [2, 3, 4],
    'min_samples_split': [2, 4, 6],
    'n_estimators': [100, 200, 300,400, 1000]
}

# Create a base model
rf = RandomForestRegressor(random_state = RANDOM_STATE)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2, return_train_score=True)
# Fit the grid search to the data
grid_search.fit(train_features, train_target);
print(grid_search.best_params_)
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test_features, test_target)


final_model = grid_search.best_estimator_
print('Final Model Parameters:\n')
pprint(final_model.get_params())
print('\n')
grid_final_accuracy = evaluate(final_model, test_features, test_target)
