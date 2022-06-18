"""
========================================================
Post pruning decision trees with cost complexity pruning
========================================================

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from tqdm import tqdm

import MODULES.et_functions as et


def plot_impurity(x, y, m):
    fig, ax = plt.subplots()
    ax.plot(x[:-1], y[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title(f"Total Impurity vs effective alpha for training set ({m})")
    plt.show()


def plot_nodes(x, node_counts, depth, m):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(x, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(x, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title(f"Depth vs alpha ({m})")
    fig.tight_layout()


def plot_scores(x, train_scores, test_scores, m):
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Accuracy vs alpha for training and testing sets ({m})")
    ax.plot(x, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(x, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()


ROOT = '../../'
DATABASE = '../../CSV/db_villabate_deficit_6.csv'

RANDOM_STATE = 12

REGRESSOR = 'rf'  # rf / dt

MODELS_FEATURES = [
        ['Rs', 'U2', 'RHmin', 'RHmax', 'Tmin',
         'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 1
        ['Rs', 'U2', 'RHmax', 'Tmin',
         'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 2
        ['Rs', 'U2', 'RHmax', 'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 3
        ['Rs', 'U2', 'RHmax', 'Tmax', 'SWC', 'NDWI', 'DOY'],  # 4
        ['Rs', 'U2', 'Tmax', 'SWC', 'NDWI', 'DOY'],   # 5
        ['Rs', 'U2', 'Tmax', 'SWC', 'DOY'],  # 6
        ['Rs', 'Tmax', 'SWC', 'DOY'],  # 7
        ['Rs', 'RHmin', 'RHmax', 'Tmin', 'Tmax'],  # 8
        ['ETo', 'SWC', 'NDVI', 'NDWI', 'DOY'],  # 9
        ['ETo', 'NDVI', 'NDWI', 'DOY'],  # 10
        ['Rs', 'Tmin', 'Tmax', 'DOY'],  # 11
        ['Rs', 'Tavg', 'RHavg', 'DOY'],  # 12
    ]

y = et.make_dataframe(
    DATABASE,
    columns='ETa',
    start='2018-01-01',
    method='drop',
    drop_index=True,
    )

best_alpha = np.zeros((len(MODELS_FEATURES)))

# %%
# returns the
# effective alphas and the corresponding total leaf impurities at each step of
# the pruning process. As alpha increases, more of the tree is pruned, which
# increases the total impurity of its leaves.

for m, fts in enumerate(MODELS_FEATURES, 1):
    X = et.make_dataframe(
        DATABASE,
        date_format='%Y-%m-%d',
        columns=fts,
        start='2018-01-01',
        method='impute',
        nn=5,
        drop_index=True,
        )

    X = X.loc[y.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.25)

    if REGRESSOR == 'dt':
        # questo non si può fare con la RandomForest
        clf = DecisionTreeRegressor(random_state=RANDOM_STATE)
        path = clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        # Plot Impurity vs Alpha
        plot_impurity(ccp_alphas, impurities, m)
    else:
        # Bisogna dare gli alpha a mano
        ccp_alphas = np.append(np.linspace(0, 0.025, 60),
                               np.linspace(0.025, 0.06, 10))

    # %%
    # Next, we train a regressor tree using the effective alphas.
    # The last value in ``ccp_alphas`` is the alpha value that prunes
    # the whole tree, leaving the tree, ``clfs[-1]``, with one node.
    clfs = []
    for ccp_alpha in tqdm(ccp_alphas):
        if REGRESSOR == 'dt':
            clf = DecisionTreeRegressor(random_state=RANDOM_STATE,
                                        ccp_alpha=ccp_alpha)
        elif REGRESSOR == 'rf':
            clf = RandomForestRegressor(random_state=RANDOM_STATE,
                                        ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    # %%
    # For the remainder of this example, we remove the last element in
    # ``clfs`` and ``ccp_alphas``, because it is the trivial tree with only
    # one node. Here we show that the number of nodes and tree depth decreases
    # as alpha increases.
    if REGRESSOR == 'dt':
        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]

        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]
        plot_nodes(ccp_alphas, node_counts, depth, m)
        print(
            f"Number of nodes in the last tree is: {clfs[-1].tree_.node_count}"
            f"with ccp_alpha: {ccp_alphas[-1]}"
            )
    # %%
    # Accuracy vs alpha for training and testing sets
    # ----------------------------------------------------
    # When ``ccp_alpha`` is set to zero and keeping the other default
    # parameters of :class:`DecisionTree`, the tree overfits, leading to a 100%
    # training accuracy and 88% testing accuracy. As alpha increases, more
    # of the tree is pruned, thus creating a decision tree that generalizes
    # better. In this example, setting ``ccp_alpha=0.015``
    # maximizes the testing accuracy.
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    best_alpha[m-1] = ccp_alphas[test_scores.index(max(test_scores))]
    plot_scores(ccp_alphas, train_scores, test_scores, m)

print('Best alpha parameters:')
for model, alpha in enumerate(best_alpha, 1):
    print(f'Model: {model} - {alpha}')
