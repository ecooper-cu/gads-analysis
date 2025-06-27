import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd

import sklearn
import scipy

import sys
import os
import pathlib
import itertools
import glob
import re
import datetime
import pickle

from itertools import groupby
from operator import itemgetter
from copy import deepcopy

from sklearn.preprocessing import QuantileTransformer, StandardScaler

package_path = os.path.join("/projects", 'emco4286', "pkgies", "mkvchain")
sys.path.append(package_path)
from model import FeatureDependentMarkovChain

pt1 = os.path.join("/projects", 'emco4286', 'data', "trajectories_with_features")

def glob_re(pattern, strings):
    return list(filter(re.compile(pattern).match, strings))

filenames = glob_re(r"gen_\d+_type_100_dtgrp_1_rating_3_state_Texas_raw.csv", os.listdir(pt1))

def consecutive_groups(iterable, ordering=lambda x: x):
    for k, g in groupby(enumerate(iterable), key=lambda x: x[0] - ordering(x[1])):
        yield map(itemgetter(1), g)

states_new = []
features_new = []
lengths_new = []
        
for f in filenames:
    # gen_id = int(re.findall(r"gen_(\d+)_.+", f)[0])

    data = pd.read_csv(os.path.join(pt1,f))
    data.set_index(pd.DatetimeIndex(data["x"]), inplace=True)
    data = data[~data.index.duplicated()]

    for c in ['y3', 'ERCOT', 'y4', 'Pcp', 'Tmax', 'Tmin', 'y6', 'y8', 'y7']:
        data = data[~data[c].isna()]

    if len(data) < 2:
        continue

    start = data.index[0]
    diffs = np.diff((data.index))
    hours = np.cumsum(diffs/np.timedelta64(1, 'h')).astype(int) - 1

    sequences = []
    for g in consecutive_groups(hours):
        sequences.append(list(g))

    for k in sequences:

        dts = pd.DatetimeIndex(start + np.array([datetime.timedelta(hours=int(h)) for h in k]))
        index = data.loc[dts[0]:dts[-1], :].index
        states = data.loc[index, "y2"].values.tolist()
        features = data.loc[index, ['y3', 'ERCOT', 'y4', 'Pcp', 'Tmax', 'Tmin', 'y6', 'y8', 'y7']].values
        l = len(k)

        states_new += [states]
        features_new += [features]

        
        lengths_new += [l]

states = np.concatenate(states_new).astype(int)
states -= 1
features = np.vstack(features_new)
lengths = np.array(lengths_new)

train_idx =  int(lengths.size * .7)
val_idx =  int(lengths.size * .8)

lengths_train = lengths[:train_idx]
lengths_val = lengths[train_idx:val_idx]
lengths_test = lengths[val_idx:]

states_train = states[:lengths_train.sum()]
states_val = states[lengths_train.sum():lengths_train.sum()+lengths_val.sum()]
states_test = states[lengths_train.sum()+lengths_val.sum():]

features_train = features[:lengths_train.sum()]
features_val = features[lengths_train.sum():lengths_train.sum()+lengths_val.sum()]
features_test = features[lengths_train.sum()+lengths_val.sum():]

ss = StandardScaler()

features_train = ss.fit_transform(features_train)
features_val = ss.transform(features_val)
features_test = ss.transform(features_test)

n = 4
model1 = FeatureDependentMarkovChain(n, lam_frob=0.1, n_iter=1, batch_size=10)
model1.fit(states_train, features_train, lengths_train, verbose=False)

print("Model #1")

predictions = model1.predict(features_val)
for i, j in itertools.product(range(4), range(4)):
    y = predictions[:, i, j]
    val = len(np.unique(y.round(decimals=3)))
    print(f"{i} -> {j} : {val}")

val2 = -np.inf
best_lam = None
model2 = None
for lam in np.logspace(-3,-1,10):
    model = FeatureDependentMarkovChain(n, lam_frob=lam,  n_iter=1)
    model.As = deepcopy(model1.As)
    model.bs = deepcopy(model1.bs)
    model.fit(states_train, features_train, lengths_train, verbose=False, warm_start=True)
    traini, vali= model.score(states_train, features_train, lengths_train, average=False), \
          model.score(states_val, features_val, lengths_val, average=False)
        #   model.score(states_test, features_test, lengths_test, average=False)
    if vali > val2:
        val2 = vali
        best_lam = lam
        model2 = model

print("\n Model #2")

predictions2 = model2.predict(features_test)
for i, j in itertools.product(range(4), range(4)):
    y = predictions2[:, i, j]
    val = len(np.unique(y.round(decimals=3)))
    print(f"{i} -> {j} : {val}")

with open("/projects/emco4286/data/texas_type_100.pkl", "wb") as f:
    pickle.dump(model2, f, protocol=4)