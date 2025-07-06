import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd

import sklearn
import scipy
import torch

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

torch.backends.cudnn.benchmark = True

pt1 = os.path.join("/projects", 'emco4286', 'data', "gads", "trajectories", "ct", "preloaded")

states = np.load(os.path.join(pt1, "states.npy"))
features = np.load(os.path.join(pt1, "features.npy"))
lengths = np.load(os.path.join(pt1, "lengths.npy"))

print("Loaded sequences \n")

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
model1 = FeatureDependentMarkovChain(n, lam_frob=0.1, n_iter=1, batch_size=100)
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
    model = FeatureDependentMarkovChain(n, lam_frob=lam,  n_iter=1, batch_size=100)
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

with open("/projects/emco4286/data/models/ct/texas_CT.pkl", "wb") as f:
    pickle.dump(model2, f, protocol=4)