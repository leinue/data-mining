import os
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.decomposition import PCA

data_folder = os.path.join(os.path.expanduser("./"))
data_filename = os.path.join(data_folder, "ad.data")

def convert_number(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

converters = defaultdict(convert_number)

converters[1558] = lambda x: 1 if x.strip() == "ad." else 0
ads = pd.read_csv(data_filename, header=None, converters=converters)

print(ads)

X = ads.drop(1558, axis=1).values
y = ads[1558]

pca = PCA(n_components=5)
Xd = pca.fit_transform(X)
np.set_printoptions(precision=3, suppress=True)

print(pca.explained_variance_ratio_)
