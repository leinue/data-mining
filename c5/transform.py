import os
import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data_folder = os.path.join(os.path.expanduser("./"))
adult_filename = os.path.join(data_folder, "adult.data") 
adult = pd.read_csv(adult_filename, header=None,
        names=["Age", "Work-Class", "fnlwgt",
        "Education", "Education-Num",
        "Marital-Status", "Occupation",
        "Relationship", "Race", "Sex",
        "Capital-gain", "Capital-loss",
        "Hours-per-week", "Native-Country",
        "Earnings-Raw"])

adult.dropna(how='all', inplace=True)

adult["LongHours"] = adult["Hours-per-week"] > 40

print(adult.columns)
print(adult["Work-Class"].unique())

X = np.arange(30).reshape((10, 3))

X[:,1] = 1

vt = VarianceThreshold()
Xt = vt.fit_transform(X)

X = adult[["Age", "Education-Num", "Capital-gain", "Capital-loss", "Hours-per-week"]].values

y = (adult["Earnings-Raw"] == ' >50K').values

print(y)

transformer = SelectKBest(score_func=chi2, k=3)
Xt_chi2 = transformer.fit_transform(X, y)
print(transformer.scores_)
