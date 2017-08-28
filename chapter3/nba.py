#-*- coding: UTF-8 -*-

import pandas as pd
from sklearn.tree import DescisionTreeClassifier

data_filename = './leagues_NBA_2014_games_games.csv'
dataset = pd.read_csv(data_filename, skiprows=[0,], parse_dates=['Dates'])

dataset.columns = ["Date", "Score Type", "Visitor Team",
      "VisitorPts", "Home Team", "HomePts", "OT?", "Notes"]

print(dataset.ix[:5])

clf = DescisionTreeClassifier(random_state=14)


