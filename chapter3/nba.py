#-*- coding: UTF-8 -*-

import pandas as pd
# from sklearn.tree import DescisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from collections import defaultdict

data_filename = './leagues_NBA_2014_games_games.csv'
dataset = pd.read_csv(data_filename, skiprows=[0,])

dataset.columns = ["Date", "Score Type", "Visitor Team",
      "VisitorPts", "Home Team", "HomePts", "OT?", "Notes"]

dataset['HomeWin'] = dataset['VisitorPts'] < dataset['HomePts']
y_true = dataset['HomeWin'].values

won_last = defaultdict(int)

print(won_last[dataset['HomeWin'][0]])

for index, row in dataset.iterrows():
    home_team = row['Home Team']
    visitor_team = row['Visitor Team']
    row['HomeLastWin'] = won_last[home_team]
    row['VisitorLastWin'] = won_last[visitor_team]
    dataset.ix[index] = row

    # print(dataset.ix[index])

    won_last[home_team] = row["HomeWin"]
    won_last[visitor_team] = not row["HomeWin"]    


print(dataset.ix[:5])
print(won_last)

# clf = DescisionTreeClassifier(random_state=14)

# x_previouswins = dateset[['HomeLastWin', 'VisitorLastWin']].values

# scores = cross_val_score(clf, x_previouswins, y, scoring='accuracy')
# print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

