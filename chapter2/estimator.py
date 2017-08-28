#-*- coding: UTF-8 -*-

import numpy as np 
import csv
import os

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

data_filder = './data'

data_filename = os.path.join(data_filder, '', 'ionosphere.data')

x = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype="bool")

with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)

    for i, row in enumerate(reader):
        data = [float(dataum) for dataum in row[:-1]]
        x[i] = data

        y[i] = row[-1] == 'g'

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=14)

estimator = KNeighborsClassifier()
estimator.fit(x_train, y_train)

y_predicted = estimator.predict(x_test)
accuracy = np.mean(y_test == y_predicted) * 100 #平均值
print(accuracy)

scores = cross_val_score(estimator, x, y, scoring="accuracy")
average_accuracy = np.mean(scores) * 100
print(average_accuracy)

avg_scores = []
all_scores = []

parameter_values = list(range(1, 21)) #include 20

x_broken = np.array(x)

x_broken[:,::2] /= 10

for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, x, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

# print(avg_scores)
# print(all_scores)

broken_scores = np.mean(cross_val_score(estimator, x, y, scoring='accuracy'))
print(broken_scores * 100)

x_transformed = MinMaxScaler().fit_transform(x_broken)
estimator = KNeighborsClassifier()
transformed_scores = cross_val_score(estimator, x_transformed, y, scoring='accuracy')
print(np.mean(transformed_scores) * 100)


scaling_pipeline = Pipeline(
    [
        ('scale', MinMaxScaler()),
        ('predict', KNeighborsClassifier())
    ]
)

scores = cross_val_score(scaling_pipeline, x_broken, y, scoring='accuracy')

print(np.mean(scores) * 100)


