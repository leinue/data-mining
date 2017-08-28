#-*- coding: UTF-8 -*-

import numpy as np
from sklearn.datasets import load_iris
from collections import defaultdict
from operator import itemgetter

from sklearn.cross_validation import train_test_split

dataset = load_iris()

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
plants = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

X = dataset.data
Y = dataset.target

attribute_means = X.mean(axis=0)

x_d = np.array(X >= attribute_means, dtype="int")

Xd_train, Xd_test, y_train, y_test = train_test_split(x_d, Y, random_state=14)

print(Xd_train, Xd_test, y_train, y_test)

def train_feature_value(X, y_true, feature_index, value):
    class_counts = defaultdict(int)
    for sample, y in zip(X, y_true):
        if sample[feature_index] == value:
            class_counts[y] += 1

    sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]

    incorrect_predictions = [class_count for class_value, class_count in class_counts.items() if class_value != most_frequent_class]
    error = sum(incorrect_predictions)

    return most_frequent_class, error

def train_on_feature(X, y_true, feature_index):
    values = set(X[:, feature_index])
    predictors = {}
    errors = []

    for current_value in values:
        most_frequent_class, error = train_feature_value(X, y_true, feature_index, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)

    total_error = sum(errors)
    return predictors, total_error



