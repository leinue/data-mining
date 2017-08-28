#-*- coding: UTF-8 -*-

import numpy as np
from collections import defaultdict
from operator import itemgetter

valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)

dataset_filename = "affinity_dataset.txt"
x = np.loadtxt(dataset_filename)
print(x[:5])

#面包、牛奶、奶酪、苹果和香蕉

num_apple_purcheases = 0
support = 0.0

for sample in x:
    for premise in range(4):
        if sample[premise] == 0:
            continue

        num_occurances[premise] += 1
        for conclusion in range(4):
            if premise == conclusion: continue

            if sample[conclusion] == 1:
                valid_rules[(premise, conclusion)] += 1
            else:
                invalid_rules[(premise, conclusion)] += 1

        support = valid_rules

confidence = defaultdict(float)

for premise, conclusion in valid_rules.keys():
    rule = (premise, conclusion)
    confidence[rule] = float(valid_rules[rule]) / float(num_occurances[premise])

features = {
    0: '面包',
    1: '牛奶',
    2: '奶酪',
    3: '苹果',
    4: '香蕉'    
}

def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print('RULE: if a persion buys {0} they will also buy {1}'.format(premise_name, conclusion_name))
    print(" - support: {0}".format(support[(premise, conclusion)]))
    print(" - confidence: {0:.3f}".format(confidence[(premise, conclusion)]))

premise = 1
conclusion = 2

print_rule(premise, conclusion, support, confidence, features)

sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)

sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)

for index in range(5):
    print('Rule #{0}'.format(index + 1))
    premise, conclusion = sorted_confidence[index][0]
    print_rule(premise, conclusion, support, confidence, features)
