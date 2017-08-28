from sklearn.datasets import load_iris
dataset = load_iris()

features = ['sepal length', 'sepal width', 'petal length', 'petal width']

X = dataset.data
Y = dataset.target

print(dataset.DESCR)
print(Y)