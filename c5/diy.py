from sklearn.base import TransformerMixin
from sklearn.utils import as_float_array
from numpy.testing import assert_array_equal
import numpy as np
from sklearn.pipeline import Pipeline

class MeanDiscrete(TransformerMixin):

    def fit(self, X):
        X = as_float_array(X)
        self.mean = X.mean(axis=0)
        return self

    def transform(self, X):
        X = as_float_array(X)
        assert X.shape[1] == self.mean.shape[0]
        return X > self.mean

def test_meandiscrete():
    X_test = np.array([[ 0,  2], [3, 5], [6, 8], [ 9, 11], [12, 14], [15, 17], [18, 20], [21, 23], [24, 26], [27, 29]])
    mean_discrete = MeanDiscrete()
    mean_discrete.fit(X_test)
    assert_array_equal(mean_discrete.mean, np.array([13.5, 15.5]))

    X_transformed = mean_discrete.transform(X_test)
    X_expected = np.array([[ 0,  0],
                            [ 0, 0],
                            [ 0, 0],
                            [ 0, 0],
                            [ 0, 0],
                            [ 1, 1],
                            [ 1, 1],
                            [ 1, 1],
                            [ 1, 1],
                            [ 1, 1]])
    assert_array_equal(X_transformed, X_expected)

test_meandiscrete()

pipeline = Pipeline([('mean_discrete', MeanDiscrete()), ('classifier', DecisionTreeClassifier(random_state=14))]) 
scores_mean_discrete = cross_val_score(pipeline, X, y, scoring='accuracy')
print("Mean Discrete performance:{0:.3f}".format(scores_mean_discrete.mean()))
