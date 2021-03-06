import numpy as np
import matplotlib.pyplot as plt

from itertools import count
from numpy.linalg import inv
from matplotlib.ticker import MultipleLocator


def get_2D_line(x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2
    return np.array([C, A, B])


def get_2D_line_function(weights):
    C, A, B = weights
    @np.vectorize
    def f(x):
        return A/-B*x + C/-B    
    return f


def get_2D_points(n_points, interval):
    points = np.ones((n_points, 3))
    points[:, 1:] = np.random.uniform(*interval, (n_points, 2))
    return points


def label_points(X, w):
    return np.sign(np.dot(X, w))


def get_2D_linearly_separated_datasets(interval, weights, train_size, test_size=None):
    X_train = get_2D_points(train_size, interval)
    y_train = label_points(X_train, weights)
    
    if test_size:
        X_test = get_2D_points(test_size, interval)
        y_test = label_points(X_test, weights)
        return X_train, y_train, X_test, y_test
    else:
        return X_train, y_train


def get_2D_binary_labeled_datasets(interval, labeler, train_size, test_size=None):
    labeler = np.vectorize(labeler)

    X_train = get_2D_points(train_size, interval)
    y_train = np.sign(labeler(*X_train[:, 1:].T))
    
    if test_size:
        X_test = get_2D_points(test_size, interval)
        y_test = np.sign(labeler(*X_test[:, 1:].T))
        return X_train, y_train, X_test, y_test
    else:
        return X_train, y_train

    
def fit_linear_regression(X, y, lambda_ = 0):
    n_objects, n_features = X.shape
    
    reguralizer = lambda_ * np.identity(n_features)
    weights = inv(X.T@X + reguralizer) @ X.T@y
    
    return weights


def fit_PLA(X, y, initial_weights=None):
    if initial_weights is None:
        initial_weights = np.zeros(X.shape[1])
    
    w_hat = initial_weights    
    for n_iter in count(start=1, step=1):
        misclf_idx = np.where(label_points(X, w_hat) != y)[0]
        if not misclf_idx.size:
            break
        rand_misclf_idx = np.random.choice(misclf_idx)
        w_hat += y[rand_misclf_idx] * X[rand_misclf_idx]        
    return w_hat, n_iter


def calculate_accuracy(X, y_true, w_hat):
    y_pred = label_points(X, w_hat)
    total = X.shape[0]
    return np.sum(y_true == y_pred) / total


def calculate_clf_error(X, y_true, w_hat):
	return 1 - calculate_accuracy(X, y_true, w_hat)


def plot_2D_points_and_lines(X, y, lines_params, INTERVAL=np.array([-1., 1.])):
    @np.vectorize
    def colorer(label_value):
        return {-1: 'red', +1: 'green'}[label_value]
    
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    
    plt.xlim(*INTERVAL*1.1)
    plt.ylim(*INTERVAL*1.1)
    majloc, minloc = [MultipleLocator(mul) for mul in (1, .1)]
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(majloc)
        axis.set_minor_locator(minloc)
    for pos in ['right', 'top']:   ax.spines[pos].set_color('none')
    for pos in ['left', 'bottom']: ax.spines[pos].set_position('zero')
    ax.set_aspect('equal')

    # Lines
    for params in lines_params:
        f = get_2D_line_function(params['weights'])
        plt.plot(INTERVAL*2, f(INTERVAL*2), linestyle=params['style'], color=params['color'])
    
    # Points
    plt.scatter(*X[:, 1:].T, c=colorer(y))

    plt.show()
