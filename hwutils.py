import numpy as np

from itertools import count
from numpy.linalg import inv

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
    return np.append(np.ones((n_points, 1)),
                     np.random.uniform(*interval, (n_points, 2)),
                     axis=1)

def label_points(X, w):
    return np.sign(np.dot(X, w))

def fit_linear_regression(X, y):
    return inv(X.T @ X) @ X.T @ y

def fit_PLA(X, y, initial_weights=np.zeros(3)):
    w_hat = initial_weights    
    for n_iter in count(start=1, step=1):
        misclf_idx = np.where(label_points(X, w_hat) != y)[0]
        if not misclf_idx.size:
            break
        rand_misclf_idx = np.random.choice(misclf_idx)
        w_hat += y[rand_misclf_idx] * X[rand_misclf_idx]        
    return w_hat, n_iter

def calculate_accuracy(X, w_true, w_hat):
    y_true = label_points(X, w_true)
    y_pred = label_points(X, w_hat)
    total = X.shape[0]
    return np.sum(y_true == y_pred) / total
