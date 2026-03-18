import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as sk_r2_score


def mae(outputs, labels):
    return mean_absolute_error(labels, outputs)

def mse(outputs, labels):
    return mean_squared_error(labels, outputs)

def rmse(outputs, labels):
    return np.sqrt(mean_squared_error(labels, outputs))

def r2_score(outputs, labels):
    return sk_r2_score(labels, outputs)

def mean_error(outputs, labels):
    return np.mean(outputs - labels)

def std_error(outputs, labels):
    return np.std(outputs - labels)


metrics = {
    'mae': mae,
    'mse': mse,
    'rmse': rmse,
    'r2score': r2_score,
    'mean_error': mean_error,
    'std_error': std_error
}
