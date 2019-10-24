import math

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.utils.linear_assignment_ import linear_assignment


def beta(alphas):
    """Multivariate beta function"""
    return math.exp(sum(map(gammaln, alphas)) - gammaln(sum(alphas)))


def cumsum_ex(arr):
    """
    Function computing the cumulative sum exluding the last element for the first element the cumsum is 0
    :param arr: array of shape [p,]
    :return: cum_sum_arr: of shape [p,] where cum_sum_arr[0] = 0 and cum_sum_arr[i] = cumsum(arr[:i])
    """
    cum_sum_arr = np.zeros_like(arr)
    for i in range(len(arr)):
        if i == 0:
            cum_sum_arr[i] = 0
        else:
            cum_sum_arr[i] = np.cumsum(arr[:i])[-1]
    return cum_sum_arr


def cluster_acc(Y_pred, Y):
    """
    Function computing cluster accuracy and confusion matrix at a permutation of the labels
    :param Y_pred: The predicted labels of shape [N, ]
    :param Y: The true labels of shape [N, ]
    :return: Clusterring_accuracy, Confusion matrix
    """
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w


def get_ind_function(X):
    """
    Returns a dictionary of indicator function for values taken by the random variables over the vocabulary of each
    dimension
    :param X: data frame of shape [N, d]
    :return: dict_C: for each X_i the indicator of values taken of shape [N, |X_i|]
    """
    dict_C = {}
    N = X.shape[0]
    for column in X.columns:
        vocabulary = np.unique(X[column])
        C = np.zeros(shape=(N, len(vocabulary)))
        for i in range(len(vocabulary)):
            v = vocabulary[i]
            C[:, i] = (X[column].values == v).astype(np.int32)
        dict_C[column] = C

    return dict_C


def normalize(params, axis=0):
    """
    Function normalizing the parameters vector params with respect to the Axis: axis
    :param params: array of parameters of shape [axis0, axis1, ..., axisp] p can be variable
    :return: params: array of same shape normalized
    """

    return params / np.sum(params, axis=axis, keepdims=True)


def most_accuring_terms(df):
    """
    Function returning the most accruing assignments in a dataframe
    :param df: dataframe of shape [N, d]
    :return: list of most accruing assignments
    """
    df_made = pd.DataFrame({col: str(col) + '=' for col in df}, index=df.index) + df.astype(str)
    accuring_asignments = pd.Series(np.reshape(df_made.values.tolist(), [-1])).value_counts()[:]
    return accuring_asignments
