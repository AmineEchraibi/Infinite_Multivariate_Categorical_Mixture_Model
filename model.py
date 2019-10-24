import numpy as np
import numpy.linalg as LA
from scipy.special import digamma
import pandas as pd
from sklearn.cluster import KMeans
from utils import get_ind_function, normalize, cumsum_ex, beta, most_accuring_terms
from scipy.special import logsumexp
import random

random.seed( 30 )
def initialize_categorical_parameters(X, K, coef):
    """
    Function initializing parameters for the categorical distribution
    :param K: The truncation level
    :param X: dataframe of shape [N, d]
    :param coef: scalar
    :return: dict_E, alpha: dictionary of params of shape [|X_i|,K] and dirichlet prior for each dimension
    """
    dict_E = {}
    cards = {}
    alpha = {}
    columns = X.columns
    for c in columns:
        cards[c] = X[c].unique().shape[0]
        alpha[c] = coef * 1 / cards[c]
        E = np.random.rand(cards[c], K)
        E = normalize(E)
        dict_E[c] = E
    return dict_E, alpha, cards


def initialize_phi(N, K):
    """
    Initializing responsibilities or posterior class probabilities
    :param N: Number of instances
    :param K: Number of classes
    :return: R : responsibilities of shape [N, K]
    """

    phi = np.random.rand(N, K)
    phi = normalize(phi, axis=1)

    return phi

def initialise_phi_with_kmeans(X, K):

    mu = KMeans(K).fit(X).cluster_centers_
    phi = np.exp( - 0.5 * LA.norm(X.reshape(X.shape[0], 1, X.shape[1]) - mu.reshape(1, K, X.shape[1]),2,2))
    return normalize(phi,1), mu


class InfiniteCategoricalMixtureModel:
    """
    The Infinite categorical mixture model for a multivariate dataset with categorical values in order to investigate
    an unknown number of clusters.
    """

    def __init__(self, X, K, concentration_parameter=5, epsilon=1e-9, init="Kmeans", coef = 10):
        """
        Initialisation function
        :param K: The truncation level
        :param X: dataframe of shape [N, d]
        """
        self.dict_C = get_ind_function(X)
        self.columns = X.columns
        self.N = X.shape[0]
        self.K = K
        self.d = len(self.columns)
        self.dict_E, self.alpha, self.cards = initialize_categorical_parameters(X, K, coef)
        if init=="Kmeans":
            X_dm = pd.get_dummies(X).values
            self.phi, mu = initialise_phi_with_kmeans(X_dm,K)
        else:
            self.phi = initialize_phi(self.N, K)
        self.eps = epsilon
        self.gamma_1 = np.ones((self.K,))
        self.gamma_2 = np.ones((self.K,))
        self.eta = concentration_parameter
        self.X = X

    def compute_gamma_1(self, phi):
        """
        The function compute the first variational parameter of the beta distribution
        :param phi: the variational parameter representing the probs of classes for each sample of shape [N, K]
        :return: gamma_1: the first variational parameter of the beta distribution of shape [K, ]
        """
        gamma_1 = 1 + np.sum(phi, axis=0)
        return gamma_1

    def compute_gamma_2(self, phi):
        """
        The function compute the first variational parameter of the beta distribution
        :param phi: the variational parameter representing the probs of classes for each sample of shape [N, K]
        :return: gamma_2: the second variational parameter of the beta distribution of shape [K, ]
        """

        gamma_2 = self.eta + np.hstack((np.cumsum(np.sum(phi, axis=0)[::-1])[-2::-1], 0))
        return gamma_2

    def compute_E(self, phi):
        """
        Function computing the hyperparameters of the Dirichlet prior.
        :param phi: the variational parameter representing the probs of classes for each sample of shape [N, K]
        :return: dict_E: dictionary of params of shape [|X_i|,K] and dirichlet prior for each dimension
        """
        dict_E = {}
        for c in self.dict_C.keys():
            dict_E[c] = np.dot(self.dict_C[c].T, phi) + self.alpha[c]
        return dict_E

    def compute_phi(self, gamma_1, gamma_2, dict_E):
        """
        Function computing the variational parameters phi
        :param gamma_1: the first variational parameter of the beta distribution of shape [K, ]
        :param gamma_2:  the second variational parameter of the beta distribution of shape [K, ]
        :param dict_E:  dictionary of params of shape [|X_i|,K] and dirichlet prior for each dimension
        :return: phi: variational parameters of the class probs for each sample of shape [N, K]
        """
        log_phi = np.expand_dims(digamma(gamma_1) - digamma(gamma_1 + gamma_2)
                                 + cumsum_ex(digamma(gamma_2) - digamma(gamma_1 + gamma_2)), axis=0)


        for c in dict_E.keys():
            E = dict_E[c]
            C = self.dict_C[c]
            alpha = self.alpha[c]
            log_phi = log_phi + np.dot( C, digamma(E) - digamma(np.sum(E, axis=0, keepdims=True)))

        log_phi = log_phi - logsumexp(log_phi, axis=1)[:, np.newaxis]
        phi = normalize(np.exp(log_phi), 1)

        return phi

    def gradient_ascent(self, max_iter=1000, debug=False):
        """
        The gradient ascent algorithm using the fixed point equations
        :param max_iter: Number of max iterations
        :param debug: debug if True
        :return: L: List the evidence lower bound at each iteration
        """
        L = []
        stop_criterion = False
        for i in range(max_iter):

            # Fixed point equations for gradient ascent

            self.gamma_1 = self.compute_gamma_1(self.phi)
            self.gamma_2 = self.compute_gamma_2(self.phi)
            self.dict_E = self.compute_E(self.phi)
            # Compute evidence lower bound
            l = self.compute_elbo(self.phi, self.gamma_1, self.gamma_2, self.dict_E)
            self.phi = self.compute_phi(self.gamma_1, self.gamma_2, self.dict_E)


            if debug:
                print("[DEBUG] elbo at iteration ", i, " is ", l)
            L.append(l)

            # Stopping criterion
            if len(L) > 2:
                stop_criterion = np.abs((L[-1] - L[-2]) / L[-2]) < 1e-9
            if stop_criterion:
                break

        return L

    def compute_elbo(self, phi, gamma_1, gamma_2, dict_E):
        """
        Function compute the evidence lower bound as defined for DPCMM.
        :param phi: variational parameters of the class probs for each sample of shape [N, K]
        :param gamma_1: gamma_1: the first variational parameter of the beta distribution of shape [K, ]
        :param gamma_2:  the second variational parameter of the beta distribution of shape [K, ]
        :param dict_E:  dictionary of params of shape [|X_i|,K] and dirichlet prior for each dimension
        :return: L: scalar of the evidence lower bound should increase with each iteration.
        """
        entropy_beta = 0
        entropy_b = 0
        log_p_x = 0
        L = (self.eta - 1) * np.sum(digamma(gamma_2) - digamma(gamma_2 + gamma_1)) \
            + np.sum(np.sum(phi * np.expand_dims(
            digamma(gamma_1) - digamma(gamma_1 + gamma_2) + cumsum_ex(digamma(gamma_2) - digamma(gamma_1 + gamma_2)),
            axis=0)))

        for k in range(self.K):
            entropy_beta = entropy_beta + np.log(self.eps + beta([gamma_1[k], gamma_2[k]])) \
                           + (gamma_1[k] + gamma_2[k] - 2) * digamma(gamma_1[k] + gamma_2[k]) \
                           - (gamma_1[k] - 1) * digamma(gamma_1[k]) \
                           - (gamma_2[k] - 1) * digamma(gamma_2[k])
            for c in self.dict_C.keys():
                E = dict_E[c]
                entropy_b = entropy_b + np.log(beta(E[:, k].tolist())) + (np.sum(E[:, k]) - self.cards[c]) * digamma(
                    np.sum(E[:, k])) - np.sum((E[:, k] - 1) * digamma(E[:, k]))

        for c in self.dict_C.keys():
            E = dict_E[c]
            C = self.dict_C[c]
            alpha = self.alpha[c]
            log_p_x = log_p_x + np.sum(np.sum((alpha + np.dot(C.T, phi) - 1) * (digamma(E)
                                                                        - digamma(
                                                                                   np.sum(E, axis=0, keepdims=True)
                                                                               )
                                                                        )
                                                                       )
                                       )


        entropy_phi = - np.sum(np.sum(phi * np.log(phi + self.eps)))
        L = L + log_p_x + entropy_phi + entropy_b + entropy_beta # if numerical instability remove entropy terms not necessary to diagnose convergence

        return L

    def infer_clusters(self):
        """
        Function returning the clustering assignments for each data sample
        :return: y_pred: array of shape [N, ]
        """
        return np.argmax(self.phi, axis=1)

    def explain_cluster(self, inferred_labels):
        """
        Function returning the most common assignments for each cluster.
        :param inferred_labels: The labels inferred using the DPMM of shape [N,]
        :return:
        """
        for cluster_index in np.unique(inferred_labels):
            inds = np.where(inferred_labels == cluster_index)[0]
            cluster_data = self.X.iloc[inds, :]
            accurring_assignments = most_accuring_terms(cluster_data)
            print("[EXPLAIN] Most accuring terms for cluster ", cluster_index, " : ")
            print("*******************************************************************************************")
            import pandas as pd
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(accurring_assignments)
            print("*******************************************************************************************")
