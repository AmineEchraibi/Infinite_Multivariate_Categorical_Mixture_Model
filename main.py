import sys

import matplotlib.pyplot as plt
import pandas as pd

from model import *
from utils import cluster_acc


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

import argparse


parser = argparse.ArgumentParser(description='Applying the IMCMM on a categorical dataframe')

parser.add_argument('--filepath', action="store", dest='filepath', help="File path for the dataframe, seperator for dataframe tab")
parser.add_argument('--cluster_dim_name', action="store",dest="c_dim", help="The column name for test on already \
labeled data to verify performance otherwise None unsupervised learning", default=None)

def main(filename, cluster_dimension=None, K=50):
    data = pd.read_csv(filename, sep='\t', engine='python').astype(str)
    print(data.columns)
    if cluster_dimension != None:
        y = pd.Categorical(data[cluster_dimension]).codes
        X = data.drop(cluster_dimension, axis=1)
    else:
        X = data
    model = InfiniteCategoricalMixtureModel(X, concentration_parameter=0.01, K=K)
    L = model.gradient_ascent(max_iter=1000, debug=True)
    y_pred = model.infer_clusters()
    if cluster_dimension != None:
        print("preformance of the model : ", cluster_acc(y_pred, y)[0])
    model.explain_cluster(y_pred)
    resultdf = data
    resultdf["cluster"] = y_pred
    plt.plot(L)
    plt.xlabel('number of itterations')
    plt.ylabel('elbo')
    plt.title('elbo of the model')
    plt.show()


if __name__ == '__main__':
    p = parser.parse_args()
    filename = p.filepath
    clust_dim = p.c_dim
    main(filename, clust_dim)

