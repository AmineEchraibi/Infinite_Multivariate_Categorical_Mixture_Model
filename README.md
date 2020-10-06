## Introduction

This notebook threats the problem of identifying patterns of behavior in networks and services, we present the Dirichlet process categorical mixture model. We suppose that all the data has been transformed into discrete categorical data. We apply the model to data gathered from GPON-FTTH network, and...

## Notations and data
### Random variables:
* We denote random variables as <img src="https://render.githubusercontent.com/render/math?math=X_1, X_2, ..., X_d"> where <img src="https://render.githubusercontent.com/render/math?math=X_i">  is the ith random variable (Power or alarm, ..). 
* We denote by <img src="https://render.githubusercontent.com/render/math?math=Val(X_i)"> as the set of discrete values taken by <img src="https://render.githubusercontent.com/render/math?math=X_i">.
* We denote <img src="https://render.githubusercontent.com/render/math?math=x_{ni}"> the nth instance of variable <img src="https://render.githubusercontent.com/render/math?math=X_i"> in the dataset.
* <img src="https://render.githubusercontent.com/render/math?math=\mathcal{D} = (x_{ni})_{n,i}"> : The complete dataset
* We denote by <img src="https://render.githubusercontent.com/render/math?math=z_n"> The hidden class variable for instance n.


## Description of the DPCMM

### Model
<div class="pull-right">
<img src="platenotation.png", width="200",alt="">
</div>



### Inference on the model:
The inference in the DPCMM is done by computing the posterior distribution: 

<img src="https://render.githubusercontent.com/render/math?math= p(\beta, z, b | \mathcal{D}) = \frac{p(\beta, z, b, \mathcal{D})}{p(\mathcal{D})} ">
We can drop the constant term <img src="https://render.githubusercontent.com/render/math?math=p(\mathcal{D})"> and the objectif is to compute or estimate <img src="https://render.githubusercontent.com/render/math?math=p(\beta, z, b, \mathcal{D})"> We have:

<img src="https://render.githubusercontent.com/render/math?math= p(\beta, z, b, \mathcal{D}) = \prod_{i=1}^d \prod_{n=1}^N p(x_{ni}|z_n, b_{z_n, i})p(z_n|\beta) \prod_{k=1}^\infty p(b_{k,i})  p(\beta_k) ">

Computing this quantity in close forme is not possible, the prior is a product of infinite terms and infinite parameters. One approach is to preform MCMC methods such as Gibbs sampling, where the Markov chain converges to the posterior of interest. The evaluation of convergence of these methods is hard and scalability is a problem also. In the following section we introduce the approach of variational inference this approach tries to approximate the posterior of interest using a family of simpler and tractable distributions. The problem is thus to find the best member of the family that is closest (in terms of kullback leibler divergence) to the posterior of interest. Thus transforming the problem of inference into an optimization problem where we can use all the mathematical background in this field to solve the inference problem. 

### Mean Field variational inference [Murphy, Blei, ...]:

The main idea is to approximate <img src="https://render.githubusercontent.com/render/math?math=p(\beta, z, b, \mathcal{D}) "> using simpler family of distribution, the mean field approach tracks back to physics and probablity theory (https://en.wikipedia.org/wiki/Mean_field_theory) For our case the 

mean field family can be written as:
<img src="https://render.githubusercontent.com/render/math?math=q(z,b,\beta) = \prod_{k=1}^K \prod_{i=1}^d q(b_{k,i}) q(\beta_k) \prod_{n=1}^N q(z_n)">
We suppose that the approximating distribution is a simple factored distribution, the infinite parameters are truncated into a level K so the product becomes finite.

And the objective is to solve the following optimization problem:
<img src="https://render.githubusercontent.com/render/math?math=\min_q \mathbb{D}_{KL} [q || p]">

The mean field theorem states that the solution to this equation <img src="https://render.githubusercontent.com/render/math?math=q^{*}"> for the set of parameters <img src="https://render.githubusercontent.com/render/math?math=\zeta = \{z, \beta, b \}"> verifies: 


<img src="https://render.githubusercontent.com/render/math?math=\log q_{j}^{*}(\zeta_j) = const + \mathbb{E}_{\zeta \setminus \{\theta_j \} \sim q^{*}} [\log p(\zeta , \mathcal{D})]">

In order to compute the approximating distribution, we need to compute the expectancies, which for exponential family distributions is tractable.


### Experiments on GPON-FTTH data:




```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils import cluster_acc
from model import *

pd.options.display.max_rows = 4000
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(16,12)})

filename = "BN_data.csv"
cluster_dim = "label"

```


```python
data = pd.read_csv(filename, sep='\t', engine='python').astype(str)

# Clustering the data using the IMCMM
X = data.drop(cluster_dim, axis=1) # Droping the labels
y_true = pd.Categorical(data[cluster_dim])
K_true = y.unique().shape[0]

truncation_level = 50
L = []
y_pred = 0
n_runs = 15
accuracies = [0]*n_runs
for t in range(n_runs):
    model = InfiniteCategoricalMixtureModel(X, concentration_parameter=0.001, K=truncation_level, coef=10)
    L_t = model.gradient_ascent(max_iter=1000,debug=False)
    y_pred_t = model.infer_clusters()
    accuracies[t] =  cluster_acc(y_pred_t, y_true.codes)
    L.append(L_t)
```


```python
# Ploting the evidence lower bound for each  run of the algorithm
for i in range(n_runs):
    plt.plot(range(len(L[i]))[:12], L[i][:12], label="Experiment " + str(i + 1))
plt.xlabel('iterations', fontsize=15)
plt.ylabel('evidence lower bound', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.savefig("VLB.pdf",bbox_inches="tight")
```


    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_7_0.png)
    



```python
# Identifying the best model from the elbo :

highest_l = -np.inf
for i in range(n_runs):
    if highest_l < L[i][-1]:
        highest_l = L[i][-1]
        index = i

print("Best model clustering accuracy : ", accuracies[index][0])
```

    Best model clustering accuracy :  0.9677777777777777



```python
acc = accuracies[index]
df = pd.DataFrame(acc[1])
df = df.loc[:, (df != 0).any(axis=0)]
D_c = {}
for i in range(len(y.categories)):
    D_c[i] = y.categories[i]
D_r = {}   
for i in range(50):
    D_r[i] = "cluster "+str(i)
df = df.rename(columns = D_c, index=D_r)
# print("Confusion matrix : ")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AltOLT</th>
      <th>AltONT</th>
      <th>FaultyONT</th>
      <th>FiberDB</th>
      <th>IOS</th>
      <th>TcOLT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cluster 0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cluster 1</th>
      <td>0</td>
      <td>17</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>145</td>
    </tr>
    <tr>
      <th>cluster 2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cluster 3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cluster 4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>150</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>cluster 5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>147</td>
      <td>3</td>
    </tr>
    <tr>
      <th>cluster 6</th>
      <td>0</td>
      <td>132</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cluster 7</th>
      <td>0</td>
      <td>0</td>
      <td>147</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cluster 8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cluster 9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cluster 10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cluster 11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cluster 12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cluster 13</th>
      <td>150</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
filename = "/home/mlws/Desktop/data/diag_data.csv"

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
data = pd.read_csv(filename, sep=',', engine='python').astype(str)
#y = pd.Categorical(data[cluster_dim])
#K = y.unique().shape[0]
T = 50
print(K)
#X = data.drop(cluster_dim, axis=1)

model = InfiniteCategoricalMixtureModel(data, concentration_parameter=0.001, K=T)
L = model.gradient_ascent(max_iter=1000,debug=False)
y_pred = model.infer_clusters()
result_df = data
result_df['cluster'] = y_pred
result_df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>voip_status</th>
      <th>olt_status</th>
      <th>ont_status</th>
      <th>ont_download_status</th>
      <th>router_ipv6_status</th>
      <th>router_line_status</th>
      <th>router_status</th>
      <th>router_sfp_status</th>
      <th>router_pnp_status</th>
      <th>router_WiFiCommSecured_status</th>
      <th>...</th>
      <th>router_WiFi_status</th>
      <th>router_plug&amp;play_status</th>
      <th>RemotePVR_status</th>
      <th>SVOD_ftth_status</th>
      <th>TVconf_ftth_status</th>
      <th>tv_profile_status</th>
      <th>rfs_status</th>
      <th>client_account_status</th>
      <th>uhd_device_status</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VOIP_KO</td>
      <td>OK</td>
      <td>Missing</td>
      <td>Missing</td>
      <td>ENABLED</td>
      <td>UP</td>
      <td>ENABLED</td>
      <td>Missing</td>
      <td>TRUE</td>
      <td>DEACTIVATED</td>
      <td>...</td>
      <td>UP</td>
      <td>Missing</td>
      <td>Missing</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>1.0</td>
      <td>Missing</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VOIP_OK</td>
      <td>OK</td>
      <td>NODEFECT</td>
      <td>PLANNED</td>
      <td>ENABLED</td>
      <td>UP</td>
      <td>ENABLED</td>
      <td>Missing</td>
      <td>TRUE</td>
      <td>DEACTIVATED</td>
      <td>...</td>
      <td>DOWN</td>
      <td>Missing</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>1.0</td>
      <td>Missing</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VOIP_OK</td>
      <td>OK</td>
      <td>Missing</td>
      <td>Missing</td>
      <td>ENABLED</td>
      <td>UP</td>
      <td>ENABLED</td>
      <td>OK</td>
      <td>TRUE</td>
      <td>DEACTIVATED</td>
      <td>...</td>
      <td>UP</td>
      <td>EUA_CERTIFIED</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>1.0</td>
      <td>UP</td>
      <td>28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VOIP_KO</td>
      <td>OK</td>
      <td>NODEFECT</td>
      <td>PLANNED</td>
      <td>ENABLED</td>
      <td>UP</td>
      <td>ENABLED</td>
      <td>Missing</td>
      <td>TRUE</td>
      <td>DEACTIVATED</td>
      <td>...</td>
      <td>UP</td>
      <td>Missing</td>
      <td>Missing</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>1.0</td>
      <td>Missing</td>
      <td>23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VOIP_OK</td>
      <td>OK</td>
      <td>NODEFECT</td>
      <td>PLANNED</td>
      <td>Missing</td>
      <td>Missing</td>
      <td>Missing</td>
      <td>Missing</td>
      <td>TRUE</td>
      <td>Missing</td>
      <td>...</td>
      <td>Missing</td>
      <td>Missing</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>OPERATIONAL</td>
      <td>1.0</td>
      <td>Missing</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
filename = "/home/mlws/Desktop/data/diag_data.csv"
import pandas as pd
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
data = pd.read_csv(filename, sep=',', engine='python').astype(str)
```


```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
from utils import most_accuring_terms
i=0
for cluster_index in np.unique(y_pred):
    inds = np.where(y_pred == cluster_index)[0]
    cluster_data = data.iloc[inds, :]
    accurring_assignments = most_accuring_terms(cluster_data)
    accurring_assignments = accurring_assignments.to_frame()
    accurring_assignments = accurring_assignments.drop("cluster="+str(cluster_index),axis=0)
    index_t = list(accurring_assignments.index)
    vals = list(accurring_assignments[0].values)
    plt.figure(i)
    plot_df = pd.DataFrame({"terms":index_t, "counts":vals}).iloc[:30, :]
    plot_df.to_csv("result_df_cluster"+str(cluster_index), index=None)
    ax = sns.barplot(x="counts", y="terms", data=plot_df)
    plt.title("Cluster "+str(cluster_index + 1), fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("counts", fontsize=15)
    plt.ylabel("", fontsize=15)
    plt.tight_layout()
    plt.savefig("cluster_"+str(i+1)+".pdf")
    i+=1
```


    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_0.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_1.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_2.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_3.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_4.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_5.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_6.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_7.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_8.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_9.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_10.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_11.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_12.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_13.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_14.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_15.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_16.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_17.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_18.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_19.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_20.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_21.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_22.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_23.png)
    



    
![png](model_analysis_and_experiments_GPON_Data_files/model_analysis_and_experiments_GPON_Data_12_24.png)
    

