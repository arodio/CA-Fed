# Dataset & Configuration

## Introduction

### Datasets

Four (4) datasets are available for the experiments:
* **Synthetic Clustered**: binary classification dataset, described in [Appendix F, Our Paper];
* **Synthetic LEAF**: multi-class classification dataset, implemented in the
  [LEAF repository](https://github.com/TalwalkarLab/leaf); 
* **MNIST**: multi-class classification dataset of handwritten digits;
* **CIFAR-10**: multi-class classification dataset of images.

### Splits

The synthetic datasets, e.g., `synthetic_clustered` and `synthetic_leaf`,
are split during data generation.
Split a classification dataset, e.g., `MNIST` or `CIFAR-10`, among `n_clients`. Two splitting strategies are available:

#### IID split (default)

The dataset is randomly shuffled and partitioned among `n_clients`.

#### Incongruent split

The dataset is split among `n_clients`, each of which belonging to one of two groups,
as follows:

1. the dataset is randomly shuffled and partitioned among `n_clients`
2. clients are randomly partitioned into `n_clusters` 
3. the data of clients from the second group is  modified 
   by  randomly  swapping out `k` pairs of labels
   
### Configuration 

In order to simulate the heterogeneity of clients availability patterns
in realistic federated systems, we split the clients in two classes uniformly at random:
* ``available`` clients with probability $\pi = 1/2 + f$ of being active
  asymptotically
* ``unavailable`` clients with  $\pi^{'} = 1/2 - f$ of being
  inactive asymptotically, where $f_{i}\in(0, 1/2)$ is a parameter controlling
  the heterogeneity of clients' asymptotic availability.
  
We furthermore split each class of clients in three sub-classes uniformly at random:
* ``stable`` clients that tend to keep the same activity state,
  with $\lambda^{(1)}=\nu$
* ``unstable`` clients that tend to switch their activity state frequently,
  with $\lambda^{(2)}=-\nu$
* ``shifting`` clients that are as likely to keep as to change their activity state,
  with $\lambda^{(3)} \sim \mathcal{N} \left(0, \varepsilon^{2}\right)$.

A deterministic split of clients into the different classes (e.g., clients are
equally split among classes), is available using the argument ```--deterministic_split```.
To vary the proportion of clients in the sub-classes, please refer to the configuration
file ```JOINT_PROBABILITY_MATRIX``` in ```data/constants.py```.

## Instructions

Run `main.py` with a choice of the following arguments:

* ```--dataset```: dataset to use, possible are `'synthetic_clustered'`, 
  `'synthetic_leaf'`, `'mnist'` and `'cifar10'`
* ```--n_tasks```: (`int`) number of tasks/clients, given as an integer
* ```--n_clusters```: (`int`) number of clusters / groups; default=`2`, 
  only used when `experiment_name=='synthetic_clustered'`
* ```--n_classes```: (`int`) number of classes, default=`10`, only used 
  when `experiment_name=='synthetic_leaf`
* ```--dimension```: (`int`) dimension of the data, only used 
  when `experiment_name=={'synthetic_clustered', 'synthetic_leaf'}`
* ```--n_train_samples```: (`int`) number of train samples, 
  only used when `experiment_name=='synthetic_clustered'`
* ```--n_test_samples```: (`int`) number of test samples, only used 
  when `experiment_name=='synthetic_clustered'`
* ```--hetero_param```: (`float`) parameter controlling clients dissimilarity,
  only used when `experiment_name=='synthetic_clustered'`
* ```--alpha```: (`float`) parameter controlling how much local models differ from 
  each other; only used when `experiment_name=='synthetic_leaf'`;
  expected to be in the range (0,1);
  default=`1.0`
* ```--beta```: (`float`) parameter controlling how much the local data at each device differs from that 
  of other devices; only used when `experiment_name=='synthetic_leaf'`;
  expected to be in the range (0,1);
  default=`1.0`
* ```--iid```: (`bool`) if selected, the iid split described above is used
* ```--incongruent_split```: (`bool`) if selected, the incongruent split described
  above is used
* ```--n_swapping_pairs```: (`int`) number of pairs to swap; default=`1`
* ```--deterministic_split```: (`bool`) if selected, the deterministic split as
  above is used
* ```--augmentation```: (`bool`) if selected, more samples are assigned to the 
  more available clients
* ```--availability_parameters```: (`list`) parameters controlling the asymptotic
  availability of the clients from each group/cluster; 
  should be a list of the same size as `n_clusters`;
  default=`[0.0, 0.25]`
* ```--stability_parameters```: (`list`) list of parameters controlling the 
  stability of clients from each group/cluster; 
  should be a list of the same size as `n_clusters`;
  default=`[0.1, 0.25]`
* ```--save_dir```: (`str`) path of the directory to save data and configuration; 
  if not specified the data is saved to `./{dataset_name}`
* ```--seed```: (int) seed to be used to initialize the random number generator;
  if not provided, the system clock is used to generate the seed'
  
## Paper Experiments

### Synthetic Clustered

In order to generate the data split and configuration for the synthetic clustered dataset experiment, run

```
python main.py \
    --dataset synthetic_clustered \
    --n_tasks 24 \
    --n_clusters 2 \
    --dimension 10 \
    --n_train_samples 150 \
    --n_test_samples 500 \
    --hetero_param 0.2 \
    --availability_parameters 0.4 0.4 \
    --stability_parameters 0.9 0.9 \
    --save_dir synthetic_clustered \
    --seed 42 
```

### Synthetic LEAF

In order to generate the data split and configuration for the synthetic LEAF dataset experiment, run

```
python main.py \
    --dataset synthetic_leaf \
    --n_tasks 24 \
    --n_classes 10 \
    --dimension 60 \
    --alpha 0.0 \
    --beta 0.0 \  
    --availability_parameters 0.4 0.4 \
    --stability_parameters 0.9 0.9 \
    --save_dir synthetic_leaf \
    --seed 42 
```

### MNIST

In order to generate the data split and configuration for MNIST experiment, run

```
python main.py \
    --dataset mnist \
    --n_tasks 24 \
    --incongruent_split \
    --n_clusters 2 \
    --n_swapping_pairs 2 \
    --availability_parameters 0.4 0.4 \
    --stability_parameters 0.9 0.9 \
    --save_dir mnist \
    --seed 42 
```

### CIFAR-10

In order to generate the data split and configuration for CIFAR-10 experiment, run

```
python main.py \
    --dataset cifar10 \
    --n_tasks 24 \
    --incongruent_split \
    --n_clusters 2 \
    --n_swapping_pairs 2 \
    --availability_parameters 0.4 0.4 \
    --stability_parameters 0.9 0.9 \
    --save_dir cifar10 \
    --seed 42 
```