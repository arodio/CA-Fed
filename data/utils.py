import os
import time
import json
import warnings

from tqdm import tqdm

from torchvision.datasets import CIFAR10, MNIST

from constants import *


def save_cfg(save_path, cfg):
    with open(save_path, "w") as f:
        json.dump(cfg, f)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def save_data(save_dir, train_data, train_targets, test_data, test_targets):
    """save data and targets as `.npy` files

    Parameters
    ----------
    save_dir: str
        directory to save data; it will be created it it does not exist

    train_data: numpy.array

    train_targets: numpy.array

    test_data: numpy.array

    test_targets: numpy.array

    """
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "train_data.npy"), "wb") as f:
        np.save(f, train_data)

    with open(os.path.join(save_dir, "train_targets.npy"), "wb") as f:
        np.save(f, train_targets)

    with open(os.path.join(save_dir, "test_data.npy"), "wb") as f:
        np.save(f, test_data)

    with open(os.path.join(save_dir, "test_targets.npy"), "wb") as f:
        np.save(f, test_targets)


def get_dataset(dataset_name, raw_data_path):
    if dataset_name == "cifar10":
        dataset = CIFAR10(root=raw_data_path, download=True, train=True)
        test_dataset = CIFAR10(root=raw_data_path, download=True, train=False)

        dataset.data = np.concatenate((dataset.data, test_dataset.data))
        dataset.targets = np.concatenate((dataset.targets, test_dataset.targets))

    elif dataset_name == "mnist":

        dataset = MNIST(root=raw_data_path, download=True, train=True)
        test_dataset = MNIST(root=raw_data_path, download=True, train=False)

        dataset.data = np.concatenate((dataset.data, test_dataset.data))
        dataset.targets = np.concatenate((dataset.targets, test_dataset.targets))

    else:
        error_message = f"{dataset_name} is not available, possible datasets are:"
        for n in DATASETS:
            error_message += n + ",\t"

        raise NotImplementedError(error_message)

    return dataset


def iid_divide(l_, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py

    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups

    """
    num_elems = len(l_)
    group_size = int(len(l_) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l_[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l_[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l_, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l_[current_index: index])
        current_index = index

    return res


def incongruent_split(
        dataset,
        n_classes,
        n_train_samples,
        n_tasks,
        n_clusters,
        n_swapping_pairs,
        availability_parameters,
        stability_parameters,
        save_dir,
        rng=None
):
    r"""incongruent split of classification dataset among `n_tasks`

    The dataset is split as follows:
        1) randomly shuffle the dataset
        2) group the dataset into equally sized chunks
        3) assign each chunk to a task
        4) partition the tasks into `n_clusters`
        3) the dataset of tasks from a given group is
         modified by randomly swapping out `n_swapping_pairs` pairs of labels

    Parameters
    ----------
    dataset: torch.utils.Dataset
        a classification dataset;
         expected to have attributes `data` and `targets` storing `numpy.array` objects

    n_classes: int
        number of classes in the dataset

    n_train_samples: int
        number of training samples

    n_tasks: int
        number of tasks

    n_clusters: int
        number of clusters

    n_swapping_pairs: int
        number of pairs to swap

    availability_parameters: List
        parameters controlling the asymptotic availability of the clients_dict from each group/cluster;
        should be a list of the same size as `n_clusters`

    stability_parameters: List
        list of parameters controlling the stability of clients_dict from each group/cluster;
        should be a list of the same size as `n_clusters`

    save_dir: str
        directory to save data for all tasks

    rng: random number generator; default is None

    Returns
    -------
        * List[Dict]: list (size=`n_tasks`) of dictionaries, storing the data and metadata for each client

    """

    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    n_samples = len(dataset)

    tasks_indices = iid_divide(rng.permutation(n_samples), n_tasks)

    clients_per_cluster = iid_divide(rng.permutation(n_tasks), n_clusters)

    all_tasks_cfg = dict()

    for cluster_id, tasks_ids in enumerate(tqdm(clients_per_cluster, desc="Clusters..")):

        labels_to_swap = rng.choice(n_classes, size=(n_swapping_pairs, 2), replace=False)

        current_n_tasks = len(tasks_ids)

        tasks_types = generate_tasks_types(
            num_tasks=current_n_tasks,
            joint_probability=np.array(JOINT_PROBABILITY_MATRIX),
            deterministic_split=False,
            rng=rng
        )

        for task_id, (availability_id, stability_id) in \
                tqdm(zip(tasks_ids, tasks_types), total=current_n_tasks, desc="Tasks.."):

            client_indices = tasks_indices[task_id]
            train_indices = client_indices[client_indices < n_train_samples]
            test_indices = client_indices[client_indices >= n_train_samples]

            train_data, train_targets = dataset.data[train_indices], dataset.targets[train_indices]
            test_data, test_targets = dataset.data[test_indices], dataset.targets[test_indices]

            for val1, val2 in labels_to_swap:
                train_targets = np.where(
                    train_targets == val1,
                    val2,
                    np.where(train_targets == val2, val1, train_targets)
                )

                test_targets = np.where(
                    test_targets == val1,
                    val2,
                    np.where(test_targets == val2, val1, test_targets)
                )

            task_dir = os.path.join(os.getcwd(), save_dir, f"task_{task_id}")

            save_data(
                save_dir=task_dir,
                train_data=train_data,
                train_targets=train_targets,
                test_data=test_data,
                test_targets=test_targets
            )

            availability = \
                compute_availability(
                    availability_type=AVAILABILITY_TYPES[availability_id],
                    availability_parameter=availability_parameters[cluster_id]
                )

            stability = \
                compute_stability(
                    stability_type=STABILITY_TYPES[stability_id],
                    stability_parameter=stability_parameters[cluster_id],
                    availability=availability,
                    rng=rng
                )

            all_tasks_cfg[str(task_id)] = {
                "cluster_id": cluster_id,
                "labels_to_swap": labels_to_swap.tolist(),
                "indices": client_indices.tolist(),
                "task_dir": task_dir,
                "availability_type": AVAILABILITY_TYPES[availability_id],
                "availability": availability,
                "stability_type": STABILITY_TYPES[stability_id],
                "stability": stability
            }

    return all_tasks_cfg


def generate_synthetic_leaf(
        n_tasks,
        n_classes,
        dimension,
        alpha,
        beta,
        iid,
        availability_parameters,
        stability_parameters,
        deterministic_split,
        augmentation,
        save_dir,
        rng
):
    """generate synthetic dataset

    Parameters
    ----------

    n_classes: int

    n_tasks: int

    dimension: int

    alpha: float

    beta: float

    iid: bool

    availability_parameters: List[float]
        parameters controlling the asymptotic availability of the clients_dict from each group/cluster;
        should be a list of the same size as `n_clusters`

    stability_parameters: List[float]
        list of parameters controlling the stability of clients_dict from each group/cluster;
        should be a list of the same size as `n_clusters`

    deterministic_split: bool
        split tasks per task type in a deterministic or stochastic way, according to JOINT_PROBABILITY_MATRIX

    augmentation: bool
        if True, more samples are assigned to the more available clients

    save_dir: str
        directory to save train_data for all tasks

    rng: random number generator; default is None

    Returns
    -------
        * List[Dict]: list (size=`n_tasks`) of dictionaries, storing the train_data and metadata for each client
    """
    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    all_tasks_cfg = dict()

    tasks_types = generate_tasks_types(
        num_tasks=n_tasks,
        joint_probability=np.array(JOINT_PROBABILITY_MATRIX),
        deterministic_split=deterministic_split,
        rng=rng
    )

    samples_per_user = compute_samples_per_user(
        n_tasks=n_tasks,
        tasks_types=tasks_types,
        rng=rng,
        augmentation=augmentation
    )

    X = [[] for _ in range(n_tasks)]
    y = [[] for _ in range(n_tasks)]

    # prior for parameters
    mean_W = rng.normal(0, alpha, n_tasks)
    mean_b = mean_W
    B = rng.normal(0, beta, n_tasks)

    mean_x = np.zeros((n_tasks, dimension))
    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(n_tasks):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = rng.normal(B[i], 1, dimension)

    W_global = b_global = None
    if iid == 1:
        W_global = rng.normal(0, 1, (dimension, n_classes))
        b_global = rng.normal(0, 1, n_classes)

    for i in range(n_tasks):

        if iid == 1:
            assert W_global is not None and b_global is not None
            W = W_global
            b = b_global
        else:
            W = rng.normal(mean_W[i], 1, (dimension, n_classes))
            b = rng.normal(mean_b[i], 1, n_classes)

        xx = rng.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X[i].extend(xx.tolist())
        y[i].extend(yy.tolist())

    for i in range(n_tasks):
        task_dir = \
            os.path.join(os.getcwd(), save_dir, f"task_{str(i)}")

        combined = list(zip(X[i], y[i]))
        rng.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        train_data = X[i][:train_len]
        test_data = X[i][train_len:]
        train_targets = y[i][:train_len]
        test_targets = y[i][train_len:]

        save_data(
            save_dir=task_dir,
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets
        )

        availability_id = tasks_types[i][0]
        stability_id = tasks_types[i][1]

        availability = \
            compute_availability(
                availability_type=AVAILABILITY_TYPES[availability_id],
                availability_parameter=availability_parameters[0]
            )

        stability = \
            compute_stability(
                stability_type=STABILITY_TYPES[stability_id],
                stability_parameter=stability_parameters[0],
                availability=availability,
                rng=rng
            )

        all_tasks_cfg[int(i)] = {
            "cluster_id": 0,
            "task_dir": task_dir,
            "availability_type": AVAILABILITY_TYPES[availability_id],
            "availability": availability,
            "stability_type": STABILITY_TYPES[stability_id],
            "stability": stability
        }

    return all_tasks_cfg


def generate_synthetic_clustered(
        n_tasks,
        n_clusters,
        dimension,
        n_train_samples,
        n_test_samples,
        hetero_param,
        availability_parameters,
        stability_parameters,
        deterministic_split,
        save_dir,
        rng
):
    """generate synthetic dataset

    Parameters
    ----------
    n_clusters: int

    n_tasks: int

    dimension: int

    n_train_samples: int

    n_test_samples: int

    hetero_param: float

    availability_parameters: List[float]
        parameters controlling the asymptotic availability of the clients_dict from each group/cluster;
        should be a list of the same size as `n_clusters`

    stability_parameters: List[float]
        list of parameters controlling the stability of clients_dict from each group/cluster;
        should be a list of the same size as `n_clusters`

    deterministic_split: bool
        split tasks per task type in a deterministic or stochastic way, according to JOINT_PROBABILITY_MATRIX

    save_dir: str
        directory to save train_data for all tasks

    rng: random number generator; default is None

    Returns
    -------
        * List[Dict]: list (size=`n_tasks`) of dictionaries, storing the train_data and metadata for each client
    """
    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    all_tasks_cfg = dict()

    n_tasks_per_cluster = max(1, n_tasks // n_clusters)

    parameters_vector = rng.normal(loc=0., scale=1, size=(dimension,))

    train_data = rng.normal(size=(n_clusters, n_tasks_per_cluster, n_train_samples, dimension))
    test_data = rng.normal(size=(n_clusters, n_tasks_per_cluster, n_test_samples, dimension))

    true_train_probability = sigmoid(train_data @ parameters_vector)
    true_test_probability = sigmoid(test_data @ parameters_vector)

    if n_clusters == 2:

        train_probability = np.copy(true_train_probability)
        test_probability = np.copy(true_test_probability)

        train_probability[1] = \
            hetero_param * (1 - true_train_probability[1]) + (1 - hetero_param) * true_train_probability[1]

        test_probability[1] = \
            hetero_param * (1 - true_test_probability[1]) + (1 - hetero_param) * true_test_probability[1]

        train_targets = rng.binomial(n=1, p=train_probability)
        test_targets = rng.binomial(n=1, p=test_probability)

        tasks_types = generate_tasks_types(
            num_tasks=n_tasks,
            joint_probability=np.array(JOINT_PROBABILITY_MATRIX),
            deterministic_split=deterministic_split,
            rng=rng
        )

        for cluster_id in range(n_clusters):

            for task_id in range(n_tasks_per_cluster):

                task_dir = \
                    os.path.join(os.getcwd(), save_dir, f"task_{str(task_id + cluster_id * n_tasks_per_cluster)}")

                save_data(
                    save_dir=task_dir,
                    train_data=train_data[cluster_id][task_id],
                    train_targets=train_targets[cluster_id][task_id],
                    test_data=test_data[cluster_id][task_id],
                    test_targets=test_targets[cluster_id][task_id]
                )

                availability_id = tasks_types[task_id + cluster_id * n_tasks_per_cluster][0]
                stability_id = tasks_types[task_id + cluster_id * n_tasks_per_cluster][1]

                availability = \
                    compute_availability(
                        availability_type=AVAILABILITY_TYPES[availability_id],
                        availability_parameter=availability_parameters[cluster_id]
                    )

                stability = \
                    compute_stability(
                        stability_type=STABILITY_TYPES[stability_id],
                        stability_parameter=stability_parameters[cluster_id],
                        availability=availability,
                        rng=rng
                    )

                all_tasks_cfg[int(task_id + cluster_id * n_tasks_per_cluster)] = {
                    "cluster_id": cluster_id,
                    "task_dir": task_dir,
                    "availability_type": AVAILABILITY_TYPES[availability_id],
                    "availability": availability,
                    "stability_type": STABILITY_TYPES[stability_id],
                    "stability": stability
                }

    else:
        raise NotImplementedError(f"n_clusters={n_clusters}. For the moment we only support n_clusters==2!")

    return all_tasks_cfg


def generate_data(
        dataset,
        split_type,
        n_classes,
        n_train_samples,
        n_tasks,
        n_clusters,
        n_swapping_pairs,
        availability_parameters,
        stability_parameters,
        save_dir,
        rng=None
):
    if split_type == "incongruent_split":
        all_tasks_cfg = incongruent_split(
            dataset=dataset,
            n_classes=n_classes,
            n_train_samples=n_train_samples,
            n_tasks=n_tasks,
            n_clusters=n_clusters,
            n_swapping_pairs=n_swapping_pairs,
            availability_parameters=availability_parameters,
            stability_parameters=stability_parameters,
            save_dir=save_dir,
            rng=rng
        )
    else:
        error_message = "only `incongruent_split` is available for the moment!!" \
                        "Please pass '--incongruent_split'"

        raise NotImplementedError(error_message)

    return all_tasks_cfg


def generate_tasks_types(
        num_tasks,
        joint_probability,
        deterministic_split,
        rng
):
    """Generate tasks_types types

    The tasks_types has an `availability` type, and a `stability` type;

    The `availability` and `stability` types are sampled according to `JOINT_PROBABILITY_MATRIX`.

    The result is given as a `numpy.array` of shape `(num_clients, 2)`. The columns
    correspond to the availability type, and stability type, respectively.

    Parameters
    ----------
    num_tasks: `int`
        number of tasks_types

    joint_probability: 2-D `numpy.array`
        every entry represents the probability of an arrival process and a capacity; should sum-up to 1

    deterministic_split: bool
        split tasks per task type in a deterministic or stochastic way, according to JOINT_PROBABILITY_MATRIX

    rng: `numpy.random._generator.Generator`

    Returns
    -------
        * `numpy.array` of shape `(num_clients, 2)`

    """
    assert np.abs(joint_probability.sum() - 1) < ERROR, "`joint_probability` should sum-up to 1!"

    tasks_indices = rng.permutation(num_tasks)
    count_per_cluster = compute_count_per_cluster(num_tasks, joint_probability, deterministic_split, rng)
    indices_per_cluster = np.split(tasks_indices, np.cumsum(count_per_cluster[:-1]))
    indices_per_cluster = np.array(indices_per_cluster, dtype=object).reshape(joint_probability.shape)

    tasks_types = np.zeros((num_tasks, 2), dtype=np.int8)

    for availability_type_idx in range(joint_probability.shape[0]):
        indices = np.concatenate(indices_per_cluster[availability_type_idx])
        tasks_types[indices, 0] = availability_type_idx

    for stability_idx in range(joint_probability.shape[1]):
        indices = np.concatenate(indices_per_cluster[:, stability_idx])
        tasks_types[indices, 1] = stability_idx

    return tasks_types


def compute_count_per_cluster(num_tasks, joint_probability, deterministic_split, rng):
    """Split tasks into tasks types according to `JOINT_PROBABILITY_MATRIX`.

    If the split is deterministic, the number of tasks per task type is computed proportionally to joint_probability.

    Otherwise, the number of tasks per task type is computed from a multinomial with joint_probability.

    The result is given as a `numpy.array` of shape `(joint_probability.flatten(), )`. The elements
    correspond to the number of tasks for each task type.

    Parameters
    ----------
    num_tasks: `int`
        number of tasks_types

    joint_probability: 2-D `numpy.array`
        every entry represents the probability of an arrival process and a capacity; should sum-up to 1

    deterministic_split: bool
        split tasks per task type in a deterministic or stochastic way, according to JOINT_PROBABILITY_MATRIX

    rng: `numpy.random._generator.Generator`

    Returns
    -------
        * `numpy.array` of shape `(joint_probability.flatten(), )`

    """
    if deterministic_split:
        count_per_task_type = (num_tasks * joint_probability.flatten()).astype(int)
        remaining_clients = num_tasks - count_per_task_type.sum()
        for task_type in range(count_per_task_type.shape[0]):
            if (remaining_clients > 0) and (count_per_task_type[task_type] != 0):
                count_per_task_type[task_type] += 1
                remaining_clients -= 1
        print(f"==> Deterministic split: {count_per_task_type}")
        return count_per_task_type
    else:
        return rng.multinomial(num_tasks, joint_probability.flatten())


def compute_samples_per_user(n_tasks, tasks_types, rng, augmentation=False, n_samples=7000, proportion=10):
    """

    Parameters
    ----------
    augmentation
    n_tasks
    n_samples
    tasks_types
    proportion
    rng

    Returns
    -------

    """
    if augmentation:
        print("==> Generating samples_per_user deterministically, more available clients receive more samples")
        availability_ids = [tasks_types[i][0] for i in range(n_tasks)]
        n_unavailable_users = sum(availability_ids)
        samples_per_user = [n_samples / (proportion * n_tasks + (1 - proportion) * n_unavailable_users)] * n_tasks
        for client_id in range(n_tasks):
            if availability_ids[client_id] == 0:
                samples_per_user[client_id] *= proportion
    else:
        print("==> Generating samples_per_user from a log-normal distribution")
        samples_per_user = rng.lognormal(4, 2, n_tasks).astype(int) + 50
    return [int(i) for i in samples_per_user]


def compute_stability(stability_type, stability_parameter, availability, rng=None):
    """ compute stability value for given stability type and parameter, and the value of availability

    Parameters
    ----------
    stability_type: str

    stability_parameter: float

    availability: float
        the value of the availability

    rng: random number generator; default is None

    Returns
    -------
        * float:

    """
    if not (0. <= stability_parameter <= 1.):
        warnings.warn("stability_parameter is automatically clipped to the interval (0, 1)")
        stability_parameter = np.clip(stability_parameter, a_min=0, a_max=1)

    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    if stability_type == "stable":
        stability = np.max([
            stability_parameter,
            1 - 1 / availability + ERROR,
            1 - 1 / (1 - availability) + ERROR
        ])

    elif stability_type == "shifting":
        # TODO: add specific argument
        eps = 0.01

        stability = np.clip(
            0.5 * (rng.normal(loc=0.0, scale=eps) + eps),
            a_min=np.max([-1 + ERROR, 1 - 1 / availability + ERROR, 1 - 1 / (1 - availability)]) + ERROR,
            a_max=1 - ERROR
        )

    elif stability_type == "unstable":
        stability = np.max([
            -stability_parameter,
            1 - 1 / availability + ERROR,
            1 - 1 / (1 - availability) + ERROR
        ])

    else:
        error_message = f"{stability_type} is not a possible stability_type; possible are:"
        for t in STABILITY_TYPES:
            error_message += f"\"{t}\", "

        raise NotImplementedError(error_message)

    return stability


def compute_availability(availability_type, availability_parameter):
    """ compute stability value for given stability type and parameter

    Parameters
    ----------
    availability_type: str

    availability_parameter: float

    Returns
    -------
        * float:

    """
    if not (-0.5 + ERROR <= availability_parameter <= 0.5 - ERROR):
        warnings.warn("availability_parameter is automatically clipped to the interval (-1/2, 1/2)")
        availability_parameter = np.clip(availability_parameter, a_min=-0.5 + ERROR, a_max=0.5 - ERROR)

    if availability_type == "available":
        availability = 1 / 2 + availability_parameter

    elif availability_type == "unavailable":
        availability = 1 / 2 - availability_parameter

    else:
        error_message = ""
        raise NotImplementedError(error_message)

    return availability
