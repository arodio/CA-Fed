"""Downloads a dataset and generates configuration file for federated simulation

Split a classification dataset, e.g., `MNIST` or `CIFAR-10`, among `n_clients`.

Two splitting strategies are available: `incongruent_split` and `iid_split`



### Incongruent split

If `incongruent_split` is selected, the dataset is split among `n_clients`,
 each of which belonging to one of two groups, as follows:

    1) the dataset is randomly shuffled and partitioned among `n_clients`
    2) clients_dict are randomly partitioned into `n_clusters`
    3) the data of clients_dict from the second group is  modified
       by  randomly  swapping out `k` pairs of labels

    Similar to
        "Clustered Federated Learning: Model-Agnostic Distributed
        Multi-Task Optimization under Privacy Constraints"
        __(https://arxiv.org/abs/1910.01991)

If  'iid_split'  is selected, the  dataset is split in an IID fashion.

Default usage is ''iid_split'

"""
import argparse

from utils import *
from constants import *


def parse_arguments(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset_name",
        help="name of dataset to use, possible are {'synthetic_clustered', 'synthetic_leaf', 'mnist', 'cifar10'}",
        required=True,
        type=str
    )
    parser.add_argument(
        "--n_tasks",
        help="number of tasks/clients_dict",
        required=True,
        type=int
    )
    parser.add_argument(
        "--n_clusters",
        help="number of clusters/groups; default is 2, only used when experiment_name=='synthetic_clustered'",
        type=int,
        default=2
    )
    parser.add_argument(
        "--n_classes",
        help="number of classes, only used when experiment_name=='synthetic_leaf';"
             "default=10",
        default=10,
        type=int
    )
    parser.add_argument(
        "--dimension",
        help="dimension of the data, only used when experiment_name=='synthetic';"
             "default=10 if experiment_name=='synthetic_clustered', default=60 if experiment_name=='synthetic_leaf'",
        default=60,
        type=int
    )
    parser.add_argument(
        "--n_train_samples",
        help="number of train samples, only used when experiment_name=='synthetic_clustered';"
             "default=150",
        default=150,
        type=int
    )
    parser.add_argument(
        "--n_test_samples",
        help="number of test samples, only used when experiment_name=='synthetic_clustered';"
             "default=500",
        default=500,
        type=int
    )
    parser.add_argument(
        "--alpha",
        help="parameter controlling how much local models differ from each other;" 
             "only used when experiment_name=='synthetic_leaf';"
             "expected to be in the range (0,1);"
             "default=1.0",
        default=1.0,
        type=float
    )
    parser.add_argument(
        "--beta",
        help="parameter controlling how much the local data at each device differs from that of other devices;" 
             "only used when experiment_name=='synthetic_leaf';"
             "expected to be in the range (0,1);"
             "default=1.0",
        default=1.0,
        type=float
    )
    parser.add_argument(
        "--iid",
        help="if selected, data are split iid, only used when experiment_name=='synthetic_leaf'",
        action='store_true'
    )
    parser.add_argument(
        "--hetero_param",
        help="parameter controlling clients dissimilarity, only used when experiment_name=='synthetic_clustered';"
             "expected to be in the range (0,1);"
             "default=1.0",
        default=1.0,
        type=float
    )
    parser.add_argument(
        "--deterministic_split",
        help="if selected, tasks are assigned to tasks types proportionally to joint_probability_matrix",
        action='store_true'
    )
    parser.add_argument(
        "--augmentation",
        help="if selected, more samples are assigned to the more available clients,"
             "only used when experiment_name=='synthetic_leaf'",
        action='store_true'
    )
    parser.add_argument(
        "--incongruent_split",
        help="if selected, the incongruent split described above is used, only used when experiment_name=='mnist'",
        action='store_true'
    )
    parser.add_argument(
        "--n_swapping_pairs",
        help="number of pairs to swap for each cluster/group; default is 1, only used when experiment_name=='mnist'",
        type=int,
        default=1
    )
    parser.add_argument(
        "--availability_parameters",
        nargs="+",
        help="parameters controlling the asymptotic availability of the clients_dict from each group/cluster;"
             "should be a list of the same size as `n_clusters`;"
             "default is [0., 0.25]",
        type=float,
        default=[0.0, 0.25]
    )
    parser.add_argument(
        "--stability_parameters",
        nargs="+",
        help="list of parameters controlling the stability of clients_dict from each group/cluster;"
             "should be a list of the same size as `n_clusters`;"
             "default is [0.1, 0.25]",
        type=float,
        default=[0.1, 0.25]
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help='path of the directory to save data and configuration;'
             'the directory will be created if not already created;'
             'if not specified the data is saved to "./{dataset_name}";',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--seed',
        help='seed for the random number generator;'
             'if not specified the system clock is used to generate the seed;',
        type=int,
        default=argparse.SUPPRESS,
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args_ = parse_arguments()

    seed = (args_.seed if (("seed" in args_) and (args_.seed >= 0)) else int(time.time()))
    rng = np.random.default_rng(seed)

    if "save_dir" in args_:
        save_dir = args_.save_dir
    else:
        save_dir = os.path.join(".", args_.dataset_name)
        warnings.warn(f"'--save_dir' is not specified, results are saved to {save_dir}!", RuntimeWarning)

    os.makedirs(save_dir, exist_ok=True)

    if args_.dataset_name == "synthetic_leaf":
        all_tasks_cfg = generate_synthetic_leaf(
            n_tasks=args_.n_tasks,
            n_classes=args_.n_classes,
            dimension=args_.dimension,
            alpha=args_.alpha,
            beta=args_.beta,
            iid=args_.iid,
            availability_parameters=args_.availability_parameters,
            stability_parameters=args_.stability_parameters,
            deterministic_split=args_.deterministic_split,
            augmentation=args_.augmentation,
            save_dir=os.path.join(save_dir, "all_tasks"),
            rng=rng
        )
    elif args_.dataset_name == "synthetic_clustered":
        all_tasks_cfg = generate_synthetic_clustered(
            n_tasks=args_.n_tasks,
            n_clusters=args_.n_clusters,
            dimension=args_.dimension,
            n_train_samples=args_.n_train_samples,
            n_test_samples=args_.n_test_samples,
            hetero_param=args_.hetero_param,
            availability_parameters=args_.availability_parameters,
            stability_parameters=args_.stability_parameters,
            deterministic_split=args_.deterministic_split,
            save_dir=os.path.join(save_dir, "all_tasks"),
            rng=rng
        )
    elif args_.dataset_name == "mnist" or args_.dataset_name == "cifar10":
        if args_.incongruent_split:
            split_type = "incongruent_split"
        else:
            warnings.warn("split type is automatically set to 'iid'")
            split_type = "iid"

        dataset = get_dataset(
            dataset_name=args_.dataset_name,
            raw_data_path=os.path.join(save_dir, "raw_data")
        )

        all_tasks_cfg = generate_data(
            split_type=split_type,
            dataset=dataset,
            n_classes=N_CLASSES[args_.dataset_name],
            n_train_samples=N_TRAIN_SAMPLES[args_.dataset_name],
            n_tasks=args_.n_tasks,
            n_clusters=args_.n_clusters,
            n_swapping_pairs=args_.n_swapping_pairs,
            availability_parameters=args_.availability_parameters,
            stability_parameters=args_.stability_parameters,
            save_dir=os.path.join(save_dir, "all_tasks"),
            rng=rng
        )
    else:
        error_message = f"{args_.dataset_name} is not available, possible datasets are:"
        for n in DATASETS:
            error_message += f" {n},"
        error_message = error_message[:-1]

        raise NotImplementedError(error_message)

    save_cfg(save_path=os.path.join(save_dir, "cfg.json"), cfg=all_tasks_cfg)
