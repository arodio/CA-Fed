# Simulating FedAvg

We provide an example for simulating a federated training using
FedAvg. We suppose that we have access to `train_loaders`, `val_loaders`
and `test_loaders` each given as a list of `torch.utils.data.DataLoader`
objects. We can use `utils.py/get_loaders`, for example, to generate
the data loaders:

```
from utils.utils import get_loaders

train_iterators, val_iterators, test_iterators = \
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            data_dir=data_dir,
            batch_size=args_.bz,
            is_validation=args_.validation
        )
      
```

In addition to the data loaders, a client needs a `Trainer` object
to be initialized. The `Trainer` object takes care of training and
evaluating a machine learning model. One can use
`utils.py/get_trainer`, for example, to initialize a `trainer` for every client.
We can generate the client's dictionary as follows:

```
from utils.utils import init_clients
from utils.utils import get_clients_weights


clients_dict = dict()

print("\n==> Initialize Clients..")
for client_id in tqdm(all_clients_cfg.keys(), position=0, leave=True):

    data_dir = all_clients_cfg[client_id]["task_dir"]

    logs_dir = os.path.join(args_.logs_dir, f"client_{client_id}")
    os.makedirs(logs_dir, exist_ok=True)
    logger = SummaryWriter(logs_dir)

    clients_dict[int(client_id)] = init_client(
            args=args_,
            client_id=client_id,
            data_dir=data_dir,
            logger=logger
        )
 
clients_weights_dict = get_clients_weights(
     objective_type=args_.objective_type,
     n_samples_per_client=n_samples_per_client
)
```

The clients' availability dynamic is generated by the `ActivitySimulator`
and estimated by the `ActivityEstimator`. We use the functions `utils.py/get_activity_simulator`
and `utils.py/get_activity_estimator` to instantiate such objects.
Finally, the `client_sampler` can be initialized as follows:

```
from utils.utils import get_activity_simulator
from utils.utils import get_activity_estimator
from clients_sampler import UnbiasedClientsSampler


activity_simulator_rng_ = np.random.default_rng(args_.seed)
activity_simulator_ = \
    get_activity_simulator(
        all_clients_cfg=all_clients_cfg, 
        rng=activity_simulator_rng_
    )

activity_estimator_rng_ = np.random.default_rng(args_.seed)
acvitity_estimator_ = \
    get_activity_estimator(
        estimator_type=args_.estimator_type,
        all_clients_cfg=all_clients_cfg,
        rng=activity_estimator_rng_
    )

clients_sampler_rng_ = np.random.default_rng(args_.seed)
clients_sampler_ = \
    UnbiasedClientsSampler(
            activity_simulator=activity_simulator_,
            activity_estimator=activity_estimator_,
            clients_weights_dict=clients_weights_dict,
            rng=clients_sampler_rng_
        )

```


Finally, the aggregator can be initialized as follows:

```
from aggregator import CentralizedAggregator


global_trainer = \
    get_trainer(
        experiment_name=args_.experiment,
        device=args_.device,
        optimizer_name=args_.server_optimizer,
        lr=args_.server_lr,
        seed=args_.seed
    )

global_logs_dir = os.path.join(args_.logs_dir, "global")
os.makedirs(global_logs_dir, exist_ok=True)
global_logger = SummaryWriter(global_logs_dir)

aggregator_ = \
   CentralizedAggregator(
        clients_dict=clients_dict,
        clients_weights_dict=clients_weights_dict,
        global_trainer=global_trainer,
        logger=global_logger,
        verbose=args_.seed,
        seed=args_.seed,
    )

```

The main training loop is as follows:

```
for ii in range(args_.n_rounds):

    active_clients = clients_sampler.get_active_clients()
    
    sampled_clients_ids, sampled_clients_weights = \
            clients_sampler.sample(active_clients=active_clients, loss_dict=None)
    
    aggregator.mix(sampled_clients_ids, sampled_clients_weights)

    if (ii % args_.log_freq) == (args_.log_freq - 1):
        aggregator.save_state(chkpts_dir)
        aggregator.write_logs()
        
```
