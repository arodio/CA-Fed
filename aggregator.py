import time
import random

from abc import ABC, abstractmethod

import numpy as np

from utils.torch_utils import *
from utils.constants import *

from tqdm import tqdm

from torch.utils.data import DataLoader


class Aggregator(ABC):
    r"""Base class for Aggregator.

    `Aggregator` dictates communications between clients_dict

    Attributes
    ----------
    clients_dict: Dict[int: Client]

    clients_weights_dict: Dict[int: Client]

    global_trainer: List[Trainer]

    n_clients:

    model_dim: dimension if the used model

    c_round: index of the current communication round

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    logger: SummaryWriter

    rng: random number generator

    np_rng: numpy random number generator

    Methods
    ----------
    __init__

    mix

    update_clients

    write_logs

    save_state

    load_state

    """
    def __init__(
            self,
            clients_dict,
            clients_weights_dict,
            global_trainer,
            logger,
            verbose=0,
            seed=None,
    ):
        """

        Parameters
        ----------
        clients_dict: Dict[int: Client]

        clients_weights_dict: Dict[int: Client]

        global_trainer: Trainer

        logger: SummaryWriter

        verbose: int

        seed: int

        """
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        self.clients_dict = clients_dict
        self.clients_weights_dict = clients_weights_dict

        self.global_trainer = global_trainer
        self.device = self.global_trainer.device

        self.verbose = verbose
        self.logger = logger

        self.model_dim = self.global_trainer.model_dim

        self.n_clients = len(clients_dict)

        self.c_round = 0

    @abstractmethod
    def mix(self, sampled_clients_ids, sampled_clients_weights):
        """mix sampled clients according to weights

        Parameters
        ----------
        sampled_clients_ids:

        sampled_clients_weights:


        Returns
        -------
            None
        """
        pass

    @abstractmethod
    def update_clients(self):
        pass

    def write_logs(self):
        global_train_loss = 0.
        global_train_metric = 0.
        global_test_loss = 0.
        global_test_metric = 0.

        for client_id, client in self.clients_dict.items():

            train_loss, train_metric, test_loss, test_metric = client.write_logs(counter=self.c_round)

            if self.verbose > 1:
                # TODO: implement mechanism to redirect writing (see https://github.com/tqdm/tqdm#redirecting-writing)
                tqdm.write("*" * 30)
                tqdm.write(f"Client {client_id}..")

                tqdm.write(f"Train Loss: {train_loss:.3f} | Train Metric: {train_metric :.3f}|", end="")
                tqdm.write(f"Test Loss: {test_loss:.3f} | Test Metric: {test_metric:.3f} |")

                tqdm.write("*" * 30)

            global_train_loss += self.clients_weights_dict[client_id] * train_loss
            global_train_metric += self.clients_weights_dict[client_id] * train_metric
            global_test_loss += self.clients_weights_dict[client_id] * test_loss
            global_test_metric += self.clients_weights_dict[client_id] * test_metric

        if self.verbose > 0:
            # TODO: implement mechanism to redirect writing (see https://github.com/tqdm/tqdm#redirecting-writing)
            tqdm.write("+" * 50)
            tqdm.write("Global..")
            tqdm.write(f"Train Loss: {global_train_loss:.3f} | Train Metric: {global_train_metric:.3f} |", end="")
            tqdm.write(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_metric:.3f} |")
            tqdm.write("+" * 50)

        self.logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
        self.logger.add_scalar("Train/Metric", global_train_metric, self.c_round)
        self.logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
        self.logger.add_scalar("Test/Metric", global_test_metric, self.c_round)
        self.logger.flush()

    def gather_loss_dict(self):
        """gathers the losses for every client

        Parameters
        ----------

        Returns
        -------
            * Dict: key is client_id and value is the corresponding loss value

        """

        loss_dict = dict()

        for client_id, client in self.clients_dict.items():
            loss_dict[client_id], _ = client.trainer.evaluate_loader(client.val_loader)

        return loss_dict


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """
    def mix(self, sampled_clients_ids, sampled_clients_weights):

        for idx in sampled_clients_ids:
            self.clients_dict[idx].step()

        self.c_round += 1

    def update_clients(self):
        pass


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.

     Clients get fully synchronized with the average client.

    """
    def mix(self, sampled_clients_ids, sampled_clients_weights):

        if len(sampled_clients_weights) == 0:
            print(f"No clients are sampled at round {self.c_round}")
            self.c_round += 1
            return

        sampled_clients_weights = torch.tensor(sampled_clients_weights, dtype=torch.float32)

        for idx, weight in zip(sampled_clients_ids, sampled_clients_weights):
            if weight <= ERROR:
                # clients with weights set to zero do not need to perform a local update
                # this is done to optimize the run time
                pass
            else:
                self.clients_dict[idx].step()

        trainers_deltas = [self.clients_dict[idx].trainer - self.global_trainer for idx in sampled_clients_ids]

        self.global_trainer.optimizer.zero_grad()

        average_models(
            trainers_deltas,
            target_trainer=self.global_trainer,
            weights=sampled_clients_weights,
            average_params=False,
            average_gradients=True
        )

        self.global_trainer.optimizer.step()

        # assign the updated model to all clients_dict
        self.update_clients()

        self.c_round += 1

    def update_clients(self):
        for client_id, client in self.clients_dict.items():

            copy_model(client.trainer.model, self.global_trainer.model)
