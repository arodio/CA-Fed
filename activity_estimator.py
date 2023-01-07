import time

import numpy as np

from abc import ABC, abstractmethod


class ActivityEstimator(ABC):
    r"""Estimates clients_dict activity

    Estimates the availabilities and stabilities parameters from the activity of clients


    Attributes
    ----------
    clients_ids: 1-D numpy.array (dtype=int)

    n_clients: int

    __rng: numpy.random._generator.Generator

    Methods
    -------
    step

    gather_activity_estimates

    """

    def __init__(self, clients_ids, availability_types, availabilities, stability_types, stabilities, rng=None):
        """

        Parameters
        ----------
        clients_ids: 1-D numpy.array (dtype=int)

        availability_types: : 1-D list (dtype=str)

        availabilities: 1-D numpy.array (dtype=float)

        stability_types: : 1-D list (dtype=str)

        stabilities: 1-D numpy.array (dtype=float)

        rng: numpy.random._generator.Generator

        """
        self.clients_ids = clients_ids
        self.n_clients = len(self.clients_ids)

        self.availability_types = availability_types
        self.availabilities = availabilities

        self.stability_types = stability_types
        self.stabilities = stabilities

        self._time_step = 1

        self.__rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    def step(self):
        self._time_step += 1

    @abstractmethod
    def gather_activity_estimates(self, active_clients):
        """returns tuple of dictionaries about the estimates of the clients' activities

        Returns
        -------
            * Tuple[dict, dict, dict, dict]:
                availability_types_dict, availability_dict, stability_types_dict, stability_dict
        """
        pass


class OracleActivityEstimator(ActivityEstimator):
    """all algorithms have access to an oracle providing the true activity parameters for each client

    """

    def gather_activity_estimates(self, active_clients):
        """returns tuple of dictionaries about the estimates of the clients' activities

        Parameters

        availability_types_dict: Dict[int: str]
            maps clients ids to their availability types, either "available" or "unavailable"

        availability_dict: Dict[int: float]
            maps clients ids to their stationary participation probability

        stability_types_dict: Dict[int: str]
            maps clients ids to their stability types, either "unstable" or "stable"

        stability_dict: Dict[int: float]
            maps clients ids to the spectral gap of their corresponding markov chains

        Returns
        -------
            * Tuple[dict, dict, dict, dict]:
                availability_types_dict, availability_dict, stability_types_dict, stability_dict
        """
        availability_types_dict = dict()
        availability_dict = dict()
        stability_types_dict = dict()
        stability_dict = dict()

        for idx, client_id in enumerate(self.clients_ids):
            availability_types_dict[int(client_id)] = str(self.availability_types[idx])
            availability_dict[int(client_id)] = float(self.availabilities[idx])
            stability_types_dict[int(client_id)] = str(self.stability_types[idx])
            stability_dict[int(client_id)] = float(self.stabilities[idx])

        return availability_types_dict, availability_dict, stability_types_dict, stability_dict


class BayesianActivityEstimator(ActivityEstimator):
    """estimates the activity parameters of clients using a Bayesian estimator with beta prior

    Attributes
    ----------

    _participations_vec: 1-D numpy.array
        entry at position client_id, is the total number of steps where client_id was active
        in the training process

    _participations_prior:  1-D numpy.array
        entry at position client_id, is the total number of steps where client_id was active
        in the prior observations

    _participations_rate: 1-D numpy.array
        entry at position client_id, is the total number of steps where client_id was active in the training process
        and in the prior observations divided by the lenght of the training process and the prior observations

    _correlations_actives_vec: 1-D numpy.array
        entry at position client_id, is the total number of transitions from state active to state active
        where client_id was involved in the training process

    _correlations_actives_prior: 1-D numpy.array
        entry at position client_id, is the total number of transitions from state active to state active
        where client_id was involved in the prior observations

    _correlations_actives_rate: 1-D numpy.array
        entry at position client_id, is the total number of transitions from state active to state active
        where client_id was involved in the training process and in the prior observations
        divided by the lenght of the training process and the prior observations

    _correlations_inactives_vec: 1-D numpy.array
        entry at position client_id, is the total number of transitions from state inactive to state inactive
        where client_id was involved in the training process

    _correlations_inactives_prior: 1-D numpy.array
        entry at position client_id, is the total number of transitions from state inactive to state inactive
        where client_id was involved in the prior observations

    _correlations_inactives_rate: 1-D numpy.array
        entry at position client_id, is the total number of transitions from state inactive to state inactive
        where client_id was involved in the training process and in the prior observations
        divided by the lenght of the training process and the prior observations

    _correlations_rate: 1-D numpy.array
        entry at position client_id, is the second largest eigenvalue of the estimated transition probability matrix
        where client_id was involved in the training process and in the prior observations
        computed as _correlations_actives_rate + _correlations_inactives_rate - 1

    _time_horizon_prior: int
        duration of the beta prior
    """

    def __init__(
            self,
            clients_ids,
            availability_types,
            availabilities,
            stability_types,
            stabilities,
            rng=None
    ):
        super(BayesianActivityEstimator, self).__init__(
            clients_ids=clients_ids,
            availability_types=availability_types,
            availabilities=availabilities,
            stability_types=stability_types,
            stabilities=stabilities,
            rng=rng
        )

        self._active_clients = list()

        self._time_horizon_prior = 100

        self._participations_vec = np.zeros(self.n_clients, dtype=int)
        self._participations_prior = self._init_participations_prior()
        self._participations_rate = np.zeros(self.n_clients, dtype=np.float32)

        self._correlations_actives_vec = np.zeros(self.n_clients, dtype=int)
        self._correlations_inactives_vec = np.zeros(self.n_clients, dtype=int)
        self._correlations_actives_prior,  self._correlations_inactives_prior = self._init_correlations_prior()
        self._correlations_actives_rate = np.zeros(self.n_clients, dtype=np.float32)
        self._correlations_inactives_rate = np.zeros(self.n_clients, dtype=np.float32)
        self._correlations_rate = np.zeros(self.n_clients, dtype=np.float32)

    def _init_participations_prior(self):
        """

        Returns
        -------
            * 1-D numpy.array
        """
        _participations_prior = self.availabilities * self._time_horizon_prior

        return _participations_prior

    def _init_correlations_prior(self):
        """

        Returns
        -------
            * 1-D numpy.array
            * 1-D numpy.array
        """
        _correlations_actives_prior = (self.availabilities + self.stabilities - self.stabilities * self.availabilities)
        _correlations_inactives_prior = (1 - self.availabilities + self.stabilities * self.availabilities)
        _correlations_actives_prior *= self._time_horizon_prior
        _correlations_inactives_prior *= self._time_horizon_prior

        return _correlations_actives_prior, _correlations_inactives_prior

    def gather_activity_estimates(self, active_clients):

        for client_id in active_clients:
            self._participations_vec[client_id] += 1

        self._participations_rate = self._participations_vec + self._participations_prior
        self._participations_rate /= (self._time_step + self._time_horizon_prior)

        active_clients_old = np.copy(self._active_clients)
        self._active_clients = active_clients
        active_clients_new = np.copy(self._active_clients)

        for client_id in range(self.n_clients):
            if (client_id in active_clients_old) and (client_id in active_clients_new):
                self._correlations_actives_vec[client_id] += 1
            elif (client_id not in active_clients_old) and (client_id not in active_clients_new):
                self._correlations_inactives_vec[client_id] += 1

        self._correlations_actives_rate = self._correlations_actives_vec + self._correlations_actives_prior
        self._correlations_inactives_rate = self._correlations_inactives_vec + self._correlations_inactives_prior
        self._correlations_actives_rate /= (self._time_step + self._time_horizon_prior)
        self._correlations_inactives_rate /= (self._time_step + self._time_horizon_prior)

        self._correlations_rate = self._correlations_actives_rate + self._correlations_inactives_rate - 1

        availability_types_dict = dict()
        availability_dict = dict()
        stability_types_dict = dict()
        stability_dict = dict()

        for idx, client_id in enumerate(self.clients_ids):
            availability_types_dict[int(client_id)] = str(self.availability_types[idx])
            availability_dict[int(client_id)] = float(self._participations_rate[idx])
            stability_types_dict[int(client_id)] = str(self.stability_types[idx])
            stability_dict[int(client_id)] = float(self._correlations_rate[idx])

        return availability_types_dict, availability_dict, stability_types_dict, stability_dict
