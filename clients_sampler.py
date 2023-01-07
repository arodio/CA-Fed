import time
import json

from abc import ABC, abstractmethod

from utils.constants import *
from utils.divergence import *


class ClientsSampler(ABC):
    r"""Base class for clients_dict sampler

    Attributes
    ----------
    activity_simulator: ActivitySimulator

    clients_ids:

    n_clients: int

    clients_weights_dict: Dict[int: float]
        maps clients ids to their corresponding weight/importance in the true objective function

    _availability_dict: Dict[int: float]
        maps clients ids to their stationary participation probability

    _stability_dict: Dict[int: float]
        maps clients ids to the spectral gap of their corresponding markov chains

    history: Dict[int: Dict[str: List]]
        stores the active and sampled clients and their weights at every time step

    _time_step: int
        tracks the number of steps

    rng:

    Methods
    ----------
    __init__

    _update_estimates

    sample_clients

    step

    save_history

    """

    def __init__(
            self,
            activity_simulator,
            activity_estimator,
            clients_weights_dict,
            rng=None,
            *args,
            **kwargs
    ):
        """

        Parameters
        ----------
        activity_simulator: ActivitySimulator

        activity_estimator: ActivityEstimator

        clients_weights_dict: Dict[int: float]

        rng:

        """

        self.activity_simulator = activity_simulator
        self.activity_estimator = activity_estimator

        self.clients_ids = list(clients_weights_dict.keys())
        self.n_clients = len(self.clients_ids)

        self.clients_weights_dict = clients_weights_dict

        self._availability_types_dict = dict()
        self._availability_dict = dict()
        self._stability_types_dict = dict()
        self._stability_dict = dict()

        self.history = dict()

        self._time_step = -1

        if rng is None:
            seed = int(time.time())
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def get_active_clients(self):
        """receive the list of active clients

        Returns
        -------
            * List[int]
        """
        return self.activity_simulator.get_active_clients()

    def _update_estimates(self, active_clients, loss_dict=None):
        """update the estimate of clients' activities

        Returns
        -------
            * Tuple[dict, dict, dict, dict]:
                availability_types_dict, availability_dict, stability_types_dict, stability_dict

        """
        self._availability_types_dict, self._availability_dict, _, _ = \
            self.activity_estimator.gather_activity_estimates(active_clients)

    def step(self, active_clients, sampled_clients_ids, sampled_clients_weights):
        """update the internal state of the clients sampler

        Parameters
        ----------
        active_clients: List[int]

        sampled_clients_ids: List[int]

        sampled_clients_weights: Dict[int: float]


        Returns
        -------
            None
        """
        self.activity_simulator.step()
        self.activity_estimator.step()
        self._time_step += 1

        current_state = {
            "active_clients": active_clients,
            "sampled_clients_ids": sampled_clients_ids,
            "sampled_clients_weights": sampled_clients_weights
        }

        self.history[self._time_step] = current_state

    def save_history(self, json_path):
        """save history and clients metadata

        save a dictionary with:
            * history: stores the active and sampled clients and their weights at every time step
            * clients_ids: list of clients ids stored as integers
            * clients_true_weights: dictionary mapping clients ids to their true weights
            * clients_availability_types: dictionary mapping clients ids to their availability types
            * clients_true_availability: dictionary mapping clients ids to their true availabilities
            * clients_stability_types: dictionary mapping clients ids to their stability types
            * clients_true_stability: dictionary mapping clients ids to their true stabilities

        Parameters
        ----------
        json_path: path of a .json file

        Returns
        -------
            None
        """
        metadata = {
            "history": self.history,
            "clients_ids": self.clients_ids,
            "clients_true_weights": self.clients_weights_dict,
            "clients_availability_types": self._availability_types_dict,
            "clients_true_availability": self._availability_dict,
            "clients_stability_types": self._stability_types_dict,
            "clients_true_stability": self._stability_dict
        }

        with open(json_path, "w") as f:
            json.dump(metadata, f)

    @abstractmethod
    def sample(self, active_clients, loss_dict):
        """sample clients_dict

        Parameters
        ----------
        active_clients: List[int]
        loss_dict: Dict[int: float] or None

        Returns
        -------
            * List[int]: indices of the sampled clients_dict
            * List[float]: weights to be associated to the sampled clients_dict
        """
        pass


class UnbiasedClientsSampler(ClientsSampler):
    """
    Samples all active clients with aggregation weight inversely proportional to their participation
    """

    def sample(self, active_clients, loss_dict=None):
        """implementation of the abstract method ClientSampler.sample for the UnbiasedClientSampler

        Parameters
        ----------
        active_clients: List[int]
        loss_dict: Dict[int: float] or None

        Returns
        -------
            * List[int]: indices of the sampled clients_dict
            * List[float]: weights to be associated to the sampled clients_dict
        """

        sampled_clients_ids, sampled_clients_weights = [], []

        self._update_estimates(active_clients)

        for client_id in active_clients:
            sampled_clients_ids.append(client_id)

            sampled_clients_weights.append(
                self.clients_weights_dict[client_id] / self._availability_dict[client_id]
            )

        self.step(active_clients, sampled_clients_ids, sampled_clients_weights)

        return sampled_clients_ids, sampled_clients_weights


class MarkovianClientsSampler(ClientsSampler):
    """Markovian clients sampler, also known as CA-Fed (Correlation-Aware Federated Learning)

    Considers the availability and stability of clients

    Attributes
    ----------
    _smoothness_param: float
        parameter used for the estimation of the loss vector

    _tolerance: float
        tolerance used for the stopping criteria

    _bias_const: float
        coefficient of the bias error, controls the number of excluded clients

    _availability_vec: 1-D numpy.array
        entry at position client_id, gives the availability value of client_id

    _clients_weights_vec: 1-D numpy.array
        entry at position client_id, gives the weight of client_id in the true objective function

    _clients_optimums_vec: 1-D numpy.array
        entry at position client_id, gives the optimum loss value of client_id

    _clients_ids_by_stability: List[int]
        clients ids ordered by the absolute value of their stability

    _clients_ids_by_availability: List[int]
        clients ids ordered by the value of their availability

    _loss_estimates_vec: 1-D numpy.array
        entry at position client_id, gives the estimated loss value of client_id

    _allocation_vec: 1-D numpy.array
        entry at position client_id, gives the current estimate of the allocation value of client_id

    __counter: int
        initialized with -1


    Methods
    ----------
    _update_estimates

    _estimate_optimization_objective

    _estimate_dissimilarity

    __init_allocation_vec

    """

    def __init__(
            self,
            activity_simulator,
            activity_estimator,
            clients_weights_dict,
            clients_optimums_dict,
            smoothness_param=0.0,
            tolerance=0.0,
            bias_const=1.0,
            rng=None
    ):
        super(MarkovianClientsSampler, self).__init__(
            activity_simulator=activity_simulator,
            activity_estimator=activity_estimator,
            clients_weights_dict=clients_weights_dict,
            rng=rng
        )

        self._smoothness_param = smoothness_param
        self._tolerance = tolerance
        self._bias_const = bias_const

        self._clients_weights_vec = np.array(
            [self.clients_weights_dict[idx] for idx in range(self.n_clients)]
        )

        assert_array_in_simplex(self._clients_weights_vec)

        self._clients_optimums_vec = np.array(
            [clients_optimums_dict[idx] for idx in range(self.n_clients)]
        )

        self._loss_estimates_vec = np.zeros(self.n_clients, dtype=np.float32)

        self._allocation_vec = np.zeros(self.n_clients, dtype=np.float32)

        self._clients_ids_by_stability = list()
        self._clients_ids_by_availability = list()

        self.__counter = -1

    def __init_allocation_vec(self):
        return np.copy(self._clients_weights_vec) / np.copy(self._availability_vec)

    def _update_estimates(self, active_clients, loss_dict=None):
        """update the estimates of clients losses

        Initialize and updates the _loss_estimates_vec attribute

        Parameters
        ----------
        active_clients: List[int]

        loss_dict: Dict[int: float]

        Returns
        -------
            None

        """
        active_clients = set(active_clients)

        # update loss estimates

        if self.__counter == -1:
            mean_active_clients_loss = 0
            for client_id in active_clients:
                self._loss_estimates_vec[client_id] = loss_dict[client_id]

                mean_active_clients_loss += loss_dict[client_id]

            mean_active_clients_loss /= len(active_clients)

            for client_id in self.clients_ids:
                if client_id not in active_clients:
                    self._loss_estimates_vec[client_id] = mean_active_clients_loss

        else:

            for client_id in active_clients:
                self._loss_estimates_vec[client_id] = \
                    self._smoothness_param * self._loss_estimates_vec[client_id] + \
                    (1 - self._smoothness_param) * loss_dict[client_id]

        # update clients' activity estimates

        self._availability_types_dict, self._availability_dict, self._stability_types_dict, self._stability_dict = \
            self.activity_estimator.gather_activity_estimates(list(active_clients))

        self._availability_vec = np.array(
            [self._availability_dict[idx] for idx in range(self.n_clients)]
        )

        self.__counter += 1

    def _estimate_dissimilarity(self):
        """estimates clients dissimilarity

        Returns
        -------
            float
        """
        return np.max(self._loss_estimates_vec - self._clients_optimums_vec)

    def _estimate_optimization_objective(self, allocation_vector):
        """computes the Markovian sampler objective

        Estimate the total error of the true global objective as sum of:
            * an optimization term: the optimization error of the biased global objective
            * a bias term, expressed in terms of product between:
                ** the total variation distance between the true and the current weights
                ** the client dissimilarity

        CA-Fed identifies the number of clients to remove by attempting
        to minimize this total error (see _truncate_clients).

        Parameters
        ----------
        allocation_vector: 1-D numpy.array
            entry at position client_id, gives the allocation value of client_id

        Returns
        -------
            * float: total error

        """
        weights = self._availability_vec * allocation_vector
        weights /= weights.sum()

        optimization_term = (self._loss_estimates_vec - self._clients_optimums_vec) @ weights

        dissimilarity = self._estimate_dissimilarity()
        tv_value = np.square(tv_distance(self._clients_weights_vec, weights))
        bias_term = dissimilarity * tv_value * self._bias_const

        total_error = optimization_term + bias_term

        return total_error

    def _truncate_clients(self, ordered_ids):
        """
        progressively truncates some clients by attempting to minimize
        the estimate of the optimization objective

        Parameters
        ----------
        ordered_ids: List[int]
            ordered list of clients_ids, the clients will be explored according to the order

        Returns
        -------
            1-D numpy.array: allocation vector

        """

        running_allocation_vec = np.copy(self._allocation_vec)
        current_allocation_vec = np.copy(self._allocation_vec)

        running_objective = \
            self._estimate_optimization_objective(running_allocation_vec)

        for client_id in ordered_ids:

            current_allocation_vec[client_id] = 0

            if current_allocation_vec.sum() == 0:
                break

            current_objective = \
                self._estimate_optimization_objective(current_allocation_vec)

            if running_objective - current_objective >= self._tolerance:
                running_allocation_vec = np.copy(current_allocation_vec)
                running_objective = current_objective
            else:
                current_allocation_vec = np.copy(running_allocation_vec)

        return running_allocation_vec

    def sample(self, active_clients, loss_dict):
        """implementation of the abstract method ClientSampler.sample for CA-Fed

        Parameters
        ----------
        active_clients: List[int]
        loss_dict: Dict[int: float] or None

        Returns
        -------
            * List[int]: indices of the sampled clients_dict
            * List[float]: weights to be associated to the sampled clients_dict
        """

        self._update_estimates(
            active_clients=active_clients,
            loss_dict=loss_dict
        )

        self._allocation_vec = self.__init_allocation_vec()

        self._clients_ids_by_stability = \
            sorted(self.clients_ids, key=lambda idx: abs(self._stability_dict[idx]), reverse=True)

        self._allocation_vec = self._truncate_clients(
            ordered_ids=self._clients_ids_by_stability
        )

        self._clients_ids_by_availability = \
            sorted(self.clients_ids, key=lambda idx: self._availability_dict[idx], reverse=False)

        self._allocation_vec = self._truncate_clients(
            ordered_ids=self._clients_ids_by_availability
        )

        sampled_clients_ids, sampled_clients_weights = [], []

        for client_id in active_clients:
            if self._allocation_vec[client_id] > ERROR:
                sampled_clients_ids.append(client_id)
                sampled_clients_weights.append(
                    self._allocation_vec[client_id]
                )

        self.step(active_clients, sampled_clients_ids, sampled_clients_weights)

        return sampled_clients_ids, sampled_clients_weights


class AdaFedClientsSampler(ClientsSampler):
    r"""Participation-Aware Federated Learning

    Implements AdaFed proposed in
    "AdaFed: Optimizing Participation-Aware FederatedLearning with
        Adaptive Aggregation Weights"__(https://ieeexplore.ieee.org/abstract/document/9762058)

    Attributes
    ----------
    _full_participation: bool
        if True, all clients are taken at every round

    Methods
    ----------
    _update_estimates

    _build_optimization_problem

    """

    def __init__(
            self,
            activity_simulator,
            activity_estimator,
            clients_weights_dict,
            full_participation=True,
            rng=None
    ):
        super(AdaFedClientsSampler, self).__init__(
            activity_simulator=activity_simulator,
            activity_estimator=activity_estimator,
            clients_weights_dict=clients_weights_dict,
            rng=rng
        )

        self._full_participation = full_participation

    def _build_optimization_problem(self):
        """

        Returns
        -------

        """
        raise NotImplementedError()

    def _normalize_weights(self, clients_weights):
        """ normalize weights before aggregation

        Parameters
        ----------
        clients_weights: List[float]


        Returns
        -------
            * List[float]

        """
        if isinstance(clients_weights, list):
            clients_weights = np.array(clients_weights)

        clients_weights /= clients_weights.sum()

        return clients_weights.tolist()

    def sample(self, active_clients, loss_dict):
        """implementation of the abstract method ClientSampler.sample for AdaFed

        Parameters
        ----------
        active_clients: List[int]
        loss_dict: Dict[int: float] or None

        Returns
        -------
            * List[int]: indices of the sampled clients_dict
            * List[float]: weights to be associated to the sampled clients_dict
        """

        self._update_estimates(active_clients=active_clients)

        sampled_clients_ids, sampled_clients_weights = [], []
        if self._full_participation:

            for client_id in active_clients:
                sampled_clients_ids.append(client_id)
                sampled_clients_weights.append(
                    self.clients_weights_dict[client_id] / self._availability_dict[client_id]
                )
        else:
            self._build_optimization_problem()

        sampled_clients_weights = self._normalize_weights(sampled_clients_weights)

        self.step(active_clients, sampled_clients_ids, sampled_clients_weights)

        return sampled_clients_ids, sampled_clients_weights


class F3AST(ClientsSampler):
    r"""Federated Averaging aided by an Adaptive Sampling Technique

    Implements F3AST proposed in
    "Federated Learning Under Intermittent Client Availability andTime-Varying
     Communication Constraints"__(https://arxiv.org/abs/2205.06730)

    Attributes
    ----------
    _smoothness_param: float

    _n_clients_per_round: int
        number of clients to sample at every round

    """

    def __init__(
            self,
            activity_simulator,
            activity_estimator,
            clients_weights_dict,
            n_clients_per_round,
            smoothness_param=0.0,
            rng=None
    ):

        super(F3AST, self).__init__(
            activity_simulator=activity_simulator,
            activity_estimator=activity_estimator,
            clients_weights_dict=clients_weights_dict,
            rng=rng
        )

        self._smoothness_param = smoothness_param
        self._n_clients_per_round = n_clients_per_round

    def _update_estimates(self, active_clients, loss_dict=None):

        # update clients' activity estimates

        self._availability_types_dict, self._availability_dict, _, _ = \
            self.activity_estimator.gather_activity_estimates(active_clients)

        # smoothing average of past participation rates

        active_clients = set(active_clients)

        for client_id in self.clients_ids:

            self._availability_dict[client_id] *= (1 - self._smoothness_param)

            if client_id in active_clients:
                self._availability_dict[client_id] += self._smoothness_param

    def sample(self, active_clients, loss_dict=None):
        """implementation of the abstract method ClientSampler.sample for F3AST

        Parameters
        ----------
        active_clients: List[int]
        loss_dict: Dict[int: float] or None

        Returns
        -------
            * List[int]: indices of the sampled clients_dict
            * List[float]: weights to be associated to the sampled clients_dict
        """

        self._update_estimates(active_clients)

        clients_ids_by_rate = sorted(
            active_clients,
            key=lambda idx: self.clients_weights_dict[idx] ** 2 / self._availability_dict[idx] ** 2,
            reverse=True
        )

        sampled_clients_ids = clients_ids_by_rate[:self._n_clients_per_round]

        sampled_clients_weights = list()
        for client_id in sampled_clients_ids:
            sampled_clients_weights.append(
                self.clients_weights_dict[client_id] / self._availability_dict[client_id]
            )

        self.step(active_clients, sampled_clients_ids, sampled_clients_weights)

        return sampled_clients_ids, sampled_clients_weights
