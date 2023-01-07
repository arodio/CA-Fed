import time

import numpy as np


class ActivitySimulator:
    r"""Simulates clients_dict activity

    The activity of each client follows a Markovian model with two states, 'active' and 'inactive'.


    Attributes
    ----------
    clients_ids: 1-D numpy.array (dtype=int)

    n_clients: int

    availabilities: 1-D numpy.array (dtype=float)

    stabilities: 1-D numpy.array (dtype=float)

    activation_probabilities: 2-D array of size (`n_clients`, 2)
        activation probability at every state per client,

    state: 1-D numpy.array(int)
        value `1` corresponds to state active and `0` corresponds to inactive

    __rng: numpy.random._generator.Generator


    Methods
    -------
    _init_state

    _build_activation_probabilities

    step

    get_active_clients

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

        self.__rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

        self.activation_probabilities = self._build_activation_probabilities()

        self.state = self._init_state()

    def _init_state(self):
        state = self.__rng.binomial(n=1, p=self.availabilities, size=self.n_clients)

        return state

    def _build_activation_probabilities(self):
        activation_probabilities = np.zeros((self.n_clients, 2), dtype=np.float32)

        activation_probabilities[:, 1] = self.stabilities + (1 - self.stabilities) * self.availabilities

        activation_probabilities[:, 0] = (1 - self.stabilities) * self.availabilities

        return activation_probabilities

    def step(self):

        p = np.take_along_axis(self.activation_probabilities, self.state.reshape(-1, 1), axis=-1).reshape(-1)

        self.state = \
            self.__rng.binomial(
                n=1,
                p=p,
                size=self.n_clients
            )

    def get_active_clients(self):
        """returns indices of active clients_dict (i.e., having state=1)

        Returns
        -------
            * List[int]
        """
        return self.clients_ids[self.state == 1].tolist()


class FileActivitySimulator(ActivitySimulator):
    r"""Reads the activity of clients from a .json file


    """
    def __init__(
            self,
            clients_ids,
            availability_types,
            availabilities,
            stability_types,
            stabilities,
            active_ids_per_time,
            rng=None
    ):
        super(FileActivitySimulator, self).__init__(
            clients_ids=clients_ids,
            availability_types=availability_types,
            availabilities=availabilities,
            stability_types=stability_types,
            stabilities=stabilities,
            rng=rng
        )

        self.active_ids_per_time = active_ids_per_time

        self.time_step = -1

    def step(self):
        self.time_step += 1

    def get_active_clients(self):
        """returns indices of active clients_dict (i.e., having state=1)

        Returns
        -------
            * List[int]
        """
        try:
            return self.active_ids_per_time[self.time_step]
        except IndexError:
            print("You are outside the simulated range for activity!!")
