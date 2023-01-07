import string

SAMPLER_TYPES = {"markov", "fast", "unbiased", "adafed"}
ESTIMATOR_TYPES = {"oracle", "bayesian"}
LOADERS_TYPES = {"mnist", "cifar10"}
AGGREGATOR_TYPES = {"centralized", "no_communication"}

MOMENTUM = 0.9
WEIGHT_DECAY = 1e-2

ERROR = 1e-10
