import numpy as np
import numpy.linalg as LA


def assert_array_in_simplex(p, error=1e-4):
    assert np.abs(np.nansum(p) - 1) <= error, f"{p} should sum-up to one"
    assert np.nanmin(p) >= -error, f"{p} should have non-negative entries"


def chi2_distance(p, q):
    """compute the chi-square distance

    p and q are expected to be in the unitary simplex

    Parameters
    ----------
    p: 1-D numpy.array

    q: 1-D numpy.array


    Returns
    -------
        * float
    """
    assert_array_in_simplex(p)
    assert_array_in_simplex(q)

    temp = np.square(p - q) / q

    return temp.sum()


def tv_distance(p, q):
    """compute the total variation distance

    # TODO: add support for math documentation
    TV(p, q) = 1/2 * ||p - q||_{1}

    p and q are expected to be in the unitary simplex

    Parameters
    ----------
    p: 1-D numpy.array

    q: 1-D numpy.array

    Returns
    -------
        * float
    """
    assert_array_in_simplex(p)
    assert_array_in_simplex(q)

    return 0.5 * LA.norm(p-q, ord=1)
