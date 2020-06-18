"""Dedicated module containing depletion function

Provided to avoid some circular imports
"""
from itertools import repeat, starmap
from multiprocessing import Pool


class PoolShim:
    """Mimic of :class:`multiprocessing.Pool` but works in serial

    Intended to be a drop-in replacement for ``Pool`` for depletion, e.g.

    .. code::

        with PoolShim() as pool:
            pool.starmap(...)

    """

    def __enter__(self):
        return self

    def __exit__(self, _exec_type, _exec_value, _traceback):
        pass

    def starmap(self, func, inputs):
        """Use :func:`itertools.starmap` for depleting"""
        return starmap(func, inputs)


class DepletionDispatcher:
    """Constrol distribution of solution to Bateman equations

    Parameters
    ----------
    use_multiprocessing : bool
        Flag to enable or disable the use of :mod:`multiprocessing` when
        solving the Bateman equations for each material. Default is
        ``True``, meaning materials will be depleted across all
        processing units. However, communication may be limited on some
        computing environments, depending on hardware and network
        configuration, and MPI restrictions. Deactivating this setting
        will avoid these issues, but will take longer as not all processing
        units are used when solving the Bateman equations

    """

    def __init__(self, use_multiprocessing):
        self._pool_class = Pool if use_multiprocessing else PoolShim

    def deplete(self, func, chain, x, rates, dt, matrix_func=None):
        """Deplete materials using given reaction rates for a specified time

        Parameters
        ----------
        func : callable
            Function to use to get new compositions. Expected to have the
            signature ``func(A, n0, t) -> n1``
        chain : openmc.deplete.Chain
            Depletion chain
        x : list of numpy.ndarray
            Atom number vectors for each material
        rates : openmc.deplete.ReactionRates
            Reaction rates (from transport operator)
        dt : float
            Time in [s] to deplete for
        maxtrix_func : callable, optional
            Function to form the depletion matrix after calling
            ``matrix_func(chain, rates, fission_yields)``, where
            ``fission_yields = {parent: {product: yield_frac}}``
            Expected to return the depletion matrix required by
            ``func``

        Returns
        -------
        x_result : list of numpy.ndarray
            Updated atom number vectors for each material

        """

        fission_yields = chain.fission_yields
        if len(fission_yields) == 1:
            fission_yields = repeat(fission_yields[0])
        elif len(fission_yields) != len(x):
            raise ValueError(
                "Number of material fission yield distributions {} is not "
                "equal to the number of compositions {}".format(
                    len(fission_yields), len(x)))

        if matrix_func is None:
            matrices = map(chain.form_matrix, rates, fission_yields)
        else:
            matrices = map(matrix_func, repeat(chain), rates, fission_yields)

        # Use multiprocessing pool to distribute work
        with self._pool_class() as pool:
            inputs = zip(matrices, x, repeat(dt))
            x_result = list(pool.starmap(func, inputs))

        return x_result


def deplete(func, chain, x, rates, dt, matrix_func=None):
    """Deplete materials using given reaction rates for a specified time

    .. note::

        If this function is expected to be called repeatedly, use
        :class:`DepletionDispatcher`

    Parameters
    ----------
    func : callable
        Function to use to get new compositions. Expected to have the
        signature ``func(A, n0, t) -> n1``
    chain : openmc.deplete.Chain
        Depletion chain
    x : list of numpy.ndarray
        Atom number vectors for each material
    rates : openmc.deplete.ReactionRates
        Reaction rates (from transport operator)
    dt : float
        Time in [s] to deplete for
    maxtrix_func : callable, optional
        Function to form the depletion matrix after calling
        ``matrix_func(chain, rates, fission_yields)``, where
        ``fission_yields = {parent: {product: yield_frac}}``
        Expected to return the depletion matrix required by
        ``func``

    Returns
    -------
    x_result : list of numpy.ndarray
        Updated atom number vectors for each material

    """

    return DepletionDispatcher(Pool).deplete(
        func, chain, x, rates, dt, matrix_func=None)
