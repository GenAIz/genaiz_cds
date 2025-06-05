"""
Code that scales data appropriately given matrix type.
"""


import numpy as np
import scipy

from sklearn.preprocessing import StandardScaler


def scale(table: np.ndarray | scipy.sparse.spmatrix, copy: bool = False,
          scaler: StandardScaler=None) -> tuple[StandardScaler, np.ndarray | scipy.sparse.spmatrix]:
    """
    Scales data in place unless copy is True.  Uses sklearn StandardScaler
    (with mean False for sparse data).  Only dense and sparse matrices are scaled.
    If a scaler is provided then it is used instead of calculating a new StandardScaler.
    Scaler properties are adjusted for sparse and dense matrices.  Be careful since
    using a scaler fit to a dense matrix may not work well, or at all, for a sparse matrix,
    and the inverse.

    :param table: table, only dense or sparse is permitted
    :param copy: true and a copy of the data is scaled
    :param scaler: the standard scaler to use, if not None
    :return: the standard scaler and the scaled data
    """
    if isinstance(table, np.ndarray):
        if scaler is None:
            ss = StandardScaler(copy=copy, with_mean=True)
            scaled = ss.fit_transform(table)
        else:
            ss = scaler
            ss.copy = copy
            ss.with_mean = True
            scaled = ss.transform(table)
    elif scipy.sparse.issparse(table):
        if scaler is None:
            ss = StandardScaler(copy=copy, with_mean=False)
            scaled = ss.fit_transform(table)
        else:
            ss = scaler
            ss.copy = copy
            ss.with_mean = False
            scaled = ss.transform(table)
    else:
        raise ValueError("Cannot normalize table of type {}".format(type(table)))
    return ss, scaled
