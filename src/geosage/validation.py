"""Validate numerical inputs without changing their values or array ordering."""

import numpy as np


def require_real_finite(values, name):
    values = np.asarray(values)
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite real numbers.")
    if not values.size:
        raise ValueError(f"{name} must not be empty.")
    return values


def require_labels(values, name, *, positive=False, max_id=None):
    values = require_real_finite(values, name)
    minimum = 1 if positive else 0
    if (values < minimum).any() or (values != np.floor(values)).any():
        raise ValueError(f"{name} must contain integer IDs >= {minimum}.")
    if max_id is not None and (values > max_id).any():
        raise ValueError(f"{name} exceeds the supported maximum ID {max_id}.")
    return values
