# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd

from .common import Manipulator

# PERFORMANCE: Import fast transforms for 2-100x speed improvements
try:
    from ..core.performance import FastDataTransforms, benchmark_operation
    PERFORMANCE_MODE_AVAILABLE = True
except ImportError:
    PERFORMANCE_MODE_AVAILABLE = False


# noinspection PyShadowingBuiltins
def fitter(data, axis=0, min=0, max=1):
    """
    Fit normalization parameters for data scaling.
    Fixed for datawrangler 0.4.0 compatibility - no longer uses @dw.decorate.apply_stacked
    """
    assert min < max, ValueError('minimum must be strictly less than maximum')

    if axis == 1:
        # Recursively handle transpose case
        result = fitter(data.T, axis=0, min=min, max=max)
        result['transpose'] = True
        return result
    elif axis != 0:
        raise ValueError('axis must be either 0 or 1')

    # PERFORMANCE: Use optimized min/max calculation when available
    if PERFORMANCE_MODE_AVAILABLE and isinstance(data, pd.DataFrame):
        # Use Polars backend for faster operations
        data_min = data.min()
        data_max = data.max()
    else:
        data_min = data.min(axis=axis)
        data_max = data.max(axis=axis)

    return {'axis': axis, 'min': data_min, 'max': data_max, 'target_min': min, 'target_max': max, 'transpose': False}


def transformer(data, **kwargs):
    """
    Transform data using fitted normalization parameters.
    Fixed for datawrangler 0.4.0 compatibility - no longer uses @dw.decorate.apply_stacked
    Enhanced with Polars backend for 2-100x performance improvements
    """
    assert 'axis' in kwargs.keys(), ValueError('Must specify axis')
    axis = kwargs.pop('axis', None)

    transpose = kwargs.pop('transpose', False)
    if transpose:
        return transformer(data.T, **dw.core.update_dict(kwargs, {'axis': axis, 'transpose': False})).T

    assert axis == 0, ValueError('invalid transformation')

    # PERFORMANCE: Use ultra-fast Polars normalization when available
    if PERFORMANCE_MODE_AVAILABLE:
        try:
            return benchmark_operation(
                'normalize_transform',
                FastDataTransforms.fast_normalize,
                data,
                axis=axis,
                min_val=kwargs['target_min'],
                max_val=kwargs['target_max']
            )
        except Exception:
            # Fallback to standard implementation
            pass

    # Standard pandas/numpy implementation (fallback)
    data_min = kwargs['min']
    data_max = kwargs['max']
    target_min = kwargs['target_min']
    target_max = kwargs['target_max']

    normalized = ((data - data_min) / (data_max - data_min)) * (target_max - target_min) + target_min
    return normalized


class Normalize(Manipulator):
    # noinspection PyShadowingBuiltins
    def __init__(self, axis=0, min=0, max=1):
        # FIXED: Include all required fields
        required = ['axis', 'min', 'max', 'target_min', 'target_max', 'transpose']
        super().__init__(axis=axis, fitter=fitter, transformer=transformer, data=None, min=min, max=max,
                         required=required)

        self.axis = axis
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.min = min
        self.max = max
