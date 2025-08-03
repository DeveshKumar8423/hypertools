#!/usr/bin/env python3
"""
HYPERTOOLS PERFORMANCE OPTIMIZATION MODULE
Leverages datawrangler 0.4.0's Polars backend for 2-100x performance improvements
"""

import numpy as np
import pandas as pd
import warnings
from typing import Union, List, Optional, Any
import time

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    warnings.warn("Polars not available. Install with 'pip install polars' for 2-100x performance improvements.")

try:
    import datawrangler as dw
    # Check if datawrangler supports Polars backend (version 0.4.0+)
    POLARS_BACKEND_AVAILABLE = hasattr(dw, 'set_backend') and POLARS_AVAILABLE
except ImportError:
    POLARS_BACKEND_AVAILABLE = False


class PerformanceOptimizer:
    """
    Performance optimization manager for hypertools operations
    """
    
    def __init__(self, auto_optimize: bool = True):
        self.auto_optimize = auto_optimize
        self.backend = 'pandas'  # Default backend
        self.performance_stats = {}
        
        if POLARS_BACKEND_AVAILABLE and auto_optimize:
            self.enable_polars_backend()
    
    def enable_polars_backend(self):
        """Enable Polars backend for dramatic performance improvements"""
        if not POLARS_BACKEND_AVAILABLE:
            warnings.warn("Polars backend not available. Using pandas fallback.")
            return False
        
        try:
            dw.set_backend('polars')
            self.backend = 'polars'
            print("Polars backend enabled - expect 2-100x performance improvements!")
            return True
        except Exception as e:
            warnings.warn(f"Failed to enable Polars backend: {e}")
            return False
    
    def disable_polars_backend(self):
        """Disable Polars backend and use pandas"""
        if POLARS_BACKEND_AVAILABLE:
            dw.set_backend('pandas')
        self.backend = 'pandas'
    
    def benchmark_operation(self, operation_name: str, func, *args, **kwargs):
        """Benchmark an operation and store performance stats"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        self.performance_stats[operation_name] = {
            'duration': duration,
            'backend': self.backend,
            'timestamp': start_time
        }
        
        return result
    
    def get_performance_report(self):
        """Get a performance report of all operations"""
        if not self.performance_stats:
            return "No performance data available."
        
        report = f"\nHYPERTOOLS PERFORMANCE REPORT (Backend: {self.backend})\n"
        report += "=" * 60 + "\n"
        
        for op_name, stats in self.performance_stats.items():
            report += f"{op_name:30} {stats['duration']:8.3f}s\n"
        
        total_time = sum(stats['duration'] for stats in self.performance_stats.values())
        report += "-" * 60 + "\n"
        report += f"{'Total Time':30} {total_time:8.3f}s\n"
        
        if self.backend == 'polars':
            report += "\nUsing Polars backend for optimal performance!\n"
        else:
            report += "\nInstall polars and datawrangler>=0.4.0 for 2-100x speed improvements!\n"
        
        return report


class FastDataTransforms:
    """
    High-performance data transformation utilities using Polars when available
    """
    
    @staticmethod
    def fast_normalize(data: Union[np.ndarray, pd.DataFrame], 
                      axis: int = 0, 
                      min_val: float = 0, 
                      max_val: float = 1) -> Union[np.ndarray, pd.DataFrame]:
        """
        Ultra-fast normalization using Polars backend when available
        """
        if POLARS_BACKEND_AVAILABLE and isinstance(data, pd.DataFrame):
            # Convert to Polars for fast operations
            pl_data = pl.from_pandas(data)
            
            if axis == 0:
                # Normalize along columns (each column independently)
                normalized = pl_data.select([
                    ((pl.col(col) - pl.col(col).min()) / 
                     (pl.col(col).max() - pl.col(col).min()) * 
                     (max_val - min_val) + min_val).alias(col)
                    for col in pl_data.columns
                ])
            else:
                # Normalize along rows (each row independently)
                # This is more complex in Polars, fall back to pandas for now
                return FastDataTransforms._pandas_normalize(data, axis, min_val, max_val)
            
            return normalized.to_pandas()
        else:
            return FastDataTransforms._pandas_normalize(data, axis, min_val, max_val)
    
    @staticmethod
    def _pandas_normalize(data, axis=0, min_val=0, max_val=1):
        """Fallback pandas normalization"""
        if isinstance(data, pd.DataFrame):
            if axis == 0:
                return (data - data.min()) / (data.max() - data.min()) * (max_val - min_val) + min_val
            else:
                return data.div(data.max(axis=1), axis=0) * (max_val - min_val) + min_val
        else:
            # NumPy array
            data_min = np.min(data, axis=axis, keepdims=True)
            data_max = np.max(data, axis=axis, keepdims=True)
            return (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
    
    @staticmethod
    def fast_zscore(data: Union[np.ndarray, pd.DataFrame], axis: int = 0) -> Union[np.ndarray, pd.DataFrame]:
        """
        Ultra-fast z-score calculation using Polars backend when available
        """
        if POLARS_BACKEND_AVAILABLE and isinstance(data, pd.DataFrame):
            pl_data = pl.from_pandas(data)
            
            if axis == 0:
                # Z-score along columns
                zscore_data = pl_data.select([
                    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
                    for col in pl_data.columns
                ])
            else:
                # Z-score along rows (more complex, fall back to pandas)
                return FastDataTransforms._pandas_zscore(data, axis)
            
            return zscore_data.to_pandas()
        else:
            return FastDataTransforms._pandas_zscore(data, axis)
    
    @staticmethod
    def _pandas_zscore(data, axis=0):
        """Fallback pandas z-score"""
        if isinstance(data, pd.DataFrame):
            if axis == 0:
                return (data - data.mean()) / data.std()
            else:
                return data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)
        else:
            # NumPy array
            return (data - np.mean(data, axis=axis, keepdims=True)) / np.std(data, axis=axis, keepdims=True)
    
    @staticmethod
    def fast_groupby_operations(data: pd.DataFrame, group_col: str, agg_funcs: dict) -> pd.DataFrame:
        """
        Fast groupby operations using Polars
        """
        if POLARS_BACKEND_AVAILABLE:
            pl_data = pl.from_pandas(data)
            
            # Convert aggregation functions to Polars expressions
            agg_exprs = []
            for col, func in agg_funcs.items():
                if func == 'mean':
                    agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
                elif func == 'sum':
                    agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
                elif func == 'std':
                    agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))
                elif func == 'count':
                    agg_exprs.append(pl.col(col).count().alias(f"{col}_count"))
            
            result = pl_data.group_by(group_col).agg(agg_exprs)
            return result.to_pandas()
        else:
            # Fallback to pandas
            return data.groupby(group_col).agg(agg_funcs)


class MemoryOptimizer:
    """
    Memory optimization utilities for large datasets
    """
    
    @staticmethod
    def optimize_dtypes(data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes to reduce memory usage
        """
        optimized = data.copy()
        
        for col in optimized.columns:
            col_type = optimized[col].dtype
            
            if col_type != 'object':
                c_min = optimized[col].min()
                c_max = optimized[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized[col] = optimized[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized[col] = optimized[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized[col] = optimized[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized[col] = optimized[col].astype(np.float32)
        
        return optimized
    
    @staticmethod
    def memory_usage_report(data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        Generate a memory usage report
        """
        if isinstance(data, pd.DataFrame):
            memory_mb = data.memory_usage(deep=True).sum() / 1024**2
            return f"DataFrame memory usage: {memory_mb:.2f} MB"
        elif isinstance(data, np.ndarray):
            memory_mb = data.nbytes / 1024**2
            return f"NumPy array memory usage: {memory_mb:.2f} MB"
        else:
            return "Unknown data type for memory analysis"


# Global performance optimizer instance
_global_optimizer = PerformanceOptimizer()

def enable_performance_mode():
    """Enable high-performance mode with Polars backend"""
    return _global_optimizer.enable_polars_backend()

def disable_performance_mode():
    """Disable high-performance mode"""
    _global_optimizer.disable_polars_backend()

def get_performance_report():
    """Get the current performance report"""
    return _global_optimizer.get_performance_report()

def benchmark_operation(name: str, func, *args, **kwargs):
    """Benchmark a function call"""
    return _global_optimizer.benchmark_operation(name, func, *args, **kwargs)


# Auto-enable performance mode if available
if __name__ != "__main__":
    enable_performance_mode() 