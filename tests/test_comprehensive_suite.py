#!/usr/bin/env python3
"""
COMPREHENSIVE HYPERTOOLS TEST SUITE
Automated tests covering all functionality, animations, edge cases, and performance
"""

import pytest
import numpy as np
import pandas as pd
import warnings
import time
import tempfile
import os
import sys
from pathlib import Path

# Add hypertools to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import hypertools as hyp

# Suppress warnings during testing
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['JUPYTER_PLATFORM_DIRS'] = '1'


class TestDataGeneration:
    """Test data generation utilities"""
    
    @staticmethod
    def generate_test_datasets():
        """Generate comprehensive test datasets covering all edge cases"""
        datasets = {
            # Standard datasets
            'normal': np.random.randn(100, 10),
            'small': np.random.randn(10, 3),
            'large': np.random.randn(1000, 50),
            'wide': np.random.randn(50, 200),
            'tall': np.random.randn(2000, 5),
            
            # Edge case datasets
            'single_point': np.random.randn(1, 5),
            'single_feature': np.random.randn(100, 1),
            'tiny': np.random.randn(3, 2),
            'minimal': np.random.randn(2, 2),
            
            # Special value datasets
            'zeros': np.zeros((20, 5)),
            'ones': np.ones((20, 5)),
            'large_values': np.random.randn(50, 5) * 1e6,
            'small_values': np.random.randn(50, 5) * 1e-6,
            
            # Structured datasets  
            'time_series': np.cumsum(np.random.randn(100, 3), axis=0),
            'categorical_like': np.random.choice([0, 1, 2], size=(100, 3)),
            'sparse_like': np.random.choice([0, 0, 0, 0, 1], size=(100, 10)),
        }
        
        # Add NaN and inf datasets
        datasets['with_nan'] = np.random.randn(50, 5)
        datasets['with_nan'][10:15, 2] = np.nan
        
        datasets['with_inf'] = np.random.randn(50, 5) 
        datasets['with_inf'][5:10, 1] = np.inf
        datasets['with_inf'][15:20, 3] = -np.inf
        
        return datasets
    
    @staticmethod
    def generate_list_datasets():
        """Generate list-based datasets for testing multi-dataset functions"""
        data1 = np.random.randn(50, 5)
        data2 = np.random.randn(45, 5) 
        data3 = np.random.randn(60, 5)
        
        return {
            'uniform_list': [data1, data2, data3],
            'mixed_sizes': [data1[:30], data2[:20], data3[:40]],
            'mixed_dims': [data1[:, :3], data2[:, :4], data3],
            'single_item_list': [data1],
        }


class TestPlotFunctionality:
    """Test all plotting functionality"""
    
    def setup_method(self):
        self.datasets = TestDataGeneration.generate_test_datasets()
        self.list_datasets = TestDataGeneration.generate_list_datasets()
    
    @pytest.mark.parametrize("dataset_name", ['normal', 'small', 'large'])
    def test_basic_plotting(self, dataset_name):
        """Test basic plotting functionality"""
        data = self.datasets[dataset_name]
        
        # Test basic plot
        fig = hyp.plot(data)
        assert fig is not None
        
        # Test with reduction
        fig = hyp.plot(data, reduce={'model': 'PCA', 'args': [], 'kwargs': {'n_components': 3}})
        assert fig is not None
    
    @pytest.mark.parametrize("param_name,param_value", [
        ('size', 8),
        ('alpha', 0.5),
        ('color', 'red'),
        ('markersize', 10),
        ('opacity', 0.7),
    ])
    def test_plot_parameters(self, param_name, param_value):
        """Test all plot parameters work correctly"""
        data = self.datasets['normal'][:50, :3]  # 3D data
        
        kwargs = {param_name: param_value}
        fig = hyp.plot(data, **kwargs)
        assert fig is not None
    
    def test_edge_case_plotting(self):
        """Test plotting with edge case datasets"""
        
        # Test single point (should work with warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = hyp.plot(self.datasets['single_point'])
            assert fig is not None
        
        # Test single feature (should auto-duplicate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = hyp.plot(self.datasets['single_feature'])
            assert fig is not None
    
    def test_list_plotting(self):
        """Test plotting with list datasets"""
        for name, data_list in self.list_datasets.items():
            fig = hyp.plot(data_list)
            assert fig is not None
    
    def test_plotting_modes(self):
        """Test different plotting modes"""
        data = self.datasets['normal'][:50, :3]
        
        modes = ['markers', 'lines', 'lines+markers']
        for mode in modes:
            fig = hyp.plot(data, mode=mode)
            assert fig is not None


class TestAnimationSystem:
    """Test animation functionality comprehensively"""
    
    def setup_method(self):
        # Create time-series data perfect for animation
        n_timepoints = 10
        n_features = 3
        self.animation_data = {}
        
        for t in range(n_timepoints):
            self.animation_data[t] = np.random.randn(20, n_features) + t * 0.5
        
        # Convert to DataFrame with time index
        frames = []
        for t, data in self.animation_data.items():
            frame = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
            frame['time'] = t
            frames.append(frame)
        
        self.time_series_df = pd.concat(frames, ignore_index=True)
    
    def test_animation_creation(self):
        """Test basic animation creation"""
        # Test window-style animation
        anim = hyp.plot(self.time_series_df, animate='window')
        assert anim is not None
    
    def test_animation_sliding_window(self):
        """Test sliding window progression"""
        from hypertools.plot.animate import Animator
        
        animator = Animator(self.time_series_df, style='window')
        
        # Test window progression
        for i in range(5):
            window_data = animator.get_window(self.time_series_df, i, i+2)
            assert len(window_data) > 0, f"Window {i}-{i+2} should contain data"
    
    def test_animation_parameters(self):
        """Test animation with various parameters"""
        params = [
            {'style': 'window', 'duration': 5},
            {'style': 'window', 'framerate': 15},
        ]
        
        for param_set in params:
            anim = hyp.plot(self.time_series_df, animate=param_set['style'], **param_set)
            assert anim is not None


class TestDataManipulation:
    """Test all data manipulation functions"""
    
    def setup_method(self):
        self.datasets = TestDataGeneration.generate_test_datasets()
    
    @pytest.mark.parametrize("dataset_name", ['normal', 'large', 'wide'])
    def test_normalization(self, dataset_name):
        """Test data normalization"""
        data = self.datasets[dataset_name]
        normalized = hyp.normalize(data)
        
        assert normalized.shape == data.shape
        assert np.allclose(normalized.min(), 0, atol=1e-10)
        assert np.allclose(normalized.max(), 1, atol=1e-10)
    
    @pytest.mark.parametrize("model", ['ZScore', 'Smooth', 'Resample'])
    def test_manip_functions(self, model):
        """Test all manipulation models"""
        data = self.datasets['normal']
        
        if model == 'Smooth':
            result = hyp.manip(data, model=model, kernel_width=5)
        elif model == 'Resample':
            result = hyp.manip(data, model=model, n_samples=50)
        else:
            result = hyp.manip(data, model=model)
        
        assert result is not None
        assert hasattr(result, 'shape')
    
    def test_smooth_validation(self):
        """Test smooth function parameter validation"""
        data = self.datasets['small']
        
        # Test auto-correction of invalid parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = hyp.manip(data, model='Smooth', kernel_width=3, order=5)
            assert result is not None
    
    def test_alignment(self):
        """Test data alignment functionality"""
        data_list = self.datasets['normal']
        split_data = [data_list[:50], data_list[50:]]
        
        aligned = hyp.align(split_data)
        assert len(aligned) == 2
        assert all(hasattr(d, 'shape') for d in aligned)


class TestDimensionalityReduction:
    """Test dimensionality reduction methods"""
    
    def setup_method(self):
        self.datasets = TestDataGeneration.generate_test_datasets()
    
    @pytest.mark.parametrize("model", ['PCA', 'IncrementalPCA'])  
    def test_linear_reduction(self, model):
        """Test linear dimensionality reduction methods"""
        data = self.datasets['large']
        
        reduced = hyp.reduce(data, model=model, n_components=3)
        assert reduced.shape == (data.shape[0], 3)
    
    @pytest.mark.slow
    @pytest.mark.parametrize("model", ['TSNE', 'UMAP'])
    def test_nonlinear_reduction(self, model):
        """Test nonlinear dimensionality reduction (marked as slow)"""
        data = self.datasets['normal'][:50]  # Smaller dataset for speed
        
        reduced = hyp.reduce(data, model=model, n_components=2)
        assert reduced.shape == (data.shape[0], 2)


class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    def setup_method(self):
        self.large_data = np.random.randn(5000, 100)
        self.medium_data = np.random.randn(1000, 50)
    
    def test_plot_performance(self):
        """Benchmark plotting performance"""
        start_time = time.time()
        fig = hyp.plot(self.medium_data)
        plot_time = time.time() - start_time
        
        assert plot_time < 10.0, f"Plotting took {plot_time:.2f}s, should be < 10s"
        assert fig is not None
    
    def test_reduce_performance(self):
        """Benchmark reduction performance"""
        start_time = time.time()
        reduced = hyp.reduce(self.large_data, model='IncrementalPCA', n_components=10)
        reduce_time = time.time() - start_time
        
        assert reduce_time < 30.0, f"Reduction took {reduce_time:.2f}s, should be < 30s"
        assert reduced.shape == (self.large_data.shape[0], 10)
    
    def test_normalize_performance(self):
        """Benchmark normalization performance"""
        start_time = time.time()
        normalized = hyp.normalize(self.large_data)
        norm_time = time.time() - start_time
        
        assert norm_time < 5.0, f"Normalization took {norm_time:.2f}s, should be < 5s"
        assert normalized.shape == self.large_data.shape


class TestIntegrationScenarios:
    """Test realistic usage scenarios"""
    
    def test_complete_analysis_pipeline(self):
        """Test a complete analysis pipeline"""
        # Generate synthetic dataset
        data = np.random.randn(200, 20)
        
        # Complete pipeline: normalize -> reduce -> plot
        normalized = hyp.normalize(data)
        reduced = hyp.reduce(normalized, model='PCA', n_components=3)
        fig = hyp.plot(reduced)
        
        assert fig is not None
        assert reduced.shape == (200, 3)
    
    def test_multi_dataset_workflow(self):
        """Test workflow with multiple datasets"""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10) + 2  # Shifted
        data3 = np.random.randn(90, 10) - 1  # Shifted
        
        datasets = [data1, data2, data3]
        
        # Align datasets
        aligned = hyp.align(datasets)
        
        # Plot aligned data
        fig = hyp.plot(aligned)
        
        assert fig is not None
        assert len(aligned) == 3
    
    def test_time_series_analysis(self):
        """Test time series analysis workflow"""
        # Create time series data
        n_timepoints = 20
        n_features = 5
        time_data = {}
        
        for t in range(n_timepoints):
            time_data[t] = np.random.randn(30, n_features) + np.sin(t/5) * 2
        
        # Convert to animation format
        frames = []
        for t, data in time_data.items():
            frame = pd.DataFrame(data, columns=['dim1', 'dim2', 'dim3', 'dim4', 'dim5'])  # String column names
            frame['time'] = t
            frames.append(frame)

        time_df = pd.concat(frames, ignore_index=True)
        
        # Create animation
        anim = hyp.plot(time_df, animate='window')
        assert anim is not None


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        invalid_inputs = [
            None,
            "string",
            [],
            {},
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((ValueError, TypeError, AttributeError)):
                hyp.plot(invalid_input)
    
    def test_empty_data(self):
        """Test handling of empty datasets"""
        empty_data = np.array([]).reshape(0, 5)
        
        with pytest.raises(ValueError):
            hyp.plot(empty_data)
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched dimensions"""
        data1 = np.random.randn(50, 5)
        data2 = np.random.randn(50, 3)  # Different number of features
        
        # This should either work (with padding) or raise a clear error
        try:
            result = hyp.plot([data1, data2])
            assert result is not None
        except ValueError as e:
            assert "dimension" in str(e).lower() or "shape" in str(e).lower()


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v", "--tb=short"]) 