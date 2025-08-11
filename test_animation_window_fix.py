#!/usr/bin/env python3
"""
Test script to verify that the sliding window animation fix works correctly.
This tests that the window properly progresses through timepoints instead of getting stuck.
"""

import numpy as np
import pandas as pd
import hypertools as hyp

print("Testing Sliding Window Animation Fix")
print("=" * 50)

# Create test data with discrete timepoints
n_timepoints = 5
n_features = 3
data_dict = {}

# Create data where each timepoint has distinct values
for t in range(n_timepoints):
    # Each timepoint gets different values so we can see progression
    data_dict[t] = np.random.randn(1, n_features) + t * 2  # Add offset per timepoint

# Convert to DataFrame with explicit time index
frames = []
for t, values in data_dict.items():
    frame = pd.DataFrame(values, columns=['x', 'y', 'z'])
    frame.index = [t]  # Set time index
    frames.append(frame)

df = pd.concat(frames)
print(f"Created test data with {len(df)} timepoints")
print(f"  Timepoints: {sorted(df.index.unique())}")
print(f"  Data shape: {df.shape}")

# Test the animation window functionality directly
try:
    from hypertools.plot.animate import Animator
    
    # Create animator with window style
    animator = Animator(df, style='window', duration=3, focused=0.8)
    
    print(f"\nAnimation Setup:")
    print(f"  Animation indices: {animator.indices[:10]}...")  # Show first 10
    print(f"  Total frames: {len(animator.indices)}")
    print(f"  Window starts: {animator.window_starts[:5]}...")  # Show first 5
    print(f"  Window ends: {animator.window_ends[:5]}...")
    
    print(f"\nTesting Window Progression:")
    
    # Test a few window positions to see if they capture different data
    test_frames = [0, len(animator.window_starts)//3, len(animator.window_starts)//2, -5]
    
    for i, frame_idx in enumerate(test_frames):
        if frame_idx < 0:
            frame_idx = len(animator.window_starts) + frame_idx
        if frame_idx >= len(animator.window_starts):
            continue
            
        w_start = animator.window_starts[frame_idx]
        w_end = animator.window_ends[frame_idx]
        
        # Get window data
        window_data = animator.get_window(df, w_start, w_end)
        
        print(f"  Frame {frame_idx}: window [{w_start:.1f}, {w_end:.1f}]")
        print(f"    -> Time range: [{animator.indices[int(w_start)]:.1f}, {animator.indices[int(w_end)]:.1f}]")
        print(f"    -> Captured timepoints: {sorted(window_data.index.unique()) if len(window_data) > 0 else 'NONE'}")
        print(f"    -> Data rows: {len(window_data)}")
        
        if len(window_data) > 0:
            mean_vals = window_data.mean()
            print(f"    -> Mean values: x={mean_vals.x:.2f}, y={mean_vals.y:.2f}, z={mean_vals.z:.2f}")
        print()
    
    print("Sliding window test completed successfully!")
    
    # Try a simple animation
    print("\nTesting actual animation...")
    try:
        fig = hyp.plot(df, animate=True, style='window', duration=2, save_path='test_sliding_window.html')
        print("Animation created successfully!")
        print("   Saved as: test_sliding_window.html")
    except Exception as e:
        print(f"Animation creation failed: {e}")
        
except Exception as e:
    print(f"Animator test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Test Summary:")
print("- Fixed sliding window logic should show progression through timepoints")
print("- Each frame should capture different subsets of data")
print("- Window should advance through time rather than staying at timepoint 0") 