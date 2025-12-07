#!/usr/bin/env python3
"""
Test to understand PyMOL's projection scaling and coordinate system.

This test investigates:
1. How PyMOL's zoom(complete=1) scales the structure
2. Whether X and Y are scaled differently
3. The relationship between model size, rotated bbox, and projection size
"""

import numpy as np
from scipy.spatial.transform import Rotation

def analyze_pymol_scaling():
    """Analyze how PyMOL might be scaling the projection."""
    
    # From debug output
    model_size = 232.11  # Å (original model bounding box)
    rotated_bbox_size = 299.76  # Å (max distance from center, doubled)
    projection_size = 251  # pixels
    pixel_size = 1.1060  # Å/pixel (alignment pixel size)
    
    # Current effective pixel size calculation
    effective_pixel_size = rotated_bbox_size / (projection_size / 1.2)
    
    print("=" * 80)
    print("PyMOL Projection Scaling Analysis")
    print("=" * 80)
    print()
    print(f"Model size (original): {model_size:.2f} Å")
    print(f"Rotated bbox size (max dist from center * 2): {rotated_bbox_size:.2f} Å")
    print(f"Projection size: {projection_size} pixels")
    print(f"Alignment pixel size: {pixel_size:.4f} Å/pixel")
    print(f"Effective pixel size: {effective_pixel_size:.4f} Å/pixel")
    print()
    
    # If PyMOL scales the rotated bbox to fit in projection_size with 1.2x padding:
    # The bbox should fit in: projection_size / 1.2 pixels
    # So: 1 pixel = rotated_bbox_size / (projection_size / 1.2) Angstroms
    print("Scaling calculation:")
    print(f"  PyMOL fits {rotated_bbox_size:.2f} Å into {projection_size / 1.2:.1f} pixels")
    print(f"  Therefore: 1 pixel = {rotated_bbox_size / (projection_size / 1.2):.4f} Å")
    print()
    
    # But wait - maybe PyMOL uses the original model size, not the rotated bbox?
    model_based_pixel_size = model_size / (projection_size / 1.2)
    print(f"If PyMOL used original model size:")
    print(f"  1 pixel = {model_based_pixel_size:.4f} Å")
    print(f"  This is {model_based_pixel_size / effective_pixel_size:.2f}x different from rotated bbox")
    print()
    
    # The -70 pixel X offset suggests a systematic error
    # -70 pixels at effective_pixel_size = -70 * 1.4331 = -100.3 Å
    # -70 pixels at pixel_size = -70 * 1.1060 = -77.4 Å
    print("X offset analysis:")
    x_offset_pixels = -70.0
    x_offset_angstroms_effective = x_offset_pixels * effective_pixel_size
    x_offset_angstroms_alignment = x_offset_pixels * pixel_size
    print(f"  -70 pixels = {x_offset_angstroms_effective:.2f} Å (at effective pixel size)")
    print(f"  -70 pixels = {x_offset_angstroms_alignment:.2f} Å (at alignment pixel size)")
    print()
    
    # As a percentage of projection size
    x_offset_percent = (x_offset_pixels / projection_size) * 100
    print(f"  -70 pixels = {x_offset_percent:.1f}% of projection size")
    print()
    
    # Hypothesis: Maybe PyMOL uses a different scale factor for X vs Y?
    # Or maybe the bounding box calculation is different for X vs Y?
    print("Hypothesis testing:")
    print("1. PyMOL might use different scale factors for X vs Y")
    print("2. PyMOL might use original model size, not rotated bbox")
    print("3. There might be a coordinate system offset in PyMOL's image rendering")
    print("4. The projection center might not match the structure center")
    print()


def test_coordinate_system_consistency():
    """Test if X and Y coordinates are handled consistently."""
    
    print("=" * 80)
    print("Coordinate System Consistency Test")
    print("=" * 80)
    print()
    
    # Test point
    euler = np.array([1.594551, -1.319266, -0.858527])
    R = Rotation.from_euler('ZYZ', euler).as_matrix()
    
    # Marker coordinates
    marker1_model = np.array([209.0, 116.0, 285.0])
    structure_center = np.array([247.03983, 222.27774, 307.78766])
    marker1_centered = marker1_model - structure_center
    marker1_rotated = R @ marker1_centered
    
    print(f"Marker 1 (model space): {marker1_model}")
    print(f"Marker 1 (centered): {marker1_centered}")
    print(f"Marker 1 (rotated): {marker1_rotated}")
    print()
    
    # Current transformation
    effective_pixel_size = 1.4331
    marker_x = marker1_rotated[1] / effective_pixel_size
    marker_y = marker1_rotated[0] / effective_pixel_size
    
    print(f"Current transformation:")
    print(f"  X = rotated[1] / effective_pixel_size = {marker1_rotated[1]:.2f} / {effective_pixel_size:.4f} = {marker_x:.2f}")
    print(f"  Y = rotated[0] / effective_pixel_size = {marker1_rotated[0]:.2f} / {effective_pixel_size:.4f} = {marker_y:.2f}")
    print()
    
    # If we need -70 pixel X offset, the actual X should be:
    actual_x = marker_x - 70.0
    print(f"If we apply -70 pixel X offset:")
    print(f"  Actual X = {marker_x:.2f} - 70.0 = {actual_x:.2f}")
    print(f"  This suggests the correct X should be {actual_x:.2f} pixels from center")
    print()
    
    # What would give us this X value?
    # If we used a different pixel size for X:
    x_pixel_size_alt = marker1_rotated[1] / actual_x
    print(f"Alternative X pixel size to get correct position:")
    print(f"  X_pixel_size = {marker1_rotated[1]:.2f} / {actual_x:.2f} = {x_pixel_size_alt:.4f} Å/pixel")
    print(f"  This is {x_pixel_size_alt / effective_pixel_size:.2f}x the effective pixel size")
    print()


if __name__ == "__main__":
    analyze_pymol_scaling()
    test_coordinate_system_consistency()

