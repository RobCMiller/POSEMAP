#!/usr/bin/env python3
"""
Test suite for verifying coordinate transformations in the projection mapping pipeline.

This test suite verifies:
1. Rotation matrix application
2. Coordinate system conventions (model space → view space → projection plane)
3. Pixel size calculations
4. Marker coordinate transformations
"""

import numpy as np
from scipy.spatial.transform import Rotation
import sys
import os

# Add the current directory to the path so we can import particle_mapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from particle_mapper import euler_to_rotation_matrix


def test_rotation_matrix_properties():
    """Test that rotation matrices have correct properties."""
    print("=" * 80)
    print("TEST 1: Rotation Matrix Properties")
    print("=" * 80)
    
    # Test with identity-like rotation
    euler = np.array([0.0, 0.0, 0.0])
    R = euler_to_rotation_matrix(euler, convention='ZYZ')
    
    # Rotation matrix should be orthogonal: R @ R.T = I
    identity = R @ R.T
    assert np.allclose(identity, np.eye(3)), "Rotation matrix should be orthogonal"
    print("✓ Rotation matrix is orthogonal")
    
    # Determinant should be 1 (proper rotation)
    det = np.linalg.det(R)
    assert np.isclose(det, 1.0), f"Determinant should be 1, got {det}"
    print(f"✓ Determinant is 1.0 (proper rotation)")
    
    print()


def test_coordinate_system_conventions():
    """Test coordinate system conventions for model space → view space → projection."""
    print("=" * 80)
    print("TEST 2: Coordinate System Conventions")
    print("=" * 80)
    
    # Define test points in model space
    # These represent known points that we can verify
    test_points = np.array([
        [1.0, 0.0, 0.0],  # Unit vector along X
        [0.0, 1.0, 0.0],  # Unit vector along Y
        [0.0, 0.0, 1.0],  # Unit vector along Z
        [1.0, 1.0, 0.0],  # Point in XY plane
    ])
    
    # Test with a simple rotation (90 degrees around Z)
    euler = np.array([np.pi/2, 0.0, 0.0])  # 90° rotation around Z
    R = euler_to_rotation_matrix(euler, convention='ZYZ')
    
    print("Rotation matrix (90° around Z):")
    print(R)
    print()
    
    # Rotate test points
    rotated = (R @ test_points.T).T
    
    print("Original points (model space):")
    print(test_points)
    print()
    print("Rotated points (view space):")
    print(rotated)
    print()
    
    # For a 90° rotation around Z:
    # [1, 0, 0] should become [0, 1, 0] (X → Y)
    # [0, 1, 0] should become [-1, 0, 0] (Y → -X)
    # [0, 0, 1] should stay [0, 0, 1] (Z unchanged)
    
    expected_rotated = np.array([
        [0.0, 1.0, 0.0],   # X → Y
        [-1.0, 0.0, 0.0],  # Y → -X
        [0.0, 0.0, 1.0],   # Z unchanged
        [-1.0, 1.0, 0.0],  # [1,1,0] → [-1,1,0]
    ])
    
    assert np.allclose(rotated, expected_rotated), "Rotation doesn't match expected result"
    print("✓ Rotation matches expected result")
    print()


def test_projection_coordinate_mapping():
    """Test how 3D coordinates map to 2D projection coordinates."""
    print("=" * 80)
    print("TEST 3: Projection Coordinate Mapping")
    print("=" * 80)
    
    # Test with a real Euler angle from the debug output
    euler = np.array([1.594551, -1.319266, -0.858527])
    R = euler_to_rotation_matrix(euler, convention='ZYZ')
    
    print("Euler angles (rad):", euler)
    print("Euler angles (deg):", np.rad2deg(euler))
    print()
    print("Rotation matrix R (from model to view):")
    print(R)
    print()
    
    # Test point: marker 1 from ChimeraX
    marker1_model = np.array([209.0, 116.0, 285.0])
    structure_center = np.array([247.03983, 222.27774, 307.78766])
    
    # Center the marker
    marker1_centered = marker1_model - structure_center
    print(f"Marker 1 (model space): {marker1_model}")
    print(f"Structure center: {structure_center}")
    print(f"Marker 1 (centered): {marker1_centered}")
    print()
    
    # Rotate to view space
    marker1_rotated = R @ marker1_centered
    print(f"Marker 1 (rotated, view space): {marker1_rotated}")
    print()
    
    # Project to 2D (XY plane)
    # In view space, we're looking down the Z-axis
    # X in projection = X in view space (right)
    # Y in projection = Y in view space (up)
    marker1_projected_2d = marker1_rotated[:2]  # Take X and Y components
    print(f"Marker 1 (projected, 2D): {marker1_projected_2d}")
    print()
    
    # But wait - the code uses rotated[1] for X and rotated[0] for Y!
    # This suggests a coordinate swap or different convention
    marker1_projected_swapped = np.array([marker1_rotated[1], marker1_rotated[0]])
    print(f"Marker 1 (projected, swapped X↔Y): {marker1_projected_swapped}")
    print()
    
    # Calculate pixel coordinates
    pixel_size = 1.1060  # Å/pixel (alignment pixel size)
    effective_pixel_size = 1.4331  # Å/pixel (using rotated bbox)
    
    # Method 1: Direct projection (X→X, Y→Y)
    marker1_pixels_direct = marker1_projected_2d / effective_pixel_size
    print(f"Marker 1 (pixels, direct): {marker1_pixels_direct}")
    
    # Method 2: Swapped (X→Y, Y→X)
    marker1_pixels_swapped = marker1_projected_swapped / effective_pixel_size
    print(f"Marker 1 (pixels, swapped): {marker1_pixels_swapped}")
    
    # Method 3: With Y negation (like COM calculation)
    marker1_pixels_negated = np.array([marker1_rotated[1] / effective_pixel_size, 
                                       -marker1_rotated[0] / effective_pixel_size])
    print(f"Marker 1 (pixels, swapped + Y negated): {marker1_pixels_negated}")
    print()
    
    # From debug output, we know:
    # Marker 1 (projected, pixels from center): (-2.21, 28.44) with effective_pixel_size
    # This matches: X = rotated[1] / effective_pixel_size, Y = rotated[0] / effective_pixel_size
    expected_pixels = np.array([-2.21, 28.44])
    print(f"Expected pixels (from debug): {expected_pixels}")
    print(f"Calculated pixels (swapped): {marker1_pixels_swapped}")
    print(f"Match: {np.allclose(marker1_pixels_swapped, expected_pixels, atol=0.1)}")
    print()


def test_pymol_coordinate_system():
    """Test PyMOL's coordinate system and how it maps to image coordinates."""
    print("=" * 80)
    print("TEST 4: PyMOL Coordinate System")
    print("=" * 80)
    
    # PyMOL's coordinate system:
    # - Origin at center of view
    # - X axis: right
    # - Y axis: up
    # - Z axis: out of screen (toward viewer)
    # - When rendering, we're looking down -Z (into the screen)
    
    # In the projection image:
    # - Origin at center (after centering)
    # - X axis: right
    # - Y axis: up (in image coordinates, but might be flipped for display)
    
    # Matplotlib uses bottom-left origin with Y up
    # So if we have a 3D point [x, y, z] in view space:
    # - Image X = x (right)
    # - Image Y = y (up)
    
    # But the code uses:
    # - Image X = rotated[1] (Y component)
    # - Image Y = rotated[0] (X component)
    
    # This suggests either:
    # 1. The rotation matrix convention is different
    # 2. There's a coordinate swap in PyMOL's rendering
    # 3. The projection plane is different (YZ plane instead of XY?)
    
    print("PyMOL coordinate system analysis:")
    print("- View space: looking down -Z axis")
    print("- Projection plane: XY plane (Z=0)")
    print("- Image coordinates: X right, Y up")
    print()
    print("Current code mapping:")
    print("- Image X = rotated[1] (Y component in view space)")
    print("- Image Y = rotated[0] (X component in view space)")
    print()
    print("This suggests a coordinate swap or different projection plane!")
    print()


def test_effective_pixel_size_calculation():
    """Test the effective pixel size calculation."""
    print("=" * 80)
    print("TEST 5: Effective Pixel Size Calculation")
    print("=" * 80)
    
    # From debug output:
    model_size = 232.11  # Å
    rotated_bbox_size = 299.76  # Å (max dist from center, doubled)
    projection_size = 251  # pixels
    pixel_size = 1.1060  # Å/pixel (alignment)
    
    # Current calculation:
    effective_pixel_size = rotated_bbox_size / (projection_size / 1.2)
    print(f"Model size: {model_size:.2f} Å")
    print(f"Rotated bbox size: {rotated_bbox_size:.2f} Å")
    print(f"Projection size: {projection_size} pixels")
    print(f"Alignment pixel size: {pixel_size:.4f} Å/pixel")
    print(f"Effective pixel size: {effective_pixel_size:.4f} Å/pixel")
    print()
    
    # The effective pixel size should represent the actual scale in the rendered image
    # If PyMOL scales the rotated bbox to fit in projection_size with 1.2x padding:
    # projection_size pixels = rotated_bbox_size * 1.2 Angstroms
    # So: 1 pixel = (rotated_bbox_size * 1.2) / projection_size Angstroms
    # Therefore: effective_pixel_size = (rotated_bbox_size * 1.2) / projection_size
    
    effective_pixel_size_alt = (rotated_bbox_size * 1.2) / projection_size
    print(f"Alternative calculation: {effective_pixel_size_alt:.4f} Å/pixel")
    print(f"Current calculation: {effective_pixel_size:.4f} Å/pixel")
    print(f"Match: {np.isclose(effective_pixel_size, effective_pixel_size_alt)}")
    print()


def test_marker_vs_com_transformation():
    """Compare marker and COM transformations to find inconsistencies."""
    print("=" * 80)
    print("TEST 6: Marker vs COM Transformation Comparison")
    print("=" * 80)
    
    euler = np.array([1.594551, -1.319266, -0.858527])
    R = euler_to_rotation_matrix(euler, convention='ZYZ')
    
    # COM from ChimeraX
    chimerax_com = np.array([246.62, 222.54, 307.65])
    structure_center = np.array([247.03983, 222.27774, 307.78766])
    com_centered = chimerax_com - structure_center
    com_rotated = R @ com_centered
    
    # Marker 1
    marker1_model = np.array([209.0, 116.0, 285.0])
    marker1_centered = marker1_model - structure_center
    marker1_rotated = R @ marker1_centered
    
    print("COM transformation:")
    print(f"  Centered: {com_centered}")
    print(f"  Rotated: {com_rotated}")
    print()
    
    print("Marker 1 transformation:")
    print(f"  Centered: {marker1_centered}")
    print(f"  Rotated: {marker1_rotated}")
    print()
    
    # COM uses: com_x = rotated[1] / pixel_size, com_y = -rotated[0] / pixel_size
    # Markers use: marker_x = rotated[1] / effective_pixel_size, marker_y = rotated[0] / effective_pixel_size
    
    pixel_size = 1.1060
    effective_pixel_size = 1.4331
    
    com_x = com_rotated[1] / pixel_size
    com_y = -com_rotated[0] / pixel_size
    
    marker_x = marker1_rotated[1] / effective_pixel_size
    marker_y = marker1_rotated[0] / effective_pixel_size
    
    print("COM pixel coordinates:")
    print(f"  X = rotated[1] / pixel_size = {com_rotated[1]:.2f} / {pixel_size:.4f} = {com_x:.2f}")
    print(f"  Y = -rotated[0] / pixel_size = -{com_rotated[0]:.2f} / {pixel_size:.4f} = {com_y:.2f}")
    print()
    
    print("Marker pixel coordinates:")
    print(f"  X = rotated[1] / effective_pixel_size = {marker1_rotated[1]:.2f} / {effective_pixel_size:.4f} = {marker_x:.2f}")
    print(f"  Y = rotated[0] / effective_pixel_size = {marker1_rotated[0]:.2f} / {effective_pixel_size:.4f} = {marker_y:.2f}")
    print()
    
    # From debug: COM offset from projection center: (0.09, 0.44)
    # This is very small, suggesting COM calculation is correct
    # But markers need -70 pixel X offset, suggesting marker calculation is wrong
    
    print("Key observation:")
    print("- COM uses pixel_size and negates Y: works correctly (offset ~0.4 pixels)")
    print("- Markers use effective_pixel_size and don't negate Y: needs -70 pixel X offset")
    print("- This suggests the issue is in the marker transformation, not the COM")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COORDINATE MAPPING TEST SUITE")
    print("=" * 80 + "\n")
    
    test_rotation_matrix_properties()
    test_coordinate_system_conventions()
    test_projection_coordinate_mapping()
    test_pymol_coordinate_system()
    test_effective_pixel_size_calculation()
    test_marker_vs_com_transformation()
    
    print("=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)

