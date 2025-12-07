#!/usr/bin/env python3
"""
Critical test: Projection size is calculated from ORIGINAL model_size,
but PyMOL's zoom scales based on ROTATED bbox_size.

This mismatch could explain the -70 pixel X offset!
"""

import numpy as np

# From debug output
model_size = 232.11  # Å (original model bounding box)
rotated_bbox_size = 299.76  # Å (max distance from center, doubled)
pixel_size = 1.1060  # Å/pixel (alignment pixel size)

# How projection_size is calculated (line 1220)
projection_size = int((model_size / pixel_size) * 1.2)
print("=" * 80)
print("PROJECTION SIZE CALCULATION MISMATCH")
print("=" * 80)
print()
print(f"Original model size: {model_size:.2f} Å")
print(f"Rotated bbox size: {rotated_bbox_size:.2f} Å")
print(f"Pixel size: {pixel_size:.4f} Å/pixel")
print()
print(f"Projection size calculation:")
print(f"  projection_size = (model_size / pixel_size) * 1.2")
print(f"  projection_size = ({model_size:.2f} / {pixel_size:.4f}) * 1.2")
print(f"  projection_size = {projection_size} pixels")
print()

# PyMOL's zoom(complete=1) scales based on rotated_bbox_size
# It fits the rotated bbox into projection_size with 1.2x padding
# So the actual scale is:
actual_scale_pixels = projection_size / 1.2  # Pixels available for structure
actual_pixel_size = rotated_bbox_size / actual_scale_pixels

print(f"PyMOL's zoom scaling:")
print(f"  PyMOL fits {rotated_bbox_size:.2f} Å into {actual_scale_pixels:.1f} pixels")
print(f"  Actual pixel size = {rotated_bbox_size:.2f} / {actual_scale_pixels:.1f} = {actual_pixel_size:.4f} Å/pixel")
print()

# But if projection_size was calculated for rotated_bbox_size:
correct_projection_size = int((rotated_bbox_size / pixel_size) * 1.2)
print(f"If projection_size was calculated for rotated_bbox_size:")
print(f"  correct_projection_size = ({rotated_bbox_size:.2f} / {pixel_size:.4f}) * 1.2")
print(f"  correct_projection_size = {correct_projection_size} pixels")
print(f"  Difference: {correct_projection_size - projection_size} pixels")
print()

# The issue: projection_size is too small!
# PyMOL has to fit a larger structure (rotated_bbox) into a smaller image
# This causes the structure to be scaled down more than expected
# The scale factor difference:
scale_factor_ratio = rotated_bbox_size / model_size
print(f"Scale factor ratio:")
print(f"  rotated_bbox_size / model_size = {rotated_bbox_size:.2f} / {model_size:.2f} = {scale_factor_ratio:.3f}")
print(f"  The rotated bbox is {scale_factor_ratio:.1%} larger than the original model")
print()

# If we used the correct projection_size, the pixel size would be:
correct_scale_pixels = correct_projection_size / 1.2
correct_pixel_size = rotated_bbox_size / correct_scale_pixels
print(f"With correct projection_size:")
print(f"  PyMOL would fit {rotated_bbox_size:.2f} Å into {correct_scale_pixels:.1f} pixels")
print(f"  Correct pixel size = {rotated_bbox_size:.2f} / {correct_scale_pixels:.1f} = {correct_pixel_size:.4f} Å/pixel")
print()

# The offset calculation
# If a marker is at X = -3.17 Å in view space:
marker_x_angstroms = -3.17
current_pixel_x = marker_x_angstroms / actual_pixel_size
correct_pixel_x = marker_x_angstroms / correct_pixel_size
offset = current_pixel_x - correct_pixel_x

print(f"Marker X coordinate example:")
print(f"  Marker X in view space: {marker_x_angstroms:.2f} Å")
print(f"  Current pixel X (with wrong projection_size): {marker_x_angstroms:.2f} / {actual_pixel_size:.4f} = {current_pixel_x:.2f} pixels")
print(f"  Correct pixel X (with correct projection_size): {marker_x_angstroms:.2f} / {correct_pixel_size:.4f} = {correct_pixel_x:.2f} pixels")
print(f"  Offset: {offset:.2f} pixels")
print()

# But wait - we're using effective_pixel_size which accounts for this!
effective_pixel_size = rotated_bbox_size / (projection_size / 1.2)
print(f"Current effective_pixel_size calculation:")
print(f"  effective_pixel_size = rotated_bbox_size / (projection_size / 1.2)")
print(f"  effective_pixel_size = {rotated_bbox_size:.2f} / ({projection_size} / 1.2)")
print(f"  effective_pixel_size = {effective_pixel_size:.4f} Å/pixel")
print(f"  This matches actual_pixel_size: {np.isclose(effective_pixel_size, actual_pixel_size)}")
print()

# So the effective_pixel_size is correct!
# But why is there still a -70 pixel offset?

# Hypothesis: Maybe the issue is that PyMOL's zoom doesn't use the full projection_size?
# Or maybe there's a coordinate system offset in how PyMOL renders the image?

# Let's check: if the offset is -70 pixels, and projection_size is 251 pixels
# The offset is -70/251 = -27.9% of the projection size
# This is close to the scale factor ratio: (rotated_bbox_size - model_size) / model_size
scale_difference_ratio = (rotated_bbox_size - model_size) / model_size
print(f"Scale difference analysis:")
print(f"  (rotated_bbox_size - model_size) / model_size = {scale_difference_ratio:.3f}")
print(f"  -70 pixels / 251 pixels = {-70/251:.3f}")
print(f"  These are similar! The offset might be related to the size difference!")
print()

# Actually, wait. The -70 pixel offset is in the X direction only.
# This suggests it's not about the overall scale, but about X specifically.
# Maybe PyMOL's zoom uses a different method for X vs Y?

print("CONCLUSION:")
print("The projection_size is calculated from the original model_size,")
print("but PyMOL's zoom scales based on the rotated_bbox_size.")
print("However, we account for this with effective_pixel_size.")
print("The -70 pixel X offset suggests a coordinate system issue,")
print("possibly related to how PyMOL renders or centers the image.")

