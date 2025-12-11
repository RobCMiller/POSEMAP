#!/usr/bin/env python3
"""
POSEMAP - Core Library
Pose-Oriented Single-particle EM Micrograph Annotation & Projection

This module provides core functions for:
1. Loading particle data from cryoSPARC .cs files
2. Matching particles between refinement and passthrough files
3. Projecting 3D volumes and atomic structures at particle orientations
4. Coordinate transformations and visualization utilities
"""

import numpy as np
import mrcfile
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.ndimage import rotate, zoom
from scipy.spatial.distance import cdist
from typing import Tuple, Optional, Dict, List
try:
    from Bio.PDB import PDBParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    import warnings
    warnings.simplefilter('ignore', PDBConstructionWarning)
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


def load_cs_file(cs_file_path: str) -> np.ndarray:
    """Load a cryosparc .cs file as a numpy structured array."""
    return np.load(cs_file_path, allow_pickle=True)


def match_particles(refinement_cs: np.ndarray, passthrough_cs: np.ndarray) -> Dict:
    """
    Match particles between refinement and passthrough .cs files by UID.
    
    Returns a dictionary with matched particle data including:
    - poses (Euler angles from refinement)
    - shifts (2D shifts from refinement)
    - micrograph_paths
    - center_x_frac, center_y_frac (fractional coordinates)
    - micrograph_shapes
    """
    # Create mapping from UID to index in each file
    refinement_uids = refinement_cs['uid']
    passthrough_uids = passthrough_cs['uid']
    
    # Find matching indices
    refinement_to_passthrough = {}
    passthrough_to_refinement = {}
    
    # Create dictionaries for fast lookup
    passthrough_uid_to_idx = {uid: idx for idx, uid in enumerate(passthrough_uids)}
    
    matched_indices = []
    for ref_idx, uid in enumerate(refinement_uids):
        if uid in passthrough_uid_to_idx:
            passthrough_idx = passthrough_uid_to_idx[uid]
            matched_indices.append((ref_idx, passthrough_idx))
            refinement_to_passthrough[ref_idx] = passthrough_idx
            passthrough_to_refinement[passthrough_idx] = ref_idx
    
    print(f"Matched {len(matched_indices)} particles out of {len(refinement_cs)}")
    
    # Extract matched data
    matched_data = {
        'refinement_indices': [m[0] for m in matched_indices],
        'passthrough_indices': [m[1] for m in matched_indices],
        'uids': refinement_uids[[m[0] for m in matched_indices]],
        'poses': refinement_cs['alignments3D/pose'][[m[0] for m in matched_indices]],
        'shifts': refinement_cs['alignments3D/shift'][[m[0] for m in matched_indices]],
        'pixel_size': refinement_cs['alignments3D/psize_A'][[m[0] for m in matched_indices]],
        'micrograph_paths': passthrough_cs['location/micrograph_path'][[m[1] for m in matched_indices]],
        'center_x_frac': passthrough_cs['location/center_x_frac'][[m[1] for m in matched_indices]],
        'center_y_frac': passthrough_cs['location/center_y_frac'][[m[1] for m in matched_indices]],
        'micrograph_shapes': passthrough_cs['location/micrograph_shape'][[m[1] for m in matched_indices]],
        'micrograph_psize': passthrough_cs['location/micrograph_psize_A'][[m[1] for m in matched_indices]],
    }
    
    return matched_data


def load_volume(volume_path: str) -> Tuple[np.ndarray, float]:
    """Load a 3D volume from an .mrc file and return the volume and pixel size."""
    with mrcfile.open(volume_path) as mrc:
        volume = mrc.data.astype(np.float32)
        pixel_size = float(mrc.voxel_size.x)  # Angstroms per pixel
    return volume, pixel_size


def euler_to_rotation_matrix(euler_angles: np.ndarray, convention: str = 'ZYZ') -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.
    
    Cryosparc typically uses ZYZ convention (phi, theta, psi).
    """
    # Create rotation object and convert to matrix
    rot = Rotation.from_euler(convention, euler_angles, degrees=False)
    return rot.as_matrix()


def project_volume(volume: np.ndarray, euler_angles: np.ndarray, 
                   output_size: Optional[Tuple[int, int]] = None,
                   pixel_size: float = 1.0,
                   half_size: Optional[float] = None,
                   rotation_correction_x: float = 0.0,
                   rotation_correction_y: float = 0.0,
                   rotation_correction_z: float = 0.0) -> np.ndarray:
    """
    Project a 3D volume at given Euler angles.
    
    Uses a more accurate approach: rotates coordinates and samples the volume.
    
    Parameters:
    -----------
    volume : np.ndarray
        3D volume array
    euler_angles : np.ndarray
        Euler angles [phi, theta, psi] in radians (ZYZ convention)
    output_size : tuple, optional
        Output projection size (height, width). If None, uses volume size.
    pixel_size : float
        Pixel size in Angstroms (for consistency, though not used in rotation)
        
    Returns:
    --------
    np.ndarray : 2D projection
    """
    if output_size is None:
        output_size = (volume.shape[1], volume.shape[2])
    
    # Get rotation matrix
    R = euler_to_rotation_matrix(euler_angles, convention='ZYZ')
    
    # Apply rotation corrections to match PyMOL's behavior
    # PyMOL applies corrections in XYZ convention (intrinsic rotations)
    # In PyMOL: R_transform = R @ R_correction (for transform_object)
    # This means: first apply R_correction to the object, then R
    # For coordinate transformation (rotating coordinates), we need the inverse:
    #   (R @ R_correction)^T = R_correction^T @ R^T
    # But since we're using R to transform coordinates (not objects), we need:
    #   R_final = R @ R_correction (same as PyMOL's transform_object)
    # Then for coordinate transformation: R_inv = (R @ R_correction)^T
    if abs(rotation_correction_x) > 1e-6 or abs(rotation_correction_y) > 1e-6 or abs(rotation_correction_z) > 1e-6:
        from scipy.spatial.transform import Rotation as Rot
        rot_x_rad = np.deg2rad(rotation_correction_x)
        rot_y_rad = np.deg2rad(rotation_correction_y)
        rot_z_rad = np.deg2rad(rotation_correction_z)
        rot_correction = Rot.from_euler('XYZ', [rot_x_rad, rot_y_rad, rot_z_rad], degrees=False)
        R_correction = rot_correction.as_matrix()
        # Apply correction AFTER the main rotation (same as PyMOL: R @ R_correction)
        # This means: first R_correction, then R (when applied to object)
        # For coordinate transformation, we use R directly (it will be inverted later)
        R = R @ R_correction
    
    # Debug: Print rotation matrix to verify it's different for different angles
    print(f"  DEBUG project_volume: Euler=[{euler_angles[0]:.6f}, {euler_angles[1]:.6f}, {euler_angles[2]:.6f}]")
    print(f"  DEBUG project_volume: Rotation corrections: X={rotation_correction_x:.2f}°, Y={rotation_correction_y:.2f}°, Z={rotation_correction_z:.2f}°")
    print(f"  DEBUG project_volume: R[0,0]={R[0,0]:.6f}, R[0,1]={R[0,1]:.6f}, R[0,2]={R[0,2]:.6f}")
    
    # Create coordinate grid for output projection
    h, w = output_size
    center_out = np.array([h/2, w/2])
    
    # Create 2D grid for projection plane
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(float)
    y_coords -= center_out[0]
    x_coords -= center_out[1]
    
    # Scale coordinates to match the physical size
    # The projection output is in pixels, each representing pixel_size Angstroms
    # So: projection_coord_in_angstroms = projection_coord_in_pixels * pixel_size
    # The volume's coordinate system is in Angstroms, centered at origin
    # The volume grid spans from -half_size to +half_size in Angstroms
    # We need to scale the projection coordinates to match this physical extent
    # The projection should show the same physical size as the micrograph
    # So scale by pixel_size to convert pixels to Angstroms
    y_coords *= pixel_size
    x_coords *= pixel_size
    
    # Create 3D coordinates (z=0 for projection plane)
    # IMPORTANT: In cryo-EM, the beam direction is typically along the Z-axis
    # The projection plane is the XY plane (z=0)
    # Coordinates are in Angstroms, centered at origin in projection plane
    # Stack as [x, y, z] where z=0 for all projection plane points
    coords_2d = np.stack([x_coords.flatten(), y_coords.flatten(), 
                          np.zeros(h*w)], axis=1)
    
    # Rotate coordinates back to volume space
    # The rotation matrix R rotates from volume space to view space
    # So we need the inverse (which is the transpose for rotation matrices)
    # This transforms view space coordinates to volume space coordinates
    R_inv = R.T
    coords_3d = (R_inv @ coords_2d.T).T
    
    # Debug: Check a sample of rotated coordinates to verify rotation is working
    sample_idx = h * w // 2  # Middle pixel
    corner_idx = 0  # Top-left corner
    center_idx = h * w // 2 + w // 2  # Center pixel
    print(f"  DEBUG project_volume: Sample coord before rotation (corner): ({coords_2d[corner_idx,0]:.2f}, {coords_2d[corner_idx,1]:.2f}, {coords_2d[corner_idx,2]:.2f})")
    print(f"  DEBUG project_volume: Sample coord after rotation (corner): ({coords_3d[corner_idx,0]:.2f}, {coords_3d[corner_idx,1]:.2f}, {coords_3d[corner_idx,2]:.2f})")
    print(f"  DEBUG project_volume: Sample coord before rotation (center): ({coords_2d[center_idx,0]:.2f}, {coords_2d[center_idx,1]:.2f}, {coords_2d[center_idx,2]:.2f})")
    print(f"  DEBUG project_volume: Sample coord after rotation (center): ({coords_3d[center_idx,0]:.2f}, {coords_3d[center_idx,1]:.2f}, {coords_3d[center_idx,2]:.2f})")
    
    # Convert from Angstroms (centered at origin) to voxel coordinates
    # Volume grid spans from -half_size to +half_size in Angstroms
    # Voxel 0 = -half_size, voxel (grid_size-1) = +half_size
    # The grid was created with: np.linspace(-half_size, half_size, grid_size)
    # So grid_spacing = 2*half_size / (grid_size - 1)
    # To convert: voxel = (angstrom + half_size) / grid_spacing
    vol_center_voxels = np.array([volume.shape[2]/2, volume.shape[1]/2, volume.shape[0]/2])
    grid_size = volume.shape[0]  # Assuming cubic volume
    
    if half_size is not None:
        # Use provided half_size for accurate conversion
        # The grid spans from -half_size to +half_size in Angstroms
        # Voxel 0 corresponds to -half_size, voxel (grid_size-1) corresponds to +half_size
        # Spacing between voxels: (2 * half_size) / (grid_size - 1)
        grid_spacing = 2.0 * half_size / (grid_size - 1)
        # Convert from Angstroms (centered at origin) to voxel coordinates
        # Add half_size to shift from [-half_size, +half_size] to [0, 2*half_size]
        # Then divide by spacing to get voxel index
        coords_3d_voxels = (coords_3d + half_size) / grid_spacing
        
        # Debug: Check voxel coordinate ranges
        print(f"  DEBUG project_volume: Voxel coord ranges: x=[{coords_3d_voxels[:, 0].min():.1f}, {coords_3d_voxels[:, 0].max():.1f}], "
              f"y=[{coords_3d_voxels[:, 1].min():.1f}, {coords_3d_voxels[:, 1].max():.1f}], "
              f"z=[{coords_3d_voxels[:, 2].min():.1f}, {coords_3d_voxels[:, 2].max():.1f}]")
        print(f"  DEBUG project_volume: Volume shape: {volume.shape}, half_size={half_size:.2f}, grid_spacing={grid_spacing:.4f}")
        # Debug: Check center voxel coordinate to see if rotation changes it
        center_voxel = coords_3d_voxels[h * w // 2]
        print(f"  DEBUG project_volume: Center voxel coord: ({center_voxel[0]:.1f}, {center_voxel[1]:.1f}, {center_voxel[2]:.1f})")
    else:
        # Fallback: estimate using pixel_size (approximate)
        coords_3d_voxels = coords_3d / pixel_size + vol_center_voxels
    
    coords_3d = coords_3d_voxels
    
    # Sample volume using trilinear interpolation
    projection = np.zeros(h * w, dtype=volume.dtype)
    
    # Get integer and fractional parts
    x0 = np.floor(coords_3d[:, 0]).astype(int)
    y0 = np.floor(coords_3d[:, 1]).astype(int)
    z0 = np.floor(coords_3d[:, 2]).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    
    # Fractional parts
    xd = coords_3d[:, 0] - x0
    yd = coords_3d[:, 1] - y0
    zd = coords_3d[:, 2] - z0
    
    # Clamp coordinates to valid range (volume shape is [z, y, x])
    # Debug: Check how many coordinates are out of bounds
    x0_clipped = np.clip(x0, 0, volume.shape[2] - 1)
    x1_clipped = np.clip(x1, 0, volume.shape[2] - 1)
    y0_clipped = np.clip(y0, 0, volume.shape[1] - 1)
    y1_clipped = np.clip(y1, 0, volume.shape[1] - 1)
    z0_clipped = np.clip(z0, 0, volume.shape[0] - 1)
    z1_clipped = np.clip(z1, 0, volume.shape[0] - 1)
    
    # Count out-of-bounds coordinates
    out_of_bounds = np.sum((x0 < 0) | (x0 >= volume.shape[2]) | 
                          (y0 < 0) | (y0 >= volume.shape[1]) |
                          (z0 < 0) | (z0 >= volume.shape[0]))
    if out_of_bounds > 0:
        print(f"  DEBUG project_volume: {out_of_bounds} coordinates out of bounds (out of {len(x0)})")
    
    x0, x1 = x0_clipped, x1_clipped
    y0, y1 = y0_clipped, y1_clipped
    z0, z1 = z0_clipped, z1_clipped
    
    # Trilinear interpolation (volume is indexed as [z, y, x] in numpy)
    c000 = volume[z0, y0, x0]
    c001 = volume[z1, y0, x0]
    c010 = volume[z0, y1, x0]
    c011 = volume[z1, y1, x0]
    c100 = volume[z0, y0, x1]
    c101 = volume[z1, y0, x1]
    c110 = volume[z0, y1, x1]
    c111 = volume[z1, y1, x1]
    
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    
    projection = c0 * (1 - zd) + c1 * zd
    
    # Reshape to 2D
    projection = projection.reshape(h, w)
    
    # Debug: Check projection statistics
    print(f"  DEBUG project_volume: Projection stats: min={projection.min():.4f}, max={projection.max():.4f}, mean={projection.mean():.4f}, std={projection.std():.4f}")
    print(f"  DEBUG project_volume: Non-zero pixels: {np.sum(projection > 0)} out of {h * w}")
    
    # Debug: Check if rotation is actually changing the sampled region
    # Print a hash of the first 100 pixels to see if projections differ
    sample_pixels = projection.flatten()[:100]
    pixel_hash = hash(tuple(sample_pixels.round(decimals=4)))
    print(f"  DEBUG project_volume: First 100 pixels hash: {pixel_hash}")
    
    # IMPORTANT: Flip vertically to match display orientation
    # The projection is generated with y=0 at top (standard image convention)
    # But we display with origin='lower' (y=0 at bottom), so flip vertically
    projection = np.flipud(projection)
    
    return projection


def pdb_to_density_map(pdb_data: Dict, pixel_size: float = 1.0, 
                       grid_size: Optional[int] = None,
                       atom_radius: float = 2.0) -> Tuple[np.ndarray, float, float]:
    """
    Convert PDB atomic coordinates to a 3D density map.
    
    Voxelizes atoms as Gaussian blobs to create a density map suitable for EM simulation.
    Uses an optimized approach with sparse representation and scipy gaussian filtering.
    
    Parameters:
    -----------
    pdb_data : dict
        PDB structure data from load_pdb_structure()
    pixel_size : float
        Pixel size in Angstroms for the density map
    grid_size : int, optional
        Size of the cubic grid. If None, auto-calculated from structure extent
    atom_radius : float
        Radius of atoms in Angstroms for Gaussian blob (default 2.0 Å)
        
    Returns:
    --------
    volume : np.ndarray
        3D density map as [z, y, x] array
    pixel_size : float
        Pixel size used (same as input)
    """
    coords = pdb_data['coords']
    
    # Ensure coords is a numpy array
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    
    if len(coords) == 0:
        raise ValueError("No coordinates in PDB data")
    
    # Verify all atoms are included (protein and nucleic acid)
    # pdb_data['coords'] contains ALL atoms from ALL chains loaded by load_pdb_structure()
    # No filtering is applied - both protein and nucleic acid chains are included
    num_atoms = len(coords)
    if 'chain_ids' in pdb_data:
        unique_chains = len(np.unique(pdb_data['chain_ids']))
        print(f"  DEBUG pdb_to_density_map: Processing {num_atoms} atoms from {unique_chains} chains (all chains included)")
    else:
        print(f"  DEBUG pdb_to_density_map: Processing {num_atoms} atoms (all atoms included)")
    
    # Calculate bounding box
    # IMPORTANT: Center the structure at origin before creating density map
    # This ensures rotations are applied correctly
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    extent = max_coords - min_coords
    center = (min_coords + max_coords) / 2.0
    
    # Center coordinates at origin
    coords_centered = coords - center
    
    # Auto-calculate grid size if not provided
    if grid_size is None:
        # Add padding (40% on each side)
        max_extent = np.max(extent) * 1.8
        grid_size = int(np.ceil(max_extent / pixel_size))
        # Make it a reasonable size (round up to nearest 32 for efficiency)
        grid_size = ((grid_size + 31) // 32) * 32
        # Limit maximum size for performance, but allow larger for better quality
        grid_size = min(grid_size, 512)  # Max 512^3 for reasonable speed
    
    # Create coordinate grid centered at origin (since we centered the coordinates)
    # The grid should span the structure with some padding
    # Calculate the actual extent we need to cover
    max_extent = np.max(extent) * 1.8  # Same as used for grid_size calculation
    padding = max_extent * 0.2  # 20% padding on each side
    half_size = (max_extent / 2.0) + padding
    # Grid is centered at origin (0, 0, 0) since coords are centered
    x = np.linspace(-half_size, half_size, grid_size)
    y = np.linspace(-half_size, half_size, grid_size)
    z = np.linspace(-half_size, half_size, grid_size)
    
    # Initialize volume
    volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    # Convert atom coordinates to grid indices
    # Use centered coordinates
    # Much faster: use vectorized operations
    x_indices = np.clip(np.round((coords_centered[:, 0] - x[0]) / (x[1] - x[0])).astype(int), 0, grid_size - 1)
    y_indices = np.clip(np.round((coords_centered[:, 1] - y[0]) / (y[1] - y[0])).astype(int), 0, grid_size - 1)
    z_indices = np.clip(np.round((coords_centered[:, 2] - z[0]) / (z[1] - z[0])).astype(int), 0, grid_size - 1)
    
    # Add atoms to volume (simple point placement)
    # Use bincount for fast accumulation
    flat_indices = z_indices * grid_size * grid_size + y_indices * grid_size + x_indices
    # Ensure flat_indices is a numpy array
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    # Use ravel() instead of .flat for np.add.at
    volume_flat = volume.ravel()
    np.add.at(volume_flat, flat_indices, 1.0)
    volume = volume_flat.reshape(volume.shape)
    
    # Apply Gaussian filter to create smooth density blobs
    # This is much faster than computing Gaussian for each atom individually
    from scipy.ndimage import gaussian_filter
    sigma = atom_radius / pixel_size  # Convert radius to pixels
    volume = gaussian_filter(volume, sigma=sigma, mode='constant', cval=0.0)
    
    # Return volume, pixel_size, and half_size (for coordinate conversion)
    return volume, pixel_size, half_size


def simulate_em_projection_from_pdb_eman2(pdb_data: Dict, euler_angles: np.ndarray,
                                          output_size: Tuple[int, int],
                                          pixel_size: float = 1.0,
                                          rotation_correction_x: float = 0.0,
                                          rotation_correction_y: float = 0.0,
                                          rotation_correction_z: float = 0.0) -> np.ndarray:
    """
    Simulate an EM projection using EMAN2.
    
    This function uses EMAN2's projection capabilities for more accurate EM simulation.
    
    WARNING: This function currently has known orientation issues. The projection
    orientation does not match the NumPy implementation. Use the NumPy method
    (use_eman2=False) for correct orientation matching.
    
    Parameters:
    -----------
    pdb_data : dict
        PDB structure data from load_pdb_structure()
    euler_angles : np.ndarray
        Euler angles [phi, theta, psi] in radians (ZYZ convention)
    output_size : tuple
        Output projection size (height, width) in pixels
    pixel_size : float
        Pixel size in Angstroms
    rotation_correction_x : float
        Rotation correction around X axis in degrees
    rotation_correction_y : float
        Rotation correction around Y axis in degrees
    rotation_correction_z : float
        Rotation correction around Z axis in degrees
    """
    try:
        from EMAN2 import EMData, Transform
        import numpy as np
    except ImportError:
        raise ImportError("EMAN2 is not available. Please install EMAN2 or use the fallback method.")
    
    # Convert PDB to density map first
    # For better resolution, calculate appropriate grid_size based on desired output size
    # We want the volume to be large enough for good quality, but not so large it's too slow
    h, w = output_size
    min_grid_size = max(h, w)
    
    # For very large output sizes (e.g., 2x resolution for downsampling), 
    # we don't need the volume to be as large - the final resize will handle quality
    # Use a more conservative approach: use output size directly, but cap reasonably
    if min_grid_size > 800:
        # For large outputs (likely 2x resolution that will be downsampled),
        # use a fixed reasonable size (640-768 range)
        desired_grid_size = 640  # Good balance of quality and speed
    else:
        # For normal sizes, use output size directly (no 1.2x multiplier)
        desired_grid_size = min_grid_size
    
    # Round up to nearest 32 for efficiency
    desired_grid_size = ((desired_grid_size + 31) // 32) * 32
    # Cap at 768 for reasonable performance (640^3 = 262M voxels, 768^3 = 453M voxels)
    # This is much faster than 1024^3 (1B voxels) while still providing excellent quality
    desired_grid_size = min(desired_grid_size, 768)
    print(f"  DEBUG EMAN2: Creating density map with grid_size={desired_grid_size} (output size={h}x{w})...")
    volume, _, half_size = pdb_to_density_map(pdb_data, pixel_size=pixel_size, atom_radius=2.0, grid_size=desired_grid_size)
    print(f"  DEBUG EMAN2: Density map created, shape={volume.shape}, converting to EMData...")
    
    # Convert numpy array to EMAN2 EMData
    # EMAN2 expects [nx, ny, nz] format (x, y, z)
    # Volume is in [z, y, x] format, so transpose to [x, y, z]
    volume_xyz = volume.transpose(2, 1, 0).astype(np.float32)  # Back to [x, y, z]
    # Ensure contiguous array
    if not volume_xyz.flags['C_CONTIGUOUS']:
        volume_xyz = np.ascontiguousarray(volume_xyz)
    
    em_volume = EMData()
    em_volume.set_size(volume_xyz.shape[0], volume_xyz.shape[1], volume_xyz.shape[2])  # [nx, ny, nz]
    print(f"  DEBUG EMAN2: Setting volume data ({volume_xyz.nbytes / 1e6:.1f} MB)...")
    em_volume.set_data_string(volume_xyz.tobytes())
    print(f"  DEBUG EMAN2: Volume data set, creating transform and projecting...")
    
    # Build R matrix same as NumPy, then use R.T (matching NumPy's coordinate transformation)
    R = euler_to_rotation_matrix(euler_angles, convention='ZYZ')
    
    # Apply rotation corrections (same as NumPy)
    if abs(rotation_correction_x) > 1e-6 or abs(rotation_correction_y) > 1e-6 or abs(rotation_correction_z) > 1e-6:
        from scipy.spatial.transform import Rotation as Rot
        rot_x_rad = np.deg2rad(rotation_correction_x)
        rot_y_rad = np.deg2rad(rotation_correction_y)
        rot_z_rad = np.deg2rad(rotation_correction_z)
        rot_correction = Rot.from_euler('XYZ', [rot_x_rad, rot_y_rad, rot_z_rad], degrees=False)
        R_correction = rot_correction.as_matrix()
        # Combine rotations: R_final = R @ R_correction (same as NumPy)
        R = R @ R_correction
    
    # Try using R.T with NO negation and NO inverse
    # Convert R.T to Euler angles for EMAN2
    R_for_eman2 = R.T  # Use R.T
    from scipy.spatial.transform import Rotation as Rot
    rot_from_matrix = Rot.from_matrix(R_for_eman2)
    euler_zyz = rot_from_matrix.as_euler('ZYZ', degrees=False)
    
    # EMAN2 uses [az, alt, phi] = [phi, theta, psi] in ZYZ convention
    # IMPORTANT: Convert numpy types to Python floats for EMAN2
    transform = Transform({"type": "eman", 
                          "az": float(euler_zyz[0]),   # phi
                          "alt": float(euler_zyz[1]),  # theta  
                          "phi": float(euler_zyz[2])}) # psi
    
    # Use inverse transform
    transform = transform.inverse()
    
    print(f"  DEBUG EMAN2: Input Euler angles: [{euler_angles[0]:.6f}, {euler_angles[1]:.6f}, {euler_angles[2]:.6f}]")
    print(f"  DEBUG EMAN2: R.T Euler angles: [{euler_zyz[0]:.6f}, {euler_zyz[1]:.6f}, {euler_zyz[2]:.6f}]")
    print(f"  DEBUG EMAN2: Using R.T Euler angles with inverse transform")
    
    # Project the volume (projection will be same size as volume's x,y dimensions)
    print(f"  DEBUG EMAN2: Projecting volume (this may take a moment for large volumes)...")
    projection = em_volume.project("standard", transform)
    print(f"  DEBUG EMAN2: Projection complete, converting to numpy array...")
    
    # Convert to numpy array
    proj_array = projection.numpy().copy()
    
    # Resize to desired output size if needed using scipy
    h, w = output_size
    proj_h, proj_w = proj_array.shape
    if proj_h != h or proj_w != w:
        from scipy.ndimage import zoom
        zoom_factor_h = h / proj_h
        zoom_factor_w = w / proj_w
        proj_array = zoom(proj_array, (zoom_factor_h, zoom_factor_w), order=1)
    
    # Try vertical flip only with inverse transform (no transpose, no horizontal flip)
    proj_array = np.flipud(proj_array)  # Flip vertically (top-to-bottom, like flipping pancake from bottom)
    
    return proj_array


def simulate_em_projection_from_pdb(pdb_data: Dict, euler_angles: np.ndarray,
                                    output_size: Tuple[int, int],
                                    pixel_size: float = 1.0,
                                    atom_radius: float = 2.0,
                                    use_eman2: bool = True,
                                    rotation_correction_x: float = 0.0,
                                    rotation_correction_y: float = 0.0,
                                    rotation_correction_z: float = 0.0) -> np.ndarray:
    """
    Simulate an EM projection from a PDB structure.
    
    Converts PDB to density map, then projects it at the given orientation.
    This produces a simulated EM image that can be compared to actual micrograph data.
    
    Parameters:
    -----------
    pdb_data : dict
        PDB structure data from load_pdb_structure()
    euler_angles : np.ndarray
        Euler angles [phi, theta, psi] in radians (ZYZ convention)
    output_size : tuple
        Output projection size (height, width) in pixels
    pixel_size : float
        Pixel size in Angstroms (same as micrograph pixel size)
    atom_radius : float
        Radius of atoms in Angstroms for density map generation
        
    Returns:
    --------
    np.ndarray : 2D projection (simulated EM image)
    """
    # Debug: Print Euler angles being used
    print(f"  DEBUG: Generating projection with Euler angles: [{euler_angles[0]:.6f}, {euler_angles[1]:.6f}, {euler_angles[2]:.6f}]")
    
    # Try EMAN2 first if requested and available
    if use_eman2:
        try:
            # Test if EMAN2 can be imported
            try:
                from EMAN2 import EMData, Transform
            except ImportError as import_err:
                print(f"EMAN2 not available (ImportError: {import_err}), falling back to NumPy projection method")
                use_eman2 = False
            except Exception as import_err:
                print(f"EMAN2 import failed (Error: {import_err}), falling back to NumPy projection method")
                use_eman2 = False
            
            if use_eman2:
                proj = simulate_em_projection_from_pdb_eman2(
                    pdb_data, euler_angles, output_size, pixel_size,
                    rotation_correction_x=rotation_correction_x,
                    rotation_correction_y=rotation_correction_y,
                    rotation_correction_z=rotation_correction_z
                )
                print(f"  DEBUG: EMAN2 projection generated, shape={proj.shape}, range=[{proj.min():.3f}, {proj.max():.3f}]")
                return proj
        except ImportError as e:
            print(f"EMAN2 not available (ImportError: {e}), falling back to NumPy projection method")
        except Exception as e:
            print(f"EMAN2 projection failed: {e}, falling back to NumPy method")
            import traceback
            traceback.print_exc()
    
    # Fallback to NumPy-based projection
    # Convert PDB to density map
    # Use the same pixel_size for the density map as the micrograph
    # This ensures the scale matches - each voxel in the density map represents
    # the same physical size (pixel_size Angstroms) as each pixel in the micrograph
    print(f"  DEBUG: Generating density map with pixel_size={pixel_size}, atom_radius={atom_radius}")
    volume, volume_pixel_size, half_size = pdb_to_density_map(pdb_data, pixel_size=pixel_size, atom_radius=atom_radius)
    print(f"  DEBUG: Density map generated, shape={volume.shape}, pixel_size={volume_pixel_size}, half_size={half_size:.2f}")
    
    # Verify pixel sizes match
    if abs(volume_pixel_size - pixel_size) > 0.001:
        print(f"Warning: Volume pixel size ({volume_pixel_size}) doesn't match requested ({pixel_size})")
    
    # Project the volume at the same pixel size
    # This ensures the projection scale matches the micrograph
    # The output_size is in pixels, and each pixel represents pixel_size Angstroms
    print(f"  DEBUG: Projecting volume with Euler angles: [{euler_angles[0]:.6f}, {euler_angles[1]:.6f}, {euler_angles[2]:.6f}]")
    projection = project_volume(volume, euler_angles, output_size=output_size, pixel_size=pixel_size, half_size=half_size,
                                rotation_correction_x=rotation_correction_x,
                                rotation_correction_y=rotation_correction_y,
                                rotation_correction_z=rotation_correction_z)
    print(f"  DEBUG: Projection generated, shape={projection.shape}, range=[{projection.min():.3f}, {projection.max():.3f}]")
    
    return projection


def get_particle_orientation_arrow(euler_angles: np.ndarray, length: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get arrow direction for visualizing particle viewing direction.
    
    This arrow represents the direction from which the particle is being viewed
    in the micrograph. Specifically, it shows the projection of the Z-axis of the
    model's coordinate system onto the micrograph plane after applying the particle's
    rotation (Euler angles). The Z-axis in the model coordinate system points along
    the viewing direction (down the beam path in cryo-EM).
    
    In the model's coordinate system:
    - Z-axis = [0, 0, 1] points along the viewing direction (beam path)
    - After rotation by Euler angles, this becomes R @ [0, 0, 1]
    - The arrow shows the 2D projection of this rotated Z-axis onto the XY plane
    
    Parameters:
    -----------
    euler_angles : np.ndarray
        Euler angles [phi, theta, psi] in radians (ZYZ convention)
    length : float
        Arrow length in pixels
        
    Returns:
    --------
    dx, dy : float
        Arrow direction components in micrograph pixel coordinates
    """
    # The viewing direction is along the Z axis after rotation
    # We can get this from the rotation matrix
    R = euler_to_rotation_matrix(euler_angles, convention='ZYZ')
    
    # The Z axis after rotation points in the direction: R @ [0, 0, 1]
    view_direction = R @ np.array([0, 0, 1])
    
    # Project onto XY plane (micrograph plane)
    dx = view_direction[0] * length
    dy = view_direction[1] * length
    
    return dx, dy


def get_particle_axes(euler_angles: np.ndarray, length: float = 30.0) -> Dict[str, np.ndarray]:
    """
    Get X, Y, Z axis directions for visualizing particle orientation.
    
    Returns:
    --------
    dict with keys 'x', 'y', 'z' containing (dx, dy) tuples for each axis
    """
    R = euler_to_rotation_matrix(euler_angles, convention='ZYZ')
    
    # Get rotated axes
    x_axis = R @ np.array([1, 0, 0])
    y_axis = R @ np.array([0, 1, 0])
    z_axis = R @ np.array([0, 0, 1])
    
    # Project onto XY plane
    axes = {
        'x': (x_axis[0] * length, x_axis[1] * length),
        'y': (y_axis[0] * length, y_axis[1] * length),
        'z': (z_axis[0] * length, z_axis[1] * length),
    }
    
    return axes


def calculate_vector_from_two_points(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """
    Calculate a normalized vector from point1 to point2.
    
    Useful for converting ChimeraX marker positions to a direction vector.
    
    Parameters:
    -----------
    point1 : np.ndarray
        3D coordinates [x, y, z] of first point
    point2 : np.ndarray
        3D coordinates [x, y, z] of second point
        
    Returns:
    --------
    np.ndarray
        Normalized 3D unit vector pointing from point1 to point2
    """
    vec = np.array(point2, dtype=float) - np.array(point1, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        raise ValueError("Points are identical, cannot determine direction")
    return vec / norm


def calculate_custom_vector_from_pdb(pdb_data: Dict, 
                                     method: str = 'user_defined',
                                     vector: Optional[np.ndarray] = None,
                                     chain_ids: Optional[List[str]] = None,
                                     atom_names: Optional[List[str]] = None,
                                     residue_names: Optional[List[str]] = None,
                                     from_center_to: Optional[str] = None) -> np.ndarray:
    """
    Calculate a custom 3D vector in the model's coordinate system from PDB structure data.
    
    This function allows you to define a structural axis (e.g., ribosome exit tunnel)
    by various methods:
    - 'user_defined': Use a manually specified 3D vector
    - 'chain_com': Calculate vector from center of mass of one chain to another
    - 'atom_selection': Calculate vector from center of mass of selected atoms
    - 'chain_axis': Calculate principal axis of a specific chain
    
    Parameters:
    -----------
    pdb_data : dict
        PDB structure data from load_pdb_structure()
    method : str
        Method to calculate vector: 'user_defined', 'chain_com', 'atom_selection', 'chain_axis'
    vector : np.ndarray, optional
        For 'user_defined': 3D vector [x, y, z] in model coordinate system
    chain_ids : list of str, optional
        For 'chain_com' or 'chain_axis': list of chain IDs to use
    atom_names : list of str, optional
        For 'atom_selection': list of atom names to select (e.g., ['CA', 'P'])
    residue_names : list of str, optional
        For 'atom_selection': list of residue names to select
    from_center_to : str, optional
        For 'chain_com': 'first_to_second' or 'second_to_first' (direction)
        
    Returns:
    --------
    np.ndarray
        3D unit vector [x, y, z] in model coordinate system (normalized)
    """
    coords = pdb_data['coords']
    chain_ids_data = pdb_data['chain_ids']
    
    if method == 'user_defined':
        if vector is None:
            raise ValueError("vector must be provided for 'user_defined' method")
        vec = np.array(vector, dtype=float)
        if vec.shape != (3,):
            raise ValueError(f"vector must be shape (3,), got {vec.shape}")
        # Normalize
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            raise ValueError("vector has zero length")
        return vec / norm
    
    elif method == 'chain_com':
        if chain_ids is None or len(chain_ids) < 2:
            raise ValueError("chain_com method requires at least 2 chain IDs")
        
        # Calculate center of mass for each chain
        coms = []
        for chain_id in chain_ids[:2]:
            mask = chain_ids_data == chain_id
            if not np.any(mask):
                raise ValueError(f"Chain {chain_id} not found in structure")
            chain_coords = coords[mask]
            com = np.mean(chain_coords, axis=0)
            coms.append(com)
        
        # Calculate vector from first to second (or reverse)
        if from_center_to == 'second_to_first':
            vec = coms[0] - coms[1]
        else:  # default: first_to_second
            vec = coms[1] - coms[0]
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            raise ValueError("Chains have identical center of mass")
        return vec / norm
    
    elif method == 'atom_selection':
        # Select atoms based on criteria
        mask = np.ones(len(coords), dtype=bool)
        
        if atom_names is not None:
            atom_mask = np.isin(pdb_data['atom_names'], atom_names)
            mask = mask & atom_mask
        
        if residue_names is not None:
            residue_mask = np.isin(pdb_data['residue_names'], residue_names)
            mask = mask & residue_mask
        
        if chain_ids is not None:
            chain_mask = np.isin(chain_ids_data, chain_ids)
            mask = mask & chain_mask
        
        if not np.any(mask):
            raise ValueError("No atoms match selection criteria")
        
        selected_coords = coords[mask]
        
        # Calculate principal axis using PCA
        # Center the coordinates
        centered = selected_coords - np.mean(selected_coords, axis=0)
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Get eigenvector with largest eigenvalue (principal axis)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        principal_idx = np.argmax(eigenvals)
        vec = eigenvecs[:, principal_idx]
        
        # Normalize (should already be normalized, but ensure)
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            raise ValueError("Could not determine principal axis")
        return vec / norm
    
    elif method == 'chain_axis':
        if chain_ids is None or len(chain_ids) == 0:
            raise ValueError("chain_axis method requires at least 1 chain ID")
        
        # Use first chain
        chain_id = chain_ids[0]
        mask = chain_ids_data == chain_id
        if not np.any(mask):
            raise ValueError(f"Chain {chain_id} not found in structure")
        
        chain_coords = coords[mask]
        
        # Calculate principal axis using PCA
        centered = chain_coords - np.mean(chain_coords, axis=0)
        cov = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        principal_idx = np.argmax(eigenvals)
        vec = eigenvecs[:, principal_idx]
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            raise ValueError("Could not determine principal axis")
        return vec / norm
    
    else:
        raise ValueError(f"Unknown method: {method}")


def project_custom_vector(vector_3d: np.ndarray, euler_angles: np.ndarray, length: float = 50.0) -> Tuple[float, float]:
    """
    Project a custom 3D vector (in model coordinate system) onto the micrograph plane.
    
    This function takes a vector defined in the model's coordinate system (e.g., 
    pointing along a structural axis like the ribosome exit tunnel) and projects
    it onto the 2D micrograph plane after applying the particle's rotation.
    
    Parameters:
    -----------
    vector_3d : np.ndarray
        3D unit vector [x, y, z] in model coordinate system
    euler_angles : np.ndarray
        Euler angles [phi, theta, psi] in radians (ZYZ convention)
    length : float
        Arrow length in pixels
        
    Returns:
    --------
    dx, dy : float
        Arrow direction components in micrograph pixel coordinates
    """
    # Get rotation matrix
    R = euler_to_rotation_matrix(euler_angles, convention='ZYZ')
    
    # Rotate the custom vector
    rotated_vector = R @ vector_3d
    
    # Project onto XY plane (micrograph plane)
    dx = rotated_vector[0] * length
    dy = rotated_vector[1] * length
    
    return dx, dy


def fractional_to_pixel_coords(center_x_frac: float, center_y_frac: float, 
                                micrograph_shape: Tuple[int, int]) -> Tuple[int, int]:
    """Convert fractional coordinates (0-1) to pixel coordinates."""
    x_pixel = int(center_x_frac * micrograph_shape[1])
    y_pixel = int(center_y_frac * micrograph_shape[0])
    return x_pixel, y_pixel


def calculate_structure_com(pdb_data: Dict) -> np.ndarray:
    """
    Calculate the center of mass (COM) of the structure from PDB data.
    
    Uses the mean of all atom coordinates (equivalent to COM if all atoms have equal mass).
    
    Parameters:
    -----------
    pdb_data : dict
        PDB structure data from load_pdb_structure()
        
    Returns:
    --------
    np.ndarray
        3D coordinates [x, y, z] of the structure's center of mass
    """
    coords = pdb_data['coords']
    if len(coords) == 0:
        return np.array([0.0, 0.0, 0.0])
    return coords.mean(axis=0)


def calculate_com_offset_correction(pdb_data: Dict, euler_angles: np.ndarray,
                                    particle_center_pixel: Tuple[float, float],
                                    pixel_size: float,
                                    projection_size_pixels: float,
                                    shifts_angstroms: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
    """
    Calculate the offset correction needed to align the structure's center of mass
    with the particle center from cryoSPARC.
    
    CRITICAL INSIGHT: PyMOL centers the structure at its COM before rotation. This means:
    1. Structure COM in original PDB coords -> PyMOL centers it -> COM becomes (0,0,0)
    2. After rotation, COM is still at (0,0,0) in rotated coords
    3. After projection, COM is at (0,0) in projection image (center of image)
    4. We place projection image centered at particle_center_pixel
    5. So structure COM appears at particle_center_pixel in micrograph coords
    
    However, if the structure COM in the original PDB doesn't match where cryoSPARC
    thinks the particle center is (in volume coords), there will be an offset.
    
    The challenge: We can't directly compare PDB coords to cryoSPARC volume coords
    without knowing the transformation. But we can calculate the offset by:
    1. Getting structure COM in original PDB coords
    2. Rotating it using particle's Euler angles (same as projection)
    3. Projecting to 2D (X, Y coords in Angstroms)
    4. Converting to pixels: com_2d_pixels = com_2d_angstroms / pixel_size
    5. The offset is: offset = particle_center_pixel - com_2d_pixels
    
    But wait - PyMOL already centers at COM, so after centering, the COM is at (0,0,0).
    So the offset should be zero... unless there's a coordinate system mismatch.
    
    Actually, the real issue might be that we need to account for the fact that the
    structure COM in PDB coords might not align with the particle center in volume coords.
    But without knowing the PDB->volume transformation, we can't calculate this directly.
    
    For now, we'll calculate the offset assuming the coordinate systems are aligned,
    which should give us zero offset. But this function can be extended if we find
    a way to properly transform between coordinate systems.
    
    Parameters:
    -----------
    pdb_data : dict
        PDB structure data from load_pdb_structure()
    euler_angles : np.ndarray
        Euler angles [phi, theta, psi] in radians (ZYZ convention) for this particle
    particle_center_pixel : tuple
        (x, y) pixel coordinates of the particle center from cryoSPARC
    pixel_size : float
        Pixel size in Angstroms per pixel
    projection_size_pixels : float
        Size of the projection image in pixels (assumed square)
        
    Returns:
    --------
    tuple
        (offset_x, offset_y) in pixels - the correction to apply to projection placement
    """
    # Get structure COM in original PDB coordinates
    if 'com' in pdb_data:
        com_pdb = pdb_data['com']
    else:
        com_pdb = calculate_structure_com(pdb_data)
    
    # CRITICAL INSIGHT: The structure COM in PDB coordinates needs to be compared to
    # where cryoSPARC thinks the particle center is. However, we can't directly compare
    # PDB coordinates to volume coordinates without knowing the transformation.
    #
    # BUT: We can calculate the offset by understanding that:
    # 1. PyMOL centers the structure at its COM before rotation
    # 2. After rotation and projection, the COM is at the center of the projection image
    # 3. The projection image is placed centered at particle_center_pixel
    # 4. So the structure COM appears at particle_center_pixel in micrograph coords
    #
    # However, if the structure COM in PDB coords doesn't match the volume center that
    # cryoSPARC uses, there will be an offset. The shifts from cryoSPARC might account
    # for some of this, but not necessarily all of it.
    #
    # The key: We need to calculate where the structure COM would appear in the
    # micrograph if we didn't center it. Then compare that to where the particle center is.
    #
    # Actually, let's think about this differently:
    # - The structure COM in PDB coords, after rotation and projection, should align
    #   with the particle center in the micrograph
    # - PyMOL centers at COM, so after centering, COM is at (0,0,0)
    # - After rotation, COM is still at (0,0,0) in rotated coords
    # - After projection, COM is at (0,0) in projection image
    # - We place projection centered at particle_center_pixel
    # - So COM appears at particle_center_pixel
    #
    # The offset should be zero UNLESS there's a coordinate system mismatch.
    #
    # Let's calculate the offset by NOT centering at COM first, then seeing where
    # the COM ends up after rotation and projection, then comparing to particle center.
    
    # CRITICAL: The previous calculation was wrong. Let me reconsider:
    #
    # PyMOL centers the structure at its COM BEFORE rotation. This means:
    # 1. Structure COM in PDB coords -> PyMOL centers it -> COM becomes (0,0,0) in centered coords
    # 2. After rotation, COM is still at (0,0,0) in rotated coords  
    # 3. After projection, COM is at (0,0) in projection image (center of image)
    # 4. We place projection image centered at particle_center_pixel
    # 5. So structure COM appears at particle_center_pixel in micrograph coords
    #
    # The projection image center = structure COM (after PyMOL centering).
    # When we place the projection centered at particle_center_pixel, the COM appears
    # at particle_center_pixel. So they should already align, and offset should be 0.
    #
    # The problem: We can't directly compare PDB coordinates to cryoSPARC volume coordinates
    # without knowing the transformation. The structure COM in PDB coords might not match
    # the particle center in volume coords.
    #
    # Since we don't know the PDB->volume transformation, we can't calculate the offset.
    # The fine-tuning sliders can be used to manually correct for any systematic offsets.
    
    # Return zero offset - the calculation needs to be reconsidered
    # Since PyMOL centers at COM and we place projection centered at particle center,
    # they should already align. We can't calculate the offset without knowing the
    # PDB->volume transformation.
    offset_x = 0.0
    offset_y = 0.0
    
    return (offset_x, offset_y)


def load_pdb_structure(pdb_path: str):
    """
    Load structure file (.pdb or .cif) and extract coordinates and chain information.
    Supports both PDB and mmCIF formats.
    
    Args:
        pdb_path: Path to .pdb or .cif structure file
    
    Returns:
        dict with keys:
            - coords: (N, 3) array of atom coordinates
            - chain_ids: (N,) array of chain identifiers
            - residue_names: (N,) array of residue names (3-letter codes)
            - atom_names: (N,) array of atom names
            - ca_coords: (M, 3) array of CA (alpha carbon) coordinates for cartoon rendering
            - ca_chain_ids: (M,) array of chain IDs for CA atoms
            - ca_residue_names: (M,) array of residue names for CA atoms
            - com: (3,) array of center of mass coordinates [x, y, z]
    """
    if not BIOPYTHON_AVAILABLE:
        raise ImportError("BioPython is required for structure file reading. Install with: pip install biopython")
    
    from pathlib import Path
    file_ext = Path(pdb_path).suffix.lower()
    
    # Use appropriate parser based on file extension
    if file_ext == '.cif':
        from Bio.PDB.MMCIFParser import MMCIFParser
        parser = MMCIFParser(QUIET=True)
    elif file_ext == '.pdb':
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Expected .pdb or .cif")
    
    structure = parser.get_structure('structure', pdb_path)
    
    coords = []
    chain_ids = []
    residue_names = []
    atom_names = []
    
    # Also extract CA atoms for cartoon/ribbon rendering
    ca_coords = []
    ca_chain_ids = []
    ca_residue_names = []
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                resname = residue.get_resname()
                for atom in residue:
                    coords.append(atom.coord)
                    chain_ids.append(chain_id)
                    residue_names.append(resname)
                    atom_names.append(atom.name)
                    
                    # Extract CA atoms for cartoon rendering
                    if atom.name == 'CA':
                        ca_coords.append(atom.coord)
                        ca_chain_ids.append(chain_id)
                        ca_residue_names.append(resname)
    
    result = {
        'coords': np.array(coords),
        'chain_ids': np.array(chain_ids),
        'residue_names': np.array(residue_names),
        'atom_names': np.array(atom_names)
    }
    
    # Add CA data if available
    if len(ca_coords) > 0:
        result['ca_coords'] = np.array(ca_coords)
        result['ca_chain_ids'] = np.array(ca_chain_ids)
        result['ca_residue_names'] = np.array(ca_residue_names)
    
    # Calculate and store center of mass
    if len(coords) > 0:
        result['com'] = np.array(coords).mean(axis=0)
    else:
        result['com'] = np.array([0.0, 0.0, 0.0])
    
    return result


def is_nucleic_acid(resname: str) -> bool:
    """Check if residue name corresponds to a nucleic acid."""
    nucleic_acids = ['A', 'T', 'G', 'C', 'U', 'DA', 'DT', 'DG', 'DC', 'DU',
                     'ADE', 'THY', 'GUA', 'CYT', 'URA', 'A5', 'T5', 'G5', 'C5', 'U5',
                     'A3', 'T3', 'G3', 'C3', 'U3']
    return resname.strip() in nucleic_acids


def get_chain_colors(chain_ids: np.ndarray, residue_names: np.ndarray,
                     chain_color_map: Optional[Dict[str, str]] = None,
                     default_protein_color: str = '#49A9CC',
                     default_nucleic_color: str = '#FF6B6B') -> np.ndarray:
    """
    Assign colors to atoms based on chain and residue type.
    
    Args:
        chain_ids: Array of chain identifiers
        residue_names: Array of residue names
        chain_color_map: Optional dict mapping chain_id -> hex color
        default_protein_color: Color for protein chains (if not in chain_color_map)
        default_nucleic_color: Color for nucleic acid chains (if not in chain_color_map)
    
    Returns:
        (N, 3) array of RGB colors in [0, 1] range
    """
    def hex_to_rgb(hex_color: str) -> np.ndarray:
        """Convert hex color to RGB [0, 1]."""
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)]) / 255.0
    
    colors = np.zeros((len(chain_ids), 3))
    
    for i, (chain_id, resname) in enumerate(zip(chain_ids, residue_names)):
        if chain_color_map and chain_id in chain_color_map:
            # Use custom color for this chain
            colors[i] = hex_to_rgb(chain_color_map[chain_id])
        elif is_nucleic_acid(resname):
            # Nucleic acid
            colors[i] = hex_to_rgb(default_nucleic_color)
        else:
            # Protein (default)
            colors[i] = hex_to_rgb(default_protein_color)
    
    return colors


def project_pdb_cartoon_pymol(pdb_data: Dict, euler_angles: np.ndarray,
                        output_size: Tuple[int, int] = (500, 500),
                        chain_color_map: Optional[Dict[str, str]] = None,
                        default_protein_color: str = '#007CBE',  # Updated to match new style
                        default_nucleic_color: str = '#3B1F2B',  # Updated to match new style
                        ribbon_width: float = 5.0,
                        pdb_path: Optional[str] = None,
                        pymol_path: Optional[str] = None,
                        rotation_correction_x: float = 0.0,
                        rotation_correction_y: float = 0.0,
                        rotation_correction_z: float = 0.0,
                        marker1_coords: Optional[np.ndarray] = None,
                        marker2_coords: Optional[np.ndarray] = None,
                        render_vector_arrow: bool = False) -> np.ndarray:
    """
    Project a structure file (.pdb or .cif) using fast pseudo-surface rendering (CA/P spheres).
    Uses only CA atoms for protein and P atoms for nucleic acids for speed.
    Properly applies rotation matrix to each particle for correct pose visualization.
    Supports both PDB and mmCIF formats.
    """
    import subprocess
    import uuid
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    import shutil
    import os
    
    if pdb_path is None:
        raise ValueError("pdb_path is required for PyMOL rendering")
    
    if not Path(pdb_path).exists():
        raise FileNotFoundError(f"Structure file not found: {pdb_path}")
    
    # Find PyMOL executable - be VERY explicit to avoid finding ChimeraX
    if pymol_path and Path(pymol_path).exists() and 'chimerax' not in str(pymol_path).lower():
        pymol_exe = pymol_path
    else:
        # Try common locations (conda environments, system PATH)
        import shutil
        # First try system PATH
        pymol_exe = shutil.which('pymol')
        if pymol_exe and 'chimerax' not in pymol_exe.lower():
            pass  # Found in PATH
        else:
            # Try conda locations
            possible_paths = [
                Path.home() / 'miniconda3' / 'bin' / 'pymol',
                Path.home() / 'anaconda3' / 'bin' / 'pymol',
            ]
            conda_base = os.environ.get('CONDA_PREFIX', '')
            if conda_base:
                possible_paths.insert(0, Path(conda_base) / 'bin' / 'pymol')
            
            pymol_exe = None
            for path in possible_paths:
                path_str = str(path)
                if path.exists() and 'chimerax' not in path_str.lower():
                    pymol_exe = path_str
                    break
    
    if not pymol_exe:
        raise RuntimeError("PyMOL not found. Install with: conda install -c conda-forge pymol-open-source")
    
    # CRITICAL: Final verification - make absolutely sure it's not ChimeraX
    if 'chimerax' in pymol_exe.lower():
        raise RuntimeError(f"ERROR: Found ChimeraX instead of PyMOL at {pymol_exe}. Please check your PATH.")
    
    # Verify it's actually PyMOL by running --version
    try:
        verify_result = subprocess.run([pymol_exe, '--version'], 
                                      capture_output=True, text=True, timeout=5)
        if 'chimerax' in verify_result.stdout.lower() or 'chimerax' in verify_result.stderr.lower():
            raise RuntimeError(f"ERROR: {pymol_exe} is actually ChimeraX, not PyMOL!")
    except subprocess.TimeoutExpired:
        pass  # Version check timed out, but continue anyway
    
    # Convert Euler angles to rotation matrix (ZYZ convention)
    # This rotation matrix rotates from object space to world space
    rot = Rotation.from_euler('ZYZ', euler_angles, degrees=False)
    R = rot.as_matrix()
    
    # Extract ZYZ Euler angles for sequential rotation
    # ZYZ convention: first rotate around Z by phi, then Y by theta, then Z by psi
    phi, theta, psi = euler_angles[0], euler_angles[1], euler_angles[2]
    phi_deg = float(phi * 180.0 / np.pi)
    theta_deg = float(theta * 180.0 / np.pi)
    psi_deg = float(psi * 180.0 / np.pi)
    
    # Also prepare transform_object matrix as backup
    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = R
    matrix_list = [float(x) for x in transform_matrix.flatten().tolist()]
    matrix_str = str(matrix_list)
    
    # Create temporary files
    h, w = output_size
    output_file = Path.cwd() / f'pymol_render_{uuid.uuid4().hex[:8]}.png'
    output_abs = str(output_file.absolute())
    pdb_abs = str(Path(pdb_path).absolute())
    
    # Convert hex colors to RGB tuples (0-1 range) for PyMOL
    def hex_to_rgb_tuple(hex_color):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)
    
    # Build PyMOL script - Fast pseudo-surface rendering using CA/P spheres
    # CRITICAL: Each script must be completely independent - PyMOL may cache state
    full_obj = "rib_all"
    rep_obj = "rib_pseudo"
    
    script_lines = [
        'import sys',
        'import pymol',
        'from pymol import cmd',
        'pymol.finish_launching(["pymol", "-cq"])',
        'cmd.reinitialize()',  # Reset all PyMOL settings to defaults
        f'cmd.load("{pdb_abs}", "{full_obj}")',
        f'cmd.hide("everything", "{full_obj}")',
        # Create pseudo-surface object: CA for protein, P for nucleic acids
        f'cmd.create("{rep_obj}", "({full_obj} and polymer.protein and name CA) or ({full_obj} and polymer.nucleic and name P)")',
        f'cmd.hide("everything", "{rep_obj}")',
        f'cmd.show("spheres", "{rep_obj}")',
        # Make spheres large enough to overlap into continuous blob
        f'cmd.set("sphere_scale", 2.2, "{rep_obj}")',
        f'cmd.set("sphere_quality", 1)',
        f'cmd.set("sphere_transparency", 0.10, "{rep_obj}")',
        # Colors
    ]
    
    # Add color commands
    if chain_color_map:
        # Apply chain-specific colors FIRST
        # IMPORTANT: PyMOL chain selection syntax is "chain A" or 'chain "A"' for quoted IDs
        colored_chains = []
        for chain_id, hex_color in chain_color_map.items():
            # Clean chain_id (remove whitespace, ensure it's a string)
            chain_id_clean = str(chain_id).strip()
            colored_chains.append(chain_id_clean)
            rgb = hex_to_rgb_tuple(hex_color)
            # Create a safe color name (no special characters)
            color_name = f"chain_{chain_id_clean.replace(' ', '_').replace('-', '_')}_color"
            
            # PyMOL chain selection: use quotes around chain ID to handle special characters
            # Format: 'chain "A"' or 'chain A' (PyMOL handles both, but quotes are safer)
            # Build the selection string directly in the script line to avoid quote escaping issues
            # Use single quotes for the outer Python string to allow double quotes inside
            script_lines.append(f'cmd.set_color("{color_name}", {list(rgb)})')
            # Use single quotes for outer string, double quotes for PyMOL chain selection
            script_lines.append(f"cmd.color('{color_name}', '{rep_obj} and chain \"{chain_id_clean}\"')")
        
        # Apply default colors to chains NOT in chain_color_map
        # Build exclusion list for chains we already colored
        if colored_chains:
            # Create exclusion string: "not (chain A or chain B or ...)"
            # Use single quotes for outer string to allow double quotes for PyMOL
            chain_exclusions = ' or '.join([f'chain "{cid}"' for cid in colored_chains])
            exclusion_sel = f'not ({chain_exclusions})'
        else:
            exclusion_sel = ''
        
        prot_rgb = hex_to_rgb_tuple(default_protein_color)
        nuc_rgb = hex_to_rgb_tuple(default_nucleic_color)
        script_lines.append(f'cmd.set_color("protein_blue_default", {list(prot_rgb)})')
        script_lines.append(f'cmd.set_color("rna_plum_default", {list(nuc_rgb)})')
        
        # Color remaining protein and nucleic chains (excluding already-colored chains)
        if exclusion_sel:
            # Use single quotes for outer string to allow double quotes in exclusion_sel
            script_lines.append(f"cmd.color('protein_blue_default', '{rep_obj} and polymer.protein and {exclusion_sel}')")
            script_lines.append(f"cmd.color('rna_plum_default', '{rep_obj} and polymer.nucleic and {exclusion_sel}')")
        else:
            # No exclusions needed (shouldn't happen if chain_color_map has items)
            script_lines.append(f'cmd.color("protein_blue_default", "{rep_obj} and polymer.protein")')
            script_lines.append(f'cmd.color("rna_plum_default", "{rep_obj} and polymer.nucleic")')
    else:
        # Use default colors: protein blue #007CBE, RNA plum #3B1F2B
        prot_rgb = hex_to_rgb_tuple(default_protein_color)
        nuc_rgb = hex_to_rgb_tuple(default_nucleic_color)
        script_lines.append(f'cmd.set_color("protein_blue", {list(prot_rgb)})')
        script_lines.append(f'cmd.set_color("rna_plum", {list(nuc_rgb)})')
        script_lines.append(f'cmd.color("protein_blue", "{rep_obj} and polymer.protein")')
        script_lines.append(f'cmd.color("rna_plum", "{rep_obj} and polymer.nucleic")')
    
        # Lighting - with ray tracing for quality
        script_lines.extend([
            'cmd.bg_color("white")',
            'cmd.set("ray_opaque_background", 0)',  # Transparent background
            'cmd.set("orthoscopic", 1)',
            'cmd.set("antialias", 2)',
            'cmd.set("ray_trace_mode", 1)',  # Enable ray tracing for quality
            'cmd.set("ray_shadows", 0)',
            'cmd.set("two_sided_lighting", 1)',
            'cmd.set("ambient", 0.6)',
            'cmd.set("direct", 0.6)',
            'cmd.set("specular", 0.2)',
            'cmd.set("shininess", 20)',
            'cmd.set("reflect", 0.0)',
            'cmd.set("ray_trace_fog", 0)',
            'cmd.set("depth_cue", 0)',
            'cmd.set("gamma", 1.0)',
        ])
    
    # Center, set origin, and apply rotation
    # CRITICAL: PyMOL's cmd.center() centers the structure on its center of mass (COM).
    # The rotation is applied around the origin (set by cmd.origin()), which is at the COM.
    # The projection is rendered centered in the image, and we place it at the particle center
    # from cryoSPARC. If the structure's COM doesn't match the particle center from cryoSPARC,
    # there will be a small offset. This is typically negligible but can cause slight misalignment.
    script_lines.append(f'cmd.center("{rep_obj}")')
    script_lines.append(f'cmd.origin("{rep_obj}")')
    
    # CRITICAL: Apply rotation to match cryoSPARC particle orientation
    # 
    # In cryoSPARC, alignments3D/pose contains Euler angles [phi, theta, psi] in ZYZ convention.
    # scipy's Rotation.from_euler('ZYZ', ...) gives us a rotation matrix R.
    #
    # The key question: what does R represent?
    # - In project_volume: R.T is used to rotate from view space back to volume space
    #   This means R rotates from volume/model space to view space
    # - In project_pdb_cartoon: R is used directly: coords_rotated = R @ coords_centered
    #   This confirms R rotates from model space to view space
    #
    # For PyMOL, we want to rotate the model so it appears as it does in the micrograph.
    # PyMOL's rotate command applies rotations sequentially in the object's coordinate system.
    #
    # ZYZ convention (intrinsic rotations):
    # 1. Rotate around Z by phi
    # 2. Rotate around Y by theta (in the rotated coordinate system)
    # 3. Rotate around Z by psi (in the twice-rotated coordinate system)
    #
    # scipy's Rotation.from_euler('ZYZ', ...) computes this as: R = R_z(psi) @ R_y(theta) @ R_z(phi)
    # This means the rotations are applied right-to-left: first phi, then theta, then psi.
    #
    # For PyMOL's rotate command, we need to apply them in the correct order.
    # Since PyMOL applies rotations sequentially in the object's coordinate system,
    # we apply them in the order: phi (Z), then theta (Y), then psi (Z).
    #
    # However, there's a subtlety: PyMOL's rotate might apply rotations in a different
    # coordinate system or order than scipy. Let's use the rotation matrix directly
    # with transform_object to ensure we get the exact same rotation as scipy.
    #
    # For transform_object, we need to determine if we should use R or R.T.
    # If R rotates from model to view, and we want to rotate the model to view space,
    # then we should apply R to model coordinates. So we use R directly.
    
    # CRITICAL: Try using sequential rotations instead of transform_object
    # 
    # We've tried multiple matrix combinations with transform_object, none worked.
    # Let's try sequential rotations with PyMOL's rotate command.
    # 
    # ZYZ convention: first rotate around Z by phi, then Y by theta, then Z by psi
    # These are intrinsic rotations (each rotation is in the rotated coordinate system)
    #
    # PyMOL's rotate command: rotate axis, angle, object
    # Rotations are applied sequentially in the object's coordinate system
    #
    # For ZYZ Euler angles, we apply:
    # 1. Rotate around Z by phi
    # 2. Rotate around Y by theta (in the rotated coordinate system)
    # 3. Rotate around Z by psi (in the twice-rotated coordinate system)
    #
    # CRITICAL: Apply rotation corrections based on user-selected angles (in degrees)
    # These allow troubleshooting to find the correct rotation correction
    from scipy.spatial.transform import Rotation as Rot
    
    # Build rotation correction matrix from X, Y, Z rotation angles
    # Apply rotations in order: X, then Y, then Z (intrinsic rotations)
    if abs(rotation_correction_x) > 1e-6 or abs(rotation_correction_y) > 1e-6 or abs(rotation_correction_z) > 1e-6:
        # Convert degrees to radians
        rot_x_rad = np.deg2rad(rotation_correction_x)
        rot_y_rad = np.deg2rad(rotation_correction_y)
        rot_z_rad = np.deg2rad(rotation_correction_z)
        
        # Create rotation from Euler angles (XYZ intrinsic convention)
        # This applies rotations in order: X, then Y, then Z
        rot_correction = Rot.from_euler('XYZ', [rot_x_rad, rot_y_rad, rot_z_rad], degrees=False)
        R_correction = rot_correction.as_matrix()
    else:
        R_correction = np.eye(3)  # No correction
    
    # Apply correction AFTER the main rotation
    # R rotates from model space to view space
    # R_correction is applied as an additional rotation (in XYZ convention)
    R_transform = R @ R_correction
    
    # Debug: Print rotation correction if applied
    if abs(rotation_correction_x) > 1e-6 or abs(rotation_correction_y) > 1e-6 or abs(rotation_correction_z) > 1e-6:
        print(f"DEBUG Rotation Correction: X={rotation_correction_x:.2f}°, Y={rotation_correction_y:.2f}°, Z={rotation_correction_z:.2f}°")
    
    # PyMOL's transform_object expects a 4x4 transformation matrix
    # Format: [r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, 0, 0, 0, 1]
    # This is row-major format
    transform_list = [
        R_transform[0,0], R_transform[0,1], R_transform[0,2], 0.0,  # First row + tx
        R_transform[1,0], R_transform[1,1], R_transform[1,2], 0.0,  # Second row + ty
        R_transform[2,0], R_transform[2,1], R_transform[2,2], 0.0,  # Third row + tz
        0.0, 0.0, 0.0, 1.0                                           # Bottom row
    ]
    transform_str = ', '.join([f'{x:.10f}' for x in transform_list])
    script_lines.append(f'cmd.transform_object("{rep_obj}", [{transform_str}], homogenous=0)')
    
    # Zoom and set viewport
    script_lines.append(f'cmd.zoom("{rep_obj}", complete=1)')
    
    # CRITICAL: Ensure structure is centered in view before rendering
    # Call center again after zoom to ensure perfect centering
    script_lines.append(f'cmd.center("{rep_obj}")')
    
    # Render vector arrow in PyMOL if markers are provided
    # This avoids coordinate transformation issues by rendering the arrow
    # in the same coordinate system and scale as the structure
    if render_vector_arrow and marker1_coords is not None and marker2_coords is not None:
        print(f"DEBUG PyMOL: Rendering arrow between markers: {marker1_coords} -> {marker2_coords}")
        # Markers are in model space (PDB coordinates)
        # The structure is centered first (cmd.center centers on COM), then rotated
        # We need to center the markers relative to structure COM, then rotate them
        # Calculate structure COM (use 'com' key if available, otherwise calculate)
        if 'com' in pdb_data:
            structure_com = pdb_data['com']
        else:
            structure_com = pdb_data['coords'].mean(axis=0)
        # Center markers relative to structure COM (same as PyMOL does)
        marker1_centered = marker1_coords - structure_com
        marker2_centered = marker2_coords - structure_com
        
        script_lines.append('# Render vector arrow between markers using fake atoms')
        script_lines.append('# Create fake atoms at centered marker positions (relative to structure COM)')
        script_lines.append(f'cmd.pseudoatom("marker1", pos=[{marker1_centered[0]:.3f}, {marker1_centered[1]:.3f}, {marker1_centered[2]:.3f}])')
        script_lines.append(f'cmd.pseudoatom("marker2", pos=[{marker2_centered[0]:.3f}, {marker2_centered[1]:.3f}, {marker2_centered[2]:.3f}])')
        script_lines.append('# Apply the same rotation to the fake atoms (after centering)')
        script_lines.append(f'cmd.transform_object("marker1", [{transform_str}], homogenous=0)')
        script_lines.append(f'cmd.transform_object("marker2", [{transform_str}], homogenous=0)')
        script_lines.append('# Draw distance/arrow between markers')
        script_lines.append('cmd.distance("vector_arrow", "marker1", "marker2")')
        script_lines.append('cmd.hide("labels", "vector_arrow")')
        script_lines.append('cmd.set("dash_gap", 0)')
        script_lines.append('cmd.set("dash_length", 0)')
        script_lines.append('cmd.color("white", "vector_arrow")')
        script_lines.append('cmd.show("dashes", "vector_arrow")')
        script_lines.append('# Hide the fake atoms (they\'re just for the distance calculation)')
        script_lines.append('cmd.hide("everything", "marker1")')
        script_lines.append('cmd.hide("everything", "marker2")')
    
    # Set viewport and render - NO ray tracing for speed (ray=0)
    script_lines.append(f'cmd.viewport({w}, {h})')
    script_lines.append(f'cmd.png("{output_abs}", width={w}, height={h}, ray=1, quiet=1)')
    script_lines.append('cmd.quit()')
    
    script_content = '\n'.join(script_lines)
    
    # Create unique script filename using UUID to ensure no collisions
    # Using hash of Euler angles can cause collisions for similar angles
    import uuid
    script_uuid = uuid.uuid4().hex[:8]
    script_file = Path.cwd().absolute() / f'pymol_script_{script_uuid}.py'
    script_file_abs = str(script_file.absolute())
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Debug: Print Euler angles and rotation matrix to verify transformations
    print(f"DEBUG PyMOL: Euler=[{euler_angles[0]:.6f}, {euler_angles[1]:.6f}, {euler_angles[2]:.6f}], "
          f"ZYZ_deg=[{phi_deg:.2f}, {theta_deg:.2f}, {psi_deg:.2f}], script={script_file.name}")
    print(f"DEBUG Rotation Matrix R (from model to view):")
    print(f"  R[0] = [{R[0,0]:.6f}, {R[0,1]:.6f}, {R[0,2]:.6f}]")
    print(f"  R[1] = [{R[1,0]:.6f}, {R[1,1]:.6f}, {R[1,2]:.6f}]")
    print(f"  R[2] = [{R[2,0]:.6f}, {R[2,1]:.6f}, {R[2,2]:.6f}]")
    
    try:
        # Run PyMOL with the script file
        # -c = run command, -q = quiet
        result = subprocess.run(
            [pymol_exe, '-cq', script_file_abs],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path.cwd().absolute())
        )
        
        # Check for errors
        if result.returncode != 0:
            error_msg = f"PyMOL failed (return code {result.returncode})"
            if result.stderr:
                error_msg += f"\nSTDERR:\n{result.stderr[-1000:]}"
            if result.stdout:
                error_msg += f"\nSTDOUT:\n{result.stdout[-1000:]}"
            raise RuntimeError(error_msg)
        
        # Check if file was created
        if not output_file.exists():
            error_msg = f"PyMOL did not create output file: {output_file}"
            if result.stderr:
                error_msg += f"\nSTDERR:\n{result.stderr[-1000:]}"
            if result.stdout:
                error_msg += f"\nSTDOUT:\n{result.stdout[-1000:]}"
            raise RuntimeError(error_msg)
        
        from PIL import Image
        import PIL.ImageEnhance as ImageEnhance
        img = Image.open(output_file)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        
        # Brighten the image for better visibility on micrographs
        image = np.array(img).astype(np.float32) / 255.0
        
        # No post-processing needed - the pseudo-surface rendering is already bright and clean
        
        if output_file.exists():
            output_file.unlink()
    finally:
        # Clean up script file
        try:
            if script_file.exists():
                script_file.unlink()
        except Exception:
            pass  # Ignore cleanup errors
    
    return image


def project_pdb_structure(pdb_data: Dict, euler_angles: np.ndarray,
                          output_size: Tuple[int, int] = (500, 500),
                          chain_color_map: Optional[Dict[str, str]] = None,
                          default_protein_color: str = '#49A9CC',
                          default_nucleic_color: str = '#FF6B6B',
                          atom_radius: float = 2.5,
                          line_width: float = 0.5,
                          pdb_path: Optional[str] = None,
                          chimerax_path: Optional[str] = None,
                        rotation_correction_x: float = 0.0,
                        rotation_correction_y: float = 0.0,
                        rotation_correction_z: float = 0.0,
                        marker1_coords: Optional[np.ndarray] = None,
                        marker2_coords: Optional[np.ndarray] = None,
                        render_vector_arrow: bool = False) -> np.ndarray:
    """
    Project a PDB structure at given Euler angles to a 2D RGBA image.
    
    Args:
        pdb_data: Dictionary from load_pdb_structure()
        euler_angles: Euler angles [phi, theta, psi] in radians (ZYZ convention)
        output_size: (height, width) of output image
        chain_color_map: Optional dict mapping chain_id -> hex color
        default_protein_color: Default color for protein chains
        default_nucleic_color: Default color for nucleic acid chains
        atom_radius: Radius of atoms in Angstroms for rendering
        line_width: Width of bonds in Angstroms
        pdb_path: Path to PDB file (required for ChimeraX rendering)
        chimerax_path: Path to ChimeraX executable (optional, will auto-detect)
    
    Returns:
        (H, W, 4) RGBA array in [0, 1] range
    """
    # Use PyMOL for cartoon/ribbon rendering if pdb_path is provided
    if pdb_path and ('ca_coords' in pdb_data and len(pdb_data['ca_coords']) > 0):
        # Use PyMOL API for proper structural rendering with correct rotation
        return project_pdb_cartoon_pymol(pdb_data, euler_angles, output_size,
                                         chain_color_map, default_protein_color, default_nucleic_color,
                                         pdb_path=pdb_path, pymol_path=chimerax_path,
                                         rotation_correction_x=rotation_correction_x,
                                         rotation_correction_y=rotation_correction_y,
                                         rotation_correction_z=rotation_correction_z,
                                         marker1_coords=marker1_coords,
                                         marker2_coords=marker2_coords,
                                         render_vector_arrow=render_vector_arrow)
    
    # If no pdb_path provided, raise error (ChimeraX rendering requires it)
    if not pdb_path:
        raise ValueError("pdb_path is required for ChimeraX rendering. Please provide the PDB file path.")
    
    # Fall back: return empty image if no CA atoms
    if 'ca_coords' not in pdb_data or len(pdb_data['ca_coords']) == 0:
        h, w = output_size
        return np.zeros((h, w, 4), dtype=np.float32)
    """
    Project a PDB structure at given Euler angles to a 2D RGBA image.
    
    Args:
        pdb_data: Dictionary from load_pdb_structure()
        euler_angles: Euler angles [phi, theta, psi] in radians (ZYZ convention)
        output_size: (height, width) of output image
        chain_color_map: Optional dict mapping chain_id -> hex color
        default_protein_color: Default color for protein chains
        default_nucleic_color: Default color for nucleic acid chains
        atom_radius: Radius of atoms in Angstroms for rendering
        line_width: Width of bonds in Angstroms
    
    Returns:
        (H, W, 4) RGBA array in [0, 1] range
    """
    # Use CA atoms for cartoon/ribbon rendering if available, otherwise fall back to all atoms
    if 'ca_coords' in pdb_data and len(pdb_data['ca_coords']) > 0:
        # Cartoon/ribbon style using CA backbone
        return project_pdb_cartoon(pdb_data, euler_angles, output_size,
                                   chain_color_map, default_protein_color, default_nucleic_color)
    
    # Fall back to atom-based rendering if no CA atoms
    coords = pdb_data['coords'].copy()
    chain_ids = pdb_data['chain_ids']
    residue_names = pdb_data['residue_names']
    
    if len(coords) == 0:
        # Return empty image if no atoms
        h, w = output_size
        return np.zeros((h, w, 4), dtype=np.float32)
    
    # Get rotation matrix
    R = euler_to_rotation_matrix(euler_angles, convention='ZYZ')
    
    # Center coordinates
    # CRITICAL: We center on the mean of all atom coordinates (center of mass if all atoms have equal mass).
    # This should match PyMOL's cmd.center() behavior. The rotation is applied around this center.
    # The projection is then centered in the output image. If the structure's center doesn't match
    # the particle center from cryoSPARC, there will be a small offset.
    center = coords.mean(axis=0)
    coords_centered = coords - center
    
    # Apply rotation
    coords_rotated = (R @ coords_centered.T).T
    
    # Project to 2D (drop Z coordinate, keep X and Y)
    coords_2d = coords_rotated[:, :2]
    
    # Get colors for each atom
    colors = get_chain_colors(chain_ids, residue_names, chain_color_map,
                             default_protein_color, default_nucleic_color)
    
    # Determine bounding box and scale
    min_coords = coords_2d.min(axis=0)
    max_coords = coords_2d.max(axis=0)
    range_coords = max_coords - min_coords
    max_range = max(range_coords)
    
    if max_range < 1e-6:
        # All atoms at same position - use default scale
        max_range = 100.0  # Default 100 Angstroms
        min_coords = coords_2d[0] - max_range / 2
        max_coords = coords_2d[0] + max_range / 2
    
    # Add padding
    padding = max(max_range * 0.1, 10.0)  # At least 10 Angstroms padding
    min_coords -= padding
    max_coords += padding
    range_coords = max_coords - min_coords
    max_range = max(range_coords)
    
    # Scale to fit output size
    scale = min(output_size) / max_range if max_range > 1e-6 else 1.0
    # Center the structure in the output image
    # Calculate center of bounding box
    bbox_center = (min_coords + max_coords) / 2.0
    # Translate coordinates so structure center is at origin, then scale
    coords_centered = coords_2d - bbox_center
    coords_scaled = coords_centered * scale
    # Translate to center of output image
    output_center = np.array([output_size[1] / 2.0, output_size[0] / 2.0])
    coords_scaled = coords_scaled + output_center
    
    # Sort atoms by depth (Z coordinate after rotation) - render back to front
    z_coords = coords_rotated[:, 2]
    sort_indices = np.argsort(-z_coords)  # Negative for back-to-front
    
    # Create output image
    h, w = output_size
    image = np.zeros((h, w, 4), dtype=np.float32)
    
    # Convert atom radius from Angstroms to pixels
    # Use larger radius for more structural/surface-like appearance
    atom_radius_px = max(2.0, atom_radius * scale)  # At least 2.0 pixels for better visibility
    
    # For very large structures, sample atoms to improve performance
    num_atoms = len(coords_scaled)
    if num_atoms > 10000:
        # Sample atoms: render every Nth atom to maintain visual quality while improving speed
        # Use a sampling rate that keeps ~10000 atoms for rendering
        sample_rate = max(1, num_atoms // 10000)
        sort_indices = sort_indices[::sample_rate]
        print(f"  Sampling atoms: rendering {len(sort_indices)} of {num_atoms} atoms (every {sample_rate}th)")
    
    # Pre-compute coordinate grids for efficiency (only if needed)
    # For large structures, we'll use a more efficient approach
    use_vectorized = num_atoms < 50000
    
    # Lighting direction (from top-left, like PyMOL/ChimeraX)
    light_dir = np.array([-0.5, 0.5, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Normalize Z coordinates for depth calculations
    z_min, z_max = z_coords.min(), z_coords.max()
    z_range = z_max - z_min + 1e-6
    
    if use_vectorized:
        # Original approach: pre-compute grids (faster for smaller structures)
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(float)
        
        # Render atoms back to front (so closer atoms appear on top)
        for idx in sort_indices:
            x, y = coords_scaled[idx]
            z = z_coords[idx]
            color = colors[idx]
            
            # Skip if atom is outside image bounds (with margin)
            margin = atom_radius_px * 2
            if x < -margin or x > w + margin or y < -margin or y > h + margin:
                continue
            
            # Calculate distance from each pixel to atom center
            dx = x_coords - x
            dy = y_coords - y
            dist_sq = dx**2 + dy**2
            dist = np.sqrt(dist_sq)
            
            # Create mask for pixels within atom radius
            mask = dist <= atom_radius_px
            
            if not mask.any():
                continue
            
            # Calculate sphere surface normal (for 3D sphere effect)
            # For a sphere, the normal at (dx, dy) is (dx/r, dy/r, sqrt(1 - (dx/r)^2 - (dy/r)^2))
            # But we approximate with a simpler approach for performance
            r_norm = np.clip(dist / atom_radius_px, 0, 1)
            dz_sphere = np.sqrt(np.maximum(0, 1 - r_norm**2))
            
            # Calculate lighting (Phong-like shading)
            # Surface normal in 2D projection (approximate)
            normal_x = np.where(mask, dx / (atom_radius_px + 1e-6), 0)
            normal_y = np.where(mask, dy / (atom_radius_px + 1e-6), 0)
            normal_z = np.where(mask, dz_sphere, 0)
            
            # Dot product with light direction
            dot_product = (normal_x * light_dir[0] + normal_y * light_dir[1] + normal_z * light_dir[2])
            dot_product = np.clip(dot_product, 0, 1)
            
            # Ambient + diffuse lighting (like PyMOL)
            ambient = 0.3
            diffuse = 0.7
            lighting = ambient + diffuse * dot_product
            
            # Add specular highlight for more realistic look
            specular_strength = 0.3
            specular_power = 32
            # View direction is (0, 0, 1) in 2D projection
            view_dir = np.array([0, 0, 1])
            reflect_dir = 2 * dot_product * np.array([normal_x, normal_y, normal_z]) - light_dir
            reflect_dir_z = np.clip(reflect_dir[2], 0, 1)
            specular = specular_strength * (reflect_dir_z ** specular_power)
            specular = np.where(mask, specular, 0)
            
            # Combine lighting
            total_lighting = lighting + specular
            total_lighting = np.clip(total_lighting, 0, 1.5)  # Allow slight overexposure for highlights
            
            # Apply lighting to color
            lit_color = color * total_lighting[:, :, np.newaxis]
            lit_color = np.clip(lit_color, 0, 1)
            
            # Use smooth falloff for sphere edges with softer edges for surface-like appearance
            # Sphere falloff: alpha = sqrt(1 - (r/R)^2) for r < R, but with softer falloff
            r_norm_clipped = np.clip(r_norm, 0, 1)
            # Use a softer falloff curve for more surface-like blending
            sphere_alpha = np.power(np.maximum(0, 1 - r_norm_clipped**2), 0.7)
            sphere_alpha = np.where(mask, sphere_alpha, 0)
            
            # Depth-based alpha adjustment (closer = more opaque)
            z_norm = (z - z_min) / z_range
            depth_factor = 0.75 + 0.25 * z_norm  # Range from 0.75 to 1.0 (more opaque overall)
            alpha = sphere_alpha * depth_factor
            
            # Enhance color saturation and contrast for more structural appearance
            lit_color = np.clip(lit_color * 1.15, 0, 1)
            
            # Add slight darkening at edges for more 3D structural appearance
            edge_factor = np.clip(1.0 - r_norm_clipped * 0.3, 0.7, 1.0)
            lit_color = lit_color * edge_factor[:, :, np.newaxis]
            
            # Alpha blend with existing image (proper over operator)
            existing_alpha = image[:, :, 3]
            new_alpha = existing_alpha + alpha * (1 - existing_alpha)
            
            # Blend colors using proper alpha compositing
            alpha_mask = alpha > 0
            for c in range(3):
                image[:, :, c] = np.where(alpha_mask,
                    np.where(existing_alpha > 0,
                        (image[:, :, c] * existing_alpha + lit_color[:, :, c] * alpha) / np.maximum(new_alpha, 1e-6),
                        lit_color[:, :, c]),
                    image[:, :, c])
            
            image[:, :, 3] = new_alpha
    else:
        # Fast approach for large structures: use integer coordinates and direct pixel access
        # This is much faster but still includes lighting for quality
        z_min, z_max = z_coords.min(), z_coords.max()
        z_range = z_max - z_min + 1e-6
        
        # Lighting direction (from top-left, like PyMOL/ChimeraX)
        light_dir = np.array([-0.5, 0.5, 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Render atoms back to front
        for idx in sort_indices:
            x, y = coords_scaled[idx]
            z = z_coords[idx]
            color = colors[idx]
            
            # Skip if atom is outside image bounds (with margin)
            margin = atom_radius_px * 2
            if x < -margin or x > w + margin or y < -margin or y > h + margin:
                continue
            
            # Calculate bounding box for this atom
            x_min = max(0, int(x - atom_radius_px))
            x_max = min(w, int(x + atom_radius_px) + 1)
            y_min = max(0, int(y - atom_radius_px))
            y_max = min(h, int(y + atom_radius_px) + 1)
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # Create local coordinate arrays for this atom's region
            y_local, x_local = np.mgrid[y_min:y_max, x_min:x_max].astype(float)
            
            # Calculate distances
            dx = x_local - x
            dy = y_local - y
            dist_sq = dx**2 + dy**2
            dist = np.sqrt(dist_sq)
            
            # Create mask
            mask = dist <= atom_radius_px
            if not mask.any():
                continue
            
            # Calculate sphere surface normal
            r_norm = np.clip(dist / atom_radius_px, 0, 1)
            dz_sphere = np.sqrt(np.maximum(0, 1 - r_norm**2))
            
            # Calculate lighting
            normal_x = np.where(mask, dx / (atom_radius_px + 1e-6), 0)
            normal_y = np.where(mask, dy / (atom_radius_px + 1e-6), 0)
            normal_z = np.where(mask, dz_sphere, 0)
            
            dot_product = (normal_x * light_dir[0] + normal_y * light_dir[1] + normal_z * light_dir[2])
            dot_product = np.clip(dot_product, 0, 1)
            
            # Ambient + diffuse lighting
            ambient = 0.3
            diffuse = 0.7
            lighting = ambient + diffuse * dot_product
            
            # Specular highlight
            specular_strength = 0.3
            specular_power = 32
            reflect_dir_z = np.clip(2 * dot_product * normal_z - light_dir[2], 0, 1)
            specular = specular_strength * (reflect_dir_z ** specular_power)
            specular = np.where(mask, specular, 0)
            
            total_lighting = lighting + specular
            total_lighting = np.clip(total_lighting, 0, 1.5)
            
            # Apply lighting to color
            lit_color = color * total_lighting[:, :, np.newaxis]
            lit_color = np.clip(lit_color * 1.1, 0, 1)  # Slight saturation boost
            
            # Sphere falloff with softer edges for surface-like appearance
            r_norm_clipped = np.clip(r_norm, 0, 1)
            sphere_alpha = np.power(np.maximum(0, 1 - r_norm_clipped**2), 0.7)
            sphere_alpha = np.where(mask, sphere_alpha, 0)
            
            # Depth-based alpha
            z_norm = (z - z_min) / z_range
            depth_factor = 0.75 + 0.25 * z_norm
            alpha = sphere_alpha * depth_factor
            
            # Enhance color saturation
            lit_color = np.clip(lit_color * 1.15, 0, 1)
            
            # Add edge darkening for 3D effect
            edge_factor = np.clip(1.0 - r_norm_clipped * 0.3, 0.7, 1.0)
            lit_color = lit_color * edge_factor[:, :, np.newaxis]
            
            # Blend with existing image (proper alpha compositing)
            existing_alpha = image[y_min:y_max, x_min:x_max, 3]
            new_alpha = existing_alpha + alpha * (1 - existing_alpha)
            
            for c in range(3):
                existing_color = image[y_min:y_max, x_min:x_max, c]
                alpha_mask = alpha > 0
                image[y_min:y_max, x_min:x_max, c] = np.where(alpha_mask,
                    np.where(existing_alpha > 0,
                        (existing_color * existing_alpha + lit_color[:, :, c] * alpha) / np.maximum(new_alpha, 1e-6),
                        lit_color[:, :, c]),
                    existing_color)
            
            image[y_min:y_max, x_min:x_max, 3] = new_alpha
    
    # Draw bonds between nearby atoms for better structure visualization
    # Only draw bonds for atoms that are close in 3D space (more efficient)
    if len(coords_scaled) > 1000:
        # For large structures, skip bond drawing for performance
        pass
    elif len(coords_scaled) > 1:
        bond_cutoff_angstrom = 2.0  # Typical covalent bond length
        bond_cutoff_px = bond_cutoff_angstrom * scale
        
        # Use a more efficient approach: only check nearby atoms
        # For each atom, find neighbors within bond cutoff
        bond_width_px = max(0.5, line_width * scale)
        
        # Sample atoms for bond drawing (every Nth atom) to improve performance
        sample_step = max(1, len(coords_scaled) // 5000)  # Limit to ~5000 bond checks
        
        for i in range(0, len(coords_scaled), sample_step):
            x1, y1 = coords_scaled[i]
            z1 = coords_rotated[i, 2]
            
            # Only check atoms that could be within bond distance
            # Use a simple distance check in 3D space
            for j in range(i + 1, min(i + 100, len(coords_scaled))):  # Limit search window
                x2, y2 = coords_scaled[j]
                z2 = coords_rotated[j, 2]
                
                # 3D distance
                dist_3d = np.sqrt((coords_rotated[i, 0] - coords_rotated[j, 0])**2 +
                                 (coords_rotated[i, 1] - coords_rotated[j, 1])**2 +
                                 (coords_rotated[i, 2] - coords_rotated[j, 2])**2)
                
                if dist_3d < bond_cutoff_angstrom and dist_3d > 0.5:
                    # Draw bond line
                    dist_2d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if dist_2d < w + h:  # Only if bond is potentially visible
                        # Average color of the two atoms
                        bond_color = (colors[i] + colors[j]) / 2.0
                        
                        # Draw line using Bresenham-like approach (simplified)
                        num_points = max(2, int(dist_2d))
                        x_line = np.linspace(x1, x2, num_points)
                        y_line = np.linspace(y1, y2, num_points)
                        
                        # Draw line segments
                        for px, py in zip(x_line, y_line):
                            px_int = int(px)
                            py_int = int(py)
                            if 0 <= px_int < w and 0 <= py_int < h:
                                # Draw small circle for line segment
                                dist_sq = (x_coords - px)**2 + (y_coords - py)**2
                                dist = np.sqrt(dist_sq)
                                line_mask = dist <= bond_width_px
                                
                                if line_mask.any():
                                    line_alpha = np.exp(-(dist**2) / (2 * (bond_width_px * 0.5)**2))
                                    line_alpha = np.where(line_mask, line_alpha * 0.4, 0)  # Semi-transparent bonds
                                    
                                    existing_alpha = image[:, :, 3]
                                    new_alpha = np.maximum(existing_alpha, line_alpha)
                                    
                                    for c in range(3):
                                        image[:, :, c] = np.where(line_mask,
                                            np.where(existing_alpha > 0,
                                                (image[:, :, c] * existing_alpha + bond_color[c] * line_alpha) / np.maximum(new_alpha, 1e-6),
                                                bond_color[c]),
                                            image[:, :, c])
                                    
                                    image[:, :, 3] = new_alpha
    
    # Post-processing: enhance contrast and saturation for structural appearance
    # Normalize alpha to [0, 1]
    image[:, :, 3] = np.clip(image[:, :, 3], 0, 1)
    
    # Apply contrast and saturation boost to RGB channels (only where there's structure)
    alpha_mask = image[:, :, 3] > 0.01
    if alpha_mask.any():
        for c in range(3):
            rgb = image[:, :, c]
            # Gamma correction for more vibrant, structural appearance
            rgb_enhanced = np.power(np.clip(rgb, 0, 1), 0.85)
            # Slight contrast boost
            rgb_enhanced = np.clip((rgb_enhanced - 0.5) * 1.1 + 0.5, 0, 1)
            image[:, :, c] = np.where(alpha_mask, rgb_enhanced, rgb)
        
        # Add subtle edge enhancement for more structural definition
        # This creates a slight outline effect that makes the structure pop
        from scipy.ndimage import gaussian_filter
        alpha_smooth = gaussian_filter(image[:, :, 3].astype(float), sigma=0.5)
        edge_enhance = np.clip((image[:, :, 3] - alpha_smooth) * 0.3, 0, 1)
        for c in range(3):
            image[:, :, c] = np.where(alpha_mask, 
                np.clip(image[:, :, c] + edge_enhance * 0.1, 0, 1), 
                image[:, :, c])
    
    # Flip vertically (image coordinates vs data coordinates)
    image = np.flipud(image)
    
    return image

