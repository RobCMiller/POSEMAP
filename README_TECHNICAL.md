# POSEMAP - Technical Documentation

**Pose-Oriented Single-particle EM Micrograph Annotation & Projection**

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Code Structure](#code-structure)
3. [Core Algorithms](#core-algorithms)
4. [API Reference](#api-reference)
5. [Coordinate Systems](#coordinate-systems)
6. [Rendering Pipeline](#rendering-pipeline)
7. [Development Guide](#development-guide)

---

## Architecture Overview

POSEMAP is a modular Python toolkit for mapping 3D cryo-EM reconstructions and atomic models back onto their originating micrographs using particle pose metadata from cryoSPARC. The system consists of two main components:

1. **Core Library** (`particle_mapper.py`): Low-level functions for data loading, coordinate transformations, and projection generation
2. **GUI Application** (`particle_mapper_gui.py`): Interactive Tkinter-based interface for visualization and annotation

### Design Principles

- **Modularity**: Core functions are independent and can be used in scripts or integrated into pipelines
- **Type Safety**: Extensive use of type hints for better IDE support and documentation
- **Coordinate System Consistency**: Rigorous handling of coordinate system transformations
- **Performance**: Background threading for projection generation, caching for fast toggling

---

## Code Structure

### `particle_mapper.py` - Core Library

**Data Loading Functions:**
- `load_cs_file()`: Load cryoSPARC .cs files as numpy structured arrays
- `match_particles()`: Match particles between refinement and passthrough files by UID
- `load_volume()`: Load 3D MRC volumes with pixel size metadata
- `load_pdb_structure()`: Parse PDB/mmCIF files and extract atomic coordinates

**Coordinate Transformation Functions:**
- `euler_to_rotation_matrix()`: Convert ZYZ Euler angles to 3x3 rotation matrices
- `fractional_to_pixel_coords()`: Convert fractional (0-1) to pixel coordinates
- `get_particle_orientation_arrow()`: Calculate 2D projection of 3D viewing direction
- `get_particle_axes()`: Get X, Y, Z axis directions for visualization

**Projection Functions:**
- `project_volume()`: Project 3D MRC volumes at specified Euler angles using trilinear interpolation
- `project_pdb_structure()`: High-level wrapper for PDB projection (dispatches to PyMOL or ChimeraX)
- `project_pdb_cartoon_pymol()`: Generate 2D projections of atomic structures using PyMOL

**Utility Functions:**
- `is_nucleic_acid()`: Identify nucleic acid residues by name
- `get_chain_colors()`: Generate color mappings for protein/nucleic acid chains

### `particle_mapper_gui.py` - GUI Application

**Main Class: `ParticleMapperGUI`**

**Initialization:**
- Auto-detects cryoSPARC files, PDB structures, and micrograph directories
- Sets up Tkinter root window and layout (left panel controls, right panel display)
- Initializes matplotlib canvas for micrograph display
- Creates projection preview window (initially hidden)

**Data Management:**
- `load_all_data()`: Loads and matches particle data from .cs files
- `load_micrograph()`: Loads individual micrograph and filters particles
- Projection caching system with thread-safe access

**Rendering Pipeline:**
- `_generate_pdb_projection_for_particle()`: Generate single particle projection
- `_generate_all_projections()`: Batch generation with background threading
- `_start_background_generation()`: Background preloading after user inactivity
- `update_display()`: Main display update function with zoom preservation

**ChimeraX Integration:**
- `open_chimerax()`: Launch ChimeraX with particle-specific view
- `_euler_to_chimerax_view()`: Convert Euler angles to ChimeraX view matrix
- Handles window geometry preservation to prevent GUI resizing

**Event Handlers:**
- `on_canvas_click()`: Handle particle selection and ChimeraX opening
- `on_canvas_release()`: Handle zoom box selection
- `toggle_projections()`, `toggle_orientations()`, `toggle_outlines()`: UI toggles

---

## Core Algorithms

### Euler Angle to Rotation Matrix Conversion

**Convention**: ZYZ (phi, theta, psi) in radians, as used by cryoSPARC

```python
R = Rotation.from_euler('ZYZ', [phi, theta, psi], degrees=False).as_matrix()
```

**Physical Interpretation:**
- `phi`: First rotation around Z-axis
- `theta`: Rotation around Y-axis  
- `psi`: Second rotation around Z-axis

**Coordinate System:**
- Z-axis: Down the electron beam (toward viewer in projection)
- Y-axis: Up
- X-axis: Right

### Volume Projection Algorithm

1. **Create 2D coordinate grid** for output projection
2. **Scale coordinates** to match volume dimensions
3. **Create 3D coordinates** with z=0 (projection plane)
4. **Apply inverse rotation** (R^T) to transform from view space to volume space
5. **Translate to volume center**
6. **Sample volume** using trilinear interpolation
7. **Reshape to 2D** projection image

**Key Insight**: The rotation matrix R rotates from volume space to view space. To project, we need the inverse transformation (R^T) to map projection plane coordinates back to volume coordinates.

### PDB Structure Projection (PyMOL)

**Rendering Method**: Fast pseudo-surface using CA/P spheres

1. **Extract atoms**: CA atoms for protein, P atoms for nucleic acids
2. **Create PyMOL script**:
   - Load structure file
   - Create pseudo-surface object (CA/P spheres)
   - Apply colors (chain-specific or default protein/nucleic)
   - Apply rotation via sequential Z-Y-Z rotations
   - Render to PNG with transparent background
3. **Execute PyMOL** in headless mode (`-cq` flags)
4. **Load rendered image** and convert to RGBA array

**Rotation Application**: Sequential rotations match the ZYZ Euler convention:
```python
cmd.rotate("z", phi_deg, object=obj)
cmd.rotate("y", theta_deg, object=obj)
cmd.rotate("z", psi_deg, object=obj)
```

### ChimeraX View Matrix Conversion

**Challenge**: ChimeraX uses a view matrix that rotates the scene/object, not the camera.

**Solution**: 
1. Convert Euler angles to rotation matrix R (same as PyMOL)
2. Apply 180° Z-axis rotation correction: `R_chimerax = R @ R_180_z`
3. Format as 12-number matrix string: `r11,r12,r13,tx,r21,r22,r23,ty,r31,r32,r33,tz`
4. Use ChimeraX command: `view matrix camera {matrix_str}`

**Centering**: Object is centered before and after view matrix application to ensure proper positioning.

---

## API Reference

### Core Functions

#### `load_cs_file(cs_file_path: str) -> np.ndarray`
Load a cryoSPARC .cs file as a numpy structured array.

**Parameters:**
- `cs_file_path`: Path to .cs file

**Returns:**
- Structured numpy array with fields from cryoSPARC

#### `match_particles(refinement_cs: np.ndarray, passthrough_cs: np.ndarray) -> Dict`
Match particles between refinement and passthrough files by UID.

**Returns:**
- Dictionary with keys:
  - `poses`: (N, 3) array of Euler angles [phi, theta, psi] in radians
  - `shifts`: (N, 2) array of 2D shifts
  - `micrograph_paths`: (N,) array of micrograph file paths
  - `center_x_frac`, `center_y_frac`: (N,) arrays of fractional coordinates (0-1)
  - `micrograph_shapes`: (N, 2) array of micrograph dimensions
  - `uids`: (N,) array of particle UIDs

#### `project_volume(volume: np.ndarray, euler_angles: np.ndarray, output_size: Optional[Tuple[int, int]] = None, pixel_size: float = 1.0) -> np.ndarray`
Project a 3D volume at given Euler angles.

**Parameters:**
- `volume`: 3D numpy array (shape: [z, y, x])
- `euler_angles`: (3,) array [phi, theta, psi] in radians (ZYZ convention)
- `output_size`: Optional (height, width) tuple. Defaults to volume size.
- `pixel_size`: Pixel size in Angstroms (for metadata, not used in calculation)

**Returns:**
- 2D numpy array (grayscale projection)

#### `project_pdb_structure(pdb_data: Dict, euler_angles: np.ndarray, output_size: Tuple[int, int] = (500, 500), chain_color_map: Optional[Dict[str, str]] = None, default_protein_color: str = '#007CBE', default_nucleic_color: str = '#3B1F2B', pdb_path: Optional[str] = None, pymol_path: Optional[str] = None, chimerax_path: Optional[str] = None) -> np.ndarray`
Project a PDB structure at given Euler angles.

**Parameters:**
- `pdb_data`: Dictionary from `load_pdb_structure()`
- `euler_angles`: (3,) array [phi, theta, psi] in radians
- `output_size`: (height, width) tuple for output image
- `chain_color_map`: Optional dict mapping chain IDs to hex colors
- `default_protein_color`: Hex color for protein chains (default: '#007CBE')
- `default_nucleic_color`: Hex color for nucleic acid chains (default: '#3B1F2B')
- `pdb_path`: Path to structure file (required)
- `pymol_path`: Optional path to PyMOL executable
- `chimerax_path`: Optional path to ChimeraX executable (not currently used)

**Returns:**
- (H, W, 4) RGBA numpy array in [0, 1] range

---

## Coordinate Systems

### cryoSPARC Coordinate System

**Euler Angles (ZYZ convention):**
- `phi`: First Z-axis rotation (radians)
- `theta`: Y-axis rotation (radians)
- `psi`: Second Z-axis rotation (radians)

**Particle Locations:**
- Stored as fractional coordinates (0-1) in passthrough file
- `center_x_frac`: Fractional X coordinate (0 = left, 1 = right)
- `center_y_frac`: Fractional Y coordinate (0 = bottom, 1 = top)

**Micrograph Coordinates:**
- Origin at bottom-left (image coordinates)
- X-axis: left to right
- Y-axis: bottom to top

### Internal Coordinate Systems

**Volume Space:**
- Array indexing: `[z, y, x]` (numpy convention)
- Origin at volume center
- Units: Angstroms

**View Space (Projection):**
- Z-axis: Toward viewer (down beam)
- Y-axis: Up
- X-axis: Right
- Projection plane: z=0

**Transformation Chain:**
1. Fractional coordinates → Pixel coordinates (micrograph)
2. Euler angles → Rotation matrix (volume to view)
3. View coordinates → Volume coordinates (inverse rotation for projection)

---

## Rendering Pipeline

### PyMOL Rendering Workflow

1. **Structure Loading**: Parse PDB/mmCIF file using BioPython
2. **Atom Selection**: Extract CA atoms (protein) and P atoms (nucleic)
3. **Script Generation**: Create PyMOL Python script with:
   - Structure loading
   - Pseudo-surface creation (large overlapping spheres)
   - Color application (chain-specific or defaults)
   - Rotation application (sequential Z-Y-Z)
   - Rendering settings (ray tracing, lighting, transparency)
   - PNG output with transparent background
4. **Execution**: Run PyMOL in headless mode (`-cq` flags)
5. **Image Processing**: Load PNG, convert to RGBA array, normalize to [0, 1]

### Caching Strategy

**Cache Key**: `(micrograph_idx, particle_idx)`

**Cache Invalidation:**
- Color changes: Clear entire cache
- Size changes: Clear entire cache (projections need regeneration)
- Micrograph change: Clear cache for old micrograph

**Background Preloading:**
- After 7 seconds of user inactivity, start preloading projections
- Uses `ThreadPoolExecutor` for parallel generation
- Updates display as projections complete

---

## Development Guide

### Setting Up Development Environment

```bash
# Clone repository
git clone <repo_url>
cd posemap

# Create conda environment
conda create -n posemap python=3.9 -y
conda activate posemap

# Install dependencies
pip install -r requirements.txt

# Install PyMOL
conda install -c conda-forge pymol-open-source -y
```

### Running Tests

```bash
# Test basic functionality
python test_setup.py

# Test PyMOL rendering
python test_pymol_render.py
```

### Code Style

- Follow PEP 8
- Use type hints for all function signatures
- Document all public functions with docstrings
- Use descriptive variable names

### Adding New Features

**Adding a new rendering backend:**

1. Create new function in `particle_mapper.py`: `project_pdb_<backend>()`
2. Update `project_pdb_structure()` to dispatch to new function
3. Add backend detection logic
4. Update GUI to support new backend

**Adding new visualization options:**

1. Add GUI controls in `setup_left_panel()`
2. Store state in `__init__()`
3. Update `update_display()` to use new option
4. Clear cache if needed when option changes

### Debugging

**Enable debug output:**
- Set environment variable: `POSEMAP_DEBUG=1`
- Check console output for detailed logging

**Common Issues:**

1. **PyMOL not found**: Check PATH, verify installation with `which pymol`
2. **Projections not appearing**: Check cache, verify Euler angles are valid
3. **ChimeraX view mismatch**: Verify 180° rotation correction is applied
4. **GUI resizing**: Check window geometry restoration logic

---

## Performance Considerations

**Projection Generation:**
- Single projection: ~1-3 seconds (depends on structure size)
- Parallel generation: Uses `ThreadPoolExecutor` with 4 workers
- Caching: Prevents regeneration of identical projections

**Memory Usage:**
- Cached projections: ~1-5 MB per projection (depends on size)
- Typical session: 50-200 MB for cached projections

**Optimization Opportunities:**
- Use smaller projection sizes for faster rendering
- Reduce ray tracing quality for speed
- Implement projection downsampling for preview

---

## External Dependencies

### Required
- **PyMOL**: For structure rendering (`conda install -c conda-forge pymol-open-source`)
- **NumPy**: Array operations
- **SciPy**: Rotation matrices, interpolation
- **mrcfile**: MRC file I/O
- **matplotlib**: GUI display
- **Pillow**: Image processing

### Optional
- **ChimeraX**: For interactive structure viewing (download from UCSF)
- **BioPython**: Structure file parsing (recommended)
- **scikit-image**: Advanced image processing (for contour detection)

---

## Version History

**v1.0.0** (Current)
- Initial release
- PyMOL-based structure rendering
- ChimeraX integration
- Interactive GUI with projection overlays
- Background preloading
- Projection caching

---

## License

[To be determined]

---

## Contact

[To be added]

