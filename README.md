# POSEMAP 🕺🗺️

**Pose-Oriented Single-particle EM Micrograph Annotation & Projection**

POSEMAP is a Python toolkit for mapping cryo-EM reconstructions and corresponding atomic models back onto their originating micrographs using particle pose metadata from cryoSPARC refinement files (.star file integration coming soon!). The code takes in cryoSPARC `.cs` files containing particle poses (Euler angles in ZYZ convention, 2D shifts, CTF parameters) as inputs, generates pose-consistent 2D projections of 3D structures using the PyMOL API, and overlays these projections onto micrographs for visualization, quality control, and downstream analysis. POSEMAP is designed as a modular Python library that can be integrated into existing cryo-EM processing pipelines or used interactively via its Tkinter-based GUI.

## Features

- **Pose-Consistent Projections**: Generate 2D projections of 3D structures (volumes or atomic models) at each particle's specific 3D orientation
- **Interactive Visualization**: Tkinter-based GUI with micrograph display, particle overlays, and real-time projection preview
- **Multiple Rendering Backends**: PyMOL for high-quality structure rendering, EMAN2 for accurate EM projections, with ChimeraX integration for interactive viewing
- **Flexible Coloring**: Chain-based coloring with custom color support for protein and nucleic acid chains
- **Background Processing**: Automatic preloading of projections with thread-safe caching
- **Coordinate System Accuracy**: Rigorous handling of ZYZ Euler angle conventions and coordinate transformations

## Installation

### Quick Install (Recommended)

```bash
# Clone the repository
git clone <repository_url>
cd posemap

# Run the install script
bash install.sh
```

### Manual Installation

#### 1. Create Conda Environment

```bash
conda create -n posemap python=3.9 -y
conda activate posemap
```

#### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Install PyMOL

```bash
conda install -c conda-forge pymol-open-source -y
```

#### 4. (Optional) Install EMAN2 (Recommended for Higher Quality Projections)

```bash
conda install -c cryoem -c conda-forge eman-dev -y
```

POSEMAP will automatically use EMAN2 for projections when available, providing higher quality results. If EMAN2 is not installed, POSEMAP will fall back to a NumPy-based projection method.

#### 5. (Optional) Install ChimeraX

Download from [UCSF ChimeraX](https://www.rbvi.ucsf.edu/chimerax/) and add to PATH.

### Verify Installation

```bash
python test_setup.py
```

## Quick Start

### GUI Application

Launch the interactive GUI:

```bash
python particle_mapper_gui.py
```

The GUI will auto-detect cryoSPARC files, PDB structures, and micrograph directories in the current working directory. Alternatively, specify paths manually:

```bash
python particle_mapper_gui.py \
    --refinement-cs refinement_particles.cs \
    --passthrough-cs passthrough_particles.cs \
    --pdb ref_volume/structure.cif \
    --micrograph-dir ref_movies
```

### Python API

```python
from particle_mapper import load_cs_file, match_particles, project_pdb_structure, load_pdb_structure

# Load cryoSPARC data
refinement_cs = load_cs_file('refinement_particles.cs')
passthrough_cs = load_cs_file('passthrough_particles.cs')
matched_data = match_particles(refinement_cs, passthrough_cs)

# Load structure
pdb_data = load_pdb_structure('structure.cif')

# Generate projection for first particle
euler_angles = matched_data['poses'][0]  # [phi, theta, psi] in radians
projection = project_pdb_structure(
    pdb_data, 
    euler_angles, 
    output_size=(500, 500),
    pdb_path='structure.cif'
)
```

## Input Data Format

POSEMAP requires the following input files:

1. **Refinement .cs file**: Contains particle poses and alignment data
   - Required fields: `uid`, `alignments3D/pose`, `alignments3D/shift`
   - Euler angles in ZYZ convention (radians)

2. **Passthrough .cs file**: Contains particle pick locations
   - Required fields: `uid`, `location/micrograph_path`, `location/center_x_frac`, `location/center_y_frac`

3. **Structure file**: Atomic model (.pdb or .cif format)
   - Used for generating 3D projections

4. **Micrograph files**: Motion-corrected micrographs (.mrc format)
   - Stored in a directory specified by user

## GUI Features

### Main Display
- **Micrograph View**: Large display area with zoom and pan tools
- **Particle Overlays**: 3D structure projections overlaid at particle locations
- **Orientation Arrows**: Visual indicators showing particle viewing directions
- **Particle Markers**: Star markers at particle centers

### Controls
- **Projection Settings**: Adjust size, transparency, and visibility
- **Color Customization**: Chain-based coloring with custom color support
- **Image Enhancement**: Low-pass filtering, brightness, and contrast adjustment
- **ChimeraX Integration**: Open structures in ChimeraX with particle-specific views

### Visualization Options
- Toggle projections on/off
- Toggle orientation arrows
- Toggle particle outlines
- Adjust projection alpha (transparency)
- Adjust projection size
- Adjust arrow length

## Coordinate Systems and Projection Mapping

POSEMAP uses the following coordinate conventions:

- **Euler Angles**: ZYZ convention (phi, theta, psi) in radians, as used by cryoSPARC
- **Particle Locations**: Fractional coordinates (0-1) converted to pixel coordinates
- **Micrograph Coordinates**: Origin at bottom-left, X right, Y up (Matplotlib convention)
- **Volume Space**: Z-axis down beam, Y-axis up, X-axis right
- **Model Space**: PDB/mmCIF coordinate system (Angstroms)
- **View Space**: 3D coordinate system after rotation, with projection plane at Z=0

### Detailed Projection Mapping Pipeline

The projection mapping process transforms 3D atomic model coordinates to 2D micrograph coordinates through the following steps:

#### 1. **Particle Pose Extraction from cryoSPARC**
   - Euler angles `[phi, theta, psi]` in ZYZ convention (radians) from `alignments3D/pose`
   - Particle center coordinates `[center_x_frac, center_y_frac]` as fractional positions (0-1) from `location/center_*_frac`
   - 2D refinement shifts `[shift_x, shift_y]` in Angstroms from `alignments3D/shift`
   - Pixel size in Angstroms from `location/micrograph_psize_A`

#### 2. **Rotation Matrix Calculation**
   - Convert Euler angles to rotation matrix `R` using `scipy.spatial.transform.Rotation.from_euler('ZYZ', [phi, theta, psi])`
   - `R` rotates coordinates **from model/volume space to view space**
   - The rotation matrix represents the orientation of the particle as viewed in the micrograph
   - **Important**: `scipy` uses extrinsic ZYZ convention, equivalent to `Rz(psi) @ Ry(theta) @ Rz(phi)`

#### 3. **Particle Center Calculation**
   - Convert fractional coordinates to pixel coordinates:
     ```
     x_pixel = center_x_frac * micrograph_width
     y_pixel = center_y_frac * micrograph_height
     ```
   - Apply 2D refinement shifts (converted from Angstroms to pixels):
     ```
     x_pixel += shift_x / pixel_size
     y_pixel += shift_y / pixel_size
     ```
   - This gives the final particle center position in micrograph pixel coordinates
   - **Note**: Shifts are applied in the same direction as the particle center (not negated)

#### 4. **3D Structure Projection (PyMOL)**
   - Load atomic model (PDB/mmCIF) and center at its center of mass (COM)
   - Calculate COM as mean of all atom coordinates: `COM = mean(coords, axis=0)`
   - Apply rotation matrix `R` using PyMOL's `transform_object`:
     - The structure is centered at COM before rotation using `cmd.center()`
     - Origin is set to COM using `cmd.origin()`
     - Rotation is applied around the origin (COM) using `transform_object` with 4x4 transformation matrix
     - This rotates the structure from model space to view space
   - Render 2D projection using PyMOL's ray tracing with `cmd.png()`
   - The projection image center corresponds to the structure's COM after rotation
   - **Critical**: PyMOL's `cmd.zoom(complete=1)` scales the structure to fit the viewport based on the **rotated** bounding box, not the original model size

#### 5. **Projection Size Calculation**
   - Base projection size is calculated from original model dimensions:
     ```
     model_size = max(max_coords - min_coords)  # Maximum extent in any dimension
     projection_size = (model_size / pixel_size) * 1.2  # 1.2x padding factor
     ```
   - However, PyMOL's `zoom(complete=1)` scales based on the **rotated** bounding box
   - The rotated bounding box size can differ from the original model size:
     ```
     rotated_bbox_size = max distance from center in XY plane * 2
     ```
   - This creates a potential scaling mismatch that affects marker coordinate transformations
   - **Effective pixel size** accounts for this:
     ```
     effective_pixel_size = rotated_bbox_size / (projection_size / 1.2)
     ```

#### 6. **Projection Placement on Micrograph**
   - The projection image is placed centered at the particle center coordinates:
     ```
     projection_extent = [
         x_pixel - projection_size/2,  # left
         x_pixel + projection_size/2,  # right
         y_pixel - projection_size/2,  # bottom
         y_pixel + projection_size/2   # top
     ]
     ```
   - Fine-tuning offsets can be applied for manual alignment:
     ```
     center_x = x_pixel + offset_x
     center_y = y_pixel + offset_y
     ```
   - The projection is overlaid using RGBA alpha blending

#### 7. **Marker Coordinate Transformation (ChimeraX Markers)**
   - Markers are defined in model space (PDB coordinates) from ChimeraX
   - Transform to micrograph coordinates:
     1. **Center**: `marker_centered = marker_abs - structure_COM`
     2. **Rotate**: `marker_rotated = R @ marker_centered`
     3. **Project to 2D**: Use XY components of rotated marker
     4. **Convert to pixels**: 
        ```
        marker_x_pixels = marker_rotated[1] / effective_pixel_size  # Note: X and Y are swapped
        marker_y_pixels = marker_rotated[0] / effective_pixel_size
        ```
     5. **Place on micrograph**: Add to particle center coordinates
   - **Critical Coordinate Swap**: PyMOL's image coordinate system swaps X and Y relative to view space:
     - View space X → Image Y
     - View space Y → Image X
   - This swap is necessary because PyMOL renders with a different coordinate convention than our micrograph display

#### 8. **Coordinate System Consistency**
   - **Critical Assumption**: The atomic model's COM in PDB coordinates should match the volume center used by cryoSPARC
   - If the model is perfectly fitted into the cryoSPARC map, the coordinate systems are aligned
   - PyMOL centers the structure at COM, so after rotation, the COM remains at the projection image center
   - The projection is placed so the COM aligns with the particle center from cryoSPARC
   - Any misalignment suggests either:
     - Imperfect pose estimates from cryoSPARC refinement
     - Coordinate system mismatch between PDB and cryoSPARC volume
     - Need for fine-tuning offsets (available in GUI)

#### 9. **Volume Projection (Reference Implementation)**
   For comparison, volume projection uses the inverse transformation:
   - Create 2D grid in projection plane (z=0 in view space)
   - Rotate coordinates **from view space back to volume space** using `R.T` (transpose = inverse for rotation matrices)
   - Translate to volume center and sample the volume
   - This confirms that `R` correctly transforms from model/volume space to view space

### Known Issues and Potential Problems

#### 1. **Projection Size Mismatch**
   - **Problem**: `projection_size` is calculated from original model size, but PyMOL scales based on rotated bbox
   - **Impact**: Marker coordinates may require offsets if not using `effective_pixel_size`
   - **Solution**: Use `effective_pixel_size` calculated from `rotated_bbox_size` for marker transformations

#### 2. **Coordinate System Swaps**
   - **Problem**: PyMOL's image rendering swaps X and Y coordinates relative to view space
   - **Impact**: Marker coordinates must swap X and Y: `image_x = view_y`, `image_y = view_x`
   - **Solution**: Apply coordinate swap in marker transformation code

#### 3. **Center of Mass Offset**
   - **Problem**: PDB COM may not exactly match cryoSPARC volume center
   - **Impact**: Small systematic offsets in projection placement
   - **Solution**: Fine-tuning sliders in GUI, or automatic COM offset correction (if PDB-to-volume transformation is known)

#### 4. **Pose Estimation Errors**
   - **Problem**: cryoSPARC pose estimates may not be perfect for all particles
   - **Impact**: Some particles show misalignment even with correct coordinate transformations
   - **Solution**: This is expected and indicates imperfect refinement - use for quality control

#### 5. **Pixel Size Inconsistencies**
   - **Problem**: Different pixel sizes in refinement vs. passthrough files
   - **Impact**: Incorrect scaling if wrong pixel size is used
   - **Solution**: Always use pixel size from passthrough file (`location/micrograph_psize_A`)

#### 6. **Sub-pixel Accuracy**
   - **Problem**: Particle centers and shifts may have sub-pixel precision
   - **Impact**: Rounding errors in pixel coordinate calculations
   - **Solution**: Use float coordinates throughout, only round for final display

### Verification Methods

#### Side-by-Side Comparison
   - Extract actual micrograph region (square box centered at particle center)
   - Compare with simulated PyMOL projection
   - Perfect alignment indicates correct coordinate transformations
   - Misalignment suggests pose errors or coordinate system issues

#### Marker Alignment
   - Place markers in ChimeraX at known structural features
   - Verify markers align with same features in projection overlay
   - Systematic offsets indicate coordinate transformation errors
   - Random offsets indicate pose estimation errors

## Rendering

### PyMOL Rendering

POSEMAP uses PyMOL for high-quality structure rendering:

- **Method**: Fast pseudo-surface rendering using CA/P spheres
- **Style**: Large overlapping spheres for continuous surface appearance
- **Colors**: Chain-specific coloring with defaults for protein (#007CBE) and nucleic acid (#3B1F2B)
- **Rotation**: Sequential Z-Y-Z rotations matching cryoSPARC Euler angles

### ChimeraX Integration

Click on a particle to open it in ChimeraX with the correct orientation:

- Automatic view matrix calculation from Euler angles
- Proper centering of structure in view
- Custom colors matching GUI settings

## Performance

- **Projection Generation**: ~1-3 seconds per particle (depends on structure size)
- **Caching**: Projections are cached for fast toggling
- **Background Preloading**: Automatic preloading after 7 seconds of inactivity
- **Memory Usage**: ~1-5 MB per cached projection

## Requirements

- Python 3.9+
- NumPy, SciPy
- matplotlib, Pillow
- mrcfile, h5py
- BioPython (for structure parsing)
- PyMOL (installed via conda)
- ChimeraX (optional, for interactive viewing)

See `requirements.txt` for complete list.

## Documentation

- **Technical Documentation**: See `README_TECHNICAL.md` for detailed architecture and API documentation
- **Code Comments**: Extensive inline documentation in source files

## Contributing

[To be added]

## License

[To be determined]

## Citation

If you use POSEMAP in your research, please cite:

[Citation to be added]

## Acknowledgments

Built for cryo-EM structure analysis and visualization.
