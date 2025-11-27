# POSEMAP Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    POSEMAP System                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Core Library    │         │   GUI Application │          │
│  │ particle_mapper  │◄────────┤ particle_mapper_ │          │
│  │      .py         │         │      gui.py      │          │
│  └──────────────────┘         └──────────────────┘          │
│         │                              │                     │
│         │                              │                     │
│         ▼                              ▼                     │
│  ┌──────────────────────────────────────────────┐           │
│  │         External Dependencies                 │           │
│  │  • PyMOL (rendering)                         │           │
│  │  • ChimeraX (viewing)                        │           │
│  │  • NumPy/SciPy (computations)                │           │
│  │  • matplotlib (display)                      │           │
│  └──────────────────────────────────────────────┘           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Data Loading Pipeline

```
cryoSPARC .cs files
    │
    ├─► refinement.cs ──┐
    │                    │
    └─► passthrough.cs ──┼─► match_particles() ──► matched_data
                         │
Structure file (.cif/.pdb) ──► load_pdb_structure() ──► pdb_data
                         │
Micrograph files (.mrc) ──► load_micrograph() ──► micrograph_image
```

### 2. Projection Generation Pipeline

```
For each particle:
    │
    ├─► Extract Euler angles [phi, theta, psi]
    │
    ├─► Convert to rotation matrix (ZYZ convention)
    │
    ├─► Generate PyMOL script:
    │   • Load structure
    │   • Create pseudo-surface (CA/P spheres)
    │   • Apply colors
    │   • Apply rotation (Z-Y-Z sequential)
    │   • Render to PNG
    │
    ├─► Execute PyMOL (headless)
    │
    └─► Load PNG → RGBA array → Cache
```

### 3. Display Pipeline

```
Cached projections
    │
    ├─► Overlay on micrograph at particle locations
    │
    ├─► Add particle markers (stars)
    │
    ├─► Add orientation arrows
    │
    ├─► Add outlines (optional)
    │
    └─► Display in matplotlib canvas
```

## Key Components

### Core Library (`particle_mapper.py`)

**Data Structures:**
- `matched_data`: Dictionary containing matched particle information
  - `poses`: (N, 3) array of Euler angles
  - `shifts`: (N, 2) array of 2D shifts
  - `center_x_frac`, `center_y_frac`: Fractional coordinates
  - `micrograph_paths`: File paths
  - `uids`: Particle identifiers

**Key Functions:**
- `load_cs_file()`: Load cryoSPARC files
- `match_particles()`: Match by UID
- `euler_to_rotation_matrix()`: ZYZ → rotation matrix
- `project_volume()`: 3D volume projection
- `project_pdb_structure()`: Atomic structure projection

### GUI Application (`particle_mapper_gui.py`)

**Main Class: `ParticleMapperGUI`**

**State Management:**
- `current_micrograph_idx`: Currently displayed micrograph
- `current_particles`: Particle data for current micrograph
- `projection_cache`: Dictionary mapping (micrograph_idx, particle_idx) → RGBA array
- `zoom_xlim`, `zoom_ylim`: Saved zoom state

**Threading:**
- Main thread: GUI event loop
- Background thread: Projection generation
- Thread-safe cache access via `cache_lock`

**Display Management:**
- `update_display()`: Main display update function
- `_cache_current_display()`: Cache rendered canvas
- Zoom preservation across toggles

## Coordinate System Transformations

### Transformation Chain

1. **cryoSPARC → Internal**
   - Fractional coords (0-1) → Pixel coords
   - Euler angles (ZYZ, radians) → Rotation matrix

2. **Volume Projection**
   - View space coords → Volume space coords (R^T)
   - Trilinear interpolation

3. **Structure Projection**
   - Rotation matrix → Sequential Z-Y-Z rotations
   - PyMOL object rotation

4. **ChimeraX View**
   - Rotation matrix → View matrix (with 180° Z correction)
   - Scene rotation

## Caching Strategy

**Cache Key**: `(micrograph_idx, particle_idx)`

**Cache Invalidation:**
- Color changes: Full cache clear
- Size changes: Full cache clear
- Micrograph change: Partial clear (old micrograph only)

**Cache Storage:**
- In-memory: RGBA arrays (H, W, 4) in [0, 1] range
- Thread-safe: Uses `threading.Lock()`

**Background Preloading:**
- Triggered after 7 seconds of user inactivity
- Uses `ThreadPoolExecutor` with 4 workers
- Updates display as projections complete

## Performance Optimizations

1. **Parallel Projection Generation**: ThreadPoolExecutor for batch processing
2. **Caching**: Avoids regeneration of identical projections
3. **Lazy Loading**: Projections generated on-demand
4. **Background Preloading**: Preloads after user inactivity
5. **Fast Toggle**: Cached images for instant projection on/off

## Error Handling

- **PyMOL Not Found**: Clear error message with installation instructions
- **ChimeraX Not Found**: Graceful degradation (optional feature)
- **File Not Found**: User-friendly error dialogs
- **Invalid Data**: Validation with helpful error messages

## Extension Points

**Adding New Rendering Backend:**
1. Create `project_pdb_<backend>()` function
2. Update `project_pdb_structure()` dispatch logic
3. Add backend detection

**Adding New Visualization:**
1. Add GUI controls
2. Update `update_display()` to use new visualization
3. Handle cache invalidation if needed

**Adding New File Format:**
1. Create loader function
2. Update auto-detection logic
3. Add to file browser filters

