# Repository Cleanup Guide

This document identifies files that should be cleaned up or organized before public release.

## Files to Remove

### Test/Debug Files
- `test_pymol_render.py` - Test script, can be moved to `tests/` directory
- `test_pymol_rendering.py` - Test script, can be moved to `tests/` directory
- `test_pymol_setcolor.py` - Test script, can be moved to `tests/` directory
- `test_pymol_color.py` - Test script, can be moved to `tests/` directory
- `test_transform.py` - Test script, can be moved to `tests/` directory
- `test_setup.py` - Keep but move to `tests/` directory

### Temporary/Generated Files
- `chimerax_script_*.cxc` - Temporary ChimeraX scripts (should be in .gitignore)
- `pymol_script_*.py` - Temporary PyMOL scripts (should be in .gitignore)
- `pymol_render_*.png` - Temporary render outputs (should be in .gitignore)
- `chimerax_render_*.png` - Temporary render outputs (should be in .gitignore)
- `test_*.png` - Test output images (should be in .gitignore)
- `test_render*.png` - Test output images (should be in .gitignore)
- `test_rotated.png` - Test output image (should be in .gitignore)
- `ribosome_pseudosurface*.png` - Example/test images (can be moved to `examples/` or removed)

### Data Files (User Data)
- `*.mrc` - Micrograph files (user data, should not be in repo)
- `*.cs` - cryoSPARC files (user data, should not be in repo)
- `*.cif` - Structure files (user data, should not be in repo)
- `*.pdb` - Structure files (user data, should not be in repo)
- `ref_movies/` - User data directory (should not be in repo)
- `ref_volume/` - User data directory (should not be in repo)
- `projection_images/` - User-generated images (should not be in repo)

### Documentation Files (To Review)
- `STATUS.md` - Development status file, can be removed or moved to `docs/`

## Recommended Directory Structure

```
posemap/
├── posemap/              # Main package (if making it a package)
│   ├── __init__.py
│   ├── particle_mapper.py
│   ├── particle_mapper_gui.py
│   └── read_cs_file.py
├── tests/                # Test scripts
│   ├── test_setup.py
│   ├── test_pymol_render.py
│   └── ...
├── examples/             # Example data and scripts
│   └── ...
├── docs/                 # Additional documentation
│   └── ...
├── README.md             # Public README
├── README_TECHNICAL.md   # Technical documentation
├── requirements.txt      # Python dependencies
├── setup.py              # Installation script
├── install.sh            # Quick install script
├── .gitignore            # Git ignore rules
└── LICENSE               # License file (to be added)
```

## Cleanup Commands

```bash
# Remove temporary files
rm -f chimerax_script_*.cxc
rm -f pymol_script_*.py
rm -f pymol_render_*.png
rm -f chimerax_render_*.png
rm -f test_*.png
rm -f test_render*.png
rm -f ribosome_pseudosurface*.png

# Remove user data (BE CAREFUL - backup first!)
# rm -rf ref_movies/
# rm -rf ref_volume/
# rm -f *.mrc *.cs *.cif *.pdb

# Create directory structure
mkdir -p tests examples docs

# Move test files
mv test_*.py tests/ 2>/dev/null || true
```

## Files to Keep

### Core Code
- `particle_mapper.py` - Core library
- `particle_mapper_gui.py` - GUI application
- `read_cs_file.py` - Utility script

### Documentation
- `README.md` - Public README
- `README_TECHNICAL.md` - Technical documentation
- `CLEANUP.md` - This file (can be removed after cleanup)

### Configuration
- `requirements.txt` - Dependencies
- `setup.py` - Installation script
- `install.sh` - Quick install script
- `.gitignore` - Git ignore rules

## Notes

- All user data files (*.mrc, *.cs, *.cif, *.pdb) should be excluded via .gitignore
- Temporary files generated during runtime are already in .gitignore
- Test files can be kept but should be organized in a tests/ directory
- Consider creating example data files (small, anonymized) for testing

