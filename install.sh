#!/bin/bash
# POSEMAP Installation Script
# Pose-Oriented Single-particle EM Micrograph Annotation & Projection

set -e  # Exit on error

echo "=========================================="
echo "POSEMAP Installation Script"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda is not installed or not in PATH."
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Environment name
ENV_NAME="posemap"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment. Activating..."
        eval "$(conda shell.bash hook)"
        conda activate ${ENV_NAME}
        echo "Installing/updating packages in existing environment..."
    fi
fi

# Create environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment '${ENV_NAME}' with Python 3.9..."
    conda create -n ${ENV_NAME} python=3.9 -y
fi

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install Python dependencies
echo ""
echo "Installing Python dependencies from requirements.txt..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found. Installing core dependencies..."
    pip install numpy scipy matplotlib pillow mrcfile h5py biopython scikit-image
fi

# Install PyMOL
echo ""
echo "Installing PyMOL..."
if conda install -c conda-forge pymol-open-source -y; then
    echo "PyMOL installed successfully."
else
    echo "WARNING: Failed to install PyMOL via conda."
    echo "You may need to install it manually:"
    echo "  conda install -c conda-forge pymol-open-source"
fi

# Verify PyMOL installation
echo ""
echo "Verifying PyMOL installation..."
if command -v pymol &> /dev/null; then
    PYMOl_VERSION=$(pymol --version 2>&1 | head -n 1)
    echo "✓ PyMOL found: ${PYMOl_VERSION}"
else
    echo "⚠ WARNING: PyMOL not found in PATH."
    echo "  POSEMAP will try to auto-detect PyMOL, but you may need to:"
    echo "  1. Install PyMOL: conda install -c conda-forge pymol-open-source"
    echo "  2. Or specify PyMOL path in the GUI"
fi

# Install EMAN2 (optional, for higher quality projections)
echo ""
echo "Installing EMAN2 (optional, for higher quality projections)..."
read -p "Do you want to install EMAN2? This will improve projection quality. (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if conda install -c cryoem -c conda-forge eman-dev -y; then
        echo "✓ EMAN2 installed successfully."
        echo "  POSEMAP will use EMAN2 for projections when available."
    else
        echo "⚠ WARNING: Failed to install EMAN2 via conda."
        echo "  POSEMAP will fall back to NumPy-based projections."
        echo "  To install EMAN2 manually:"
        echo "    conda install -c cryoem -c conda-forge eman-dev"
    fi
else
    echo "⚠ Skipping EMAN2 installation."
    echo "  POSEMAP will use NumPy-based projections (slower but functional)."
    echo "  To install EMAN2 later:"
    echo "    conda install -c cryoem -c conda-forge eman-dev"
fi

# Verify EMAN2 installation
echo ""
echo "Verifying EMAN2 installation..."
python -c "from EMAN2 import EMData; print('✓ EMAN2 is available')" 2>/dev/null && echo "  POSEMAP will use EMAN2 for projections." || echo "  ⚠ EMAN2 not available - will use NumPy fallback."

# Check for ChimeraX (optional)
echo ""
echo "Checking for ChimeraX (optional)..."
if command -v chimerax &> /dev/null; then
    echo "✓ ChimeraX found in PATH"
elif [ -d "/Applications/ChimeraX.app" ]; then
    echo "✓ ChimeraX found at /Applications/ChimeraX.app"
else
    echo "⚠ ChimeraX not found (optional)."
    echo "  Download from: https://www.rbvi.ucsf.edu/chimerax/"
fi

# Run test script if available
echo ""
echo "Running installation test..."
if [ -f test_setup.py ]; then
    python test_setup.py
    if [ $? -eq 0 ]; then
        echo "✓ Installation test passed!"
    else
        echo "⚠ Installation test had warnings (this may be OK)"
    fi
else
    echo "⚠ test_setup.py not found, skipping test"
fi

# Summary
echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To use POSEMAP:"
echo "  1. Activate the environment:"
echo "     conda activate ${ENV_NAME}"
echo ""
echo "  2. Run the GUI:"
echo "     python particle_mapper_gui.py"
echo ""
echo "  3. Or use the Python API:"
echo "     python -c 'from particle_mapper import load_cs_file; print(\"POSEMAP loaded successfully\")'"
echo ""
echo "For more information, see README.md"
echo ""

