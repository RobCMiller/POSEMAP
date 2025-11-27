#!/usr/bin/env python3
"""
Setup script for POSEMAP
Pose-Oriented Single-particle EM Micrograph Annotation & Projection
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="posemap",
    version="1.0.0",
    description="Pose-Oriented Single-particle EM Micrograph Annotation & Projection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Davis Lab",
    author_email="",  # To be filled
    url="",  # To be filled
    packages=find_packages(),
    py_modules=["particle_mapper", "particle_mapper_gui", "read_cs_file"],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "posemap-gui=particle_mapper_gui:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="cryo-EM, cryoSPARC, structure visualization, particle picking, electron microscopy",
    project_urls={
        "Documentation": "",  # To be filled
        "Source": "",  # To be filled
        "Bug Reports": "",  # To be filled
    },
)

