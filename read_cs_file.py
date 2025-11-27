#!/usr/bin/env python3
"""
Read cryosparc .cs file and extract data as numpy arrays.
Cryosparc .cs files are NumPy structured arrays containing particle data.
"""

import numpy as np
import sys
from pathlib import Path


def read_cs_file(cs_file_path):
    """
    Read a cryosparc .cs file and return data as numpy structured array.
    
    Parameters:
    -----------
    cs_file_path : str or Path
        Path to the .cs file
        
    Returns:
    --------
    numpy.ndarray : Structured array containing all particle data
    """
    cs_file_path = Path(cs_file_path)
    
    if not cs_file_path.exists():
        raise FileNotFoundError(f"File not found: {cs_file_path}")
    
    # Load the numpy structured array
    data = np.load(cs_file_path, allow_pickle=True)
    
    return data


def print_data_summary(data):
    """Print a summary of the loaded data."""
    print("\n" + "="*60)
    print("Data Summary:")
    print("="*60)
    
    print(f"\nOverall:")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Number of particles: {len(data)}")
    
    # Print field information for structured arrays
    if data.dtype.names:
        print(f"\nFields ({len(data.dtype.names)}):")
        for field_name in data.dtype.names:
            field_data = data[field_name]
            print(f"\n  {field_name}:")
            print(f"    Shape: {field_data.shape}")
            print(f"    Dtype: {field_data.dtype}")
            if field_data.size > 0:
                if np.issubdtype(field_data.dtype, np.number):
                    print(f"    Min: {np.nanmin(field_data)}")
                    print(f"    Max: {np.nanmax(field_data)}")
                    print(f"    Mean: {np.nanmean(field_data)}")
                if field_data.size < 20:
                    print(f"    Data: {field_data}")
                else:
                    print(f"    Sample (first 5): {field_data.flat[:5]}")
            else:
                print(f"    Empty array")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cs_file = sys.argv[1]
    else:
        cs_file = "cryosparc_P269_J199_007_particles.cs"
    
    print(f"Reading cryosparc file: {cs_file}")
    print("="*60)
    
    try:
        data = read_cs_file(cs_file)
        print_data_summary(data)
        
        # Save individual arrays if requested
        if len(sys.argv) > 2 and sys.argv[2] == "--save":
            output_dir = Path("extracted_data")
            output_dir.mkdir(exist_ok=True)
            # Save the full structured array
            np.save(output_dir / "full_data.npy", data)
            print(f"\nSaved full data to {output_dir / 'full_data.npy'}")
            
            # Save individual fields
            if data.dtype.names:
                for field_name in data.dtype.names:
                    field_data = data[field_name]
                    np.save(output_dir / f"{field_name}.npy", field_data)
                    print(f"Saved {field_name} to {output_dir / f'{field_name}.npy'}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

