#!/usr/bin/env python3
"""
POSEMAP - GUI Application
Pose-Oriented Single-particle EM Micrograph Annotation & Projection

Interactive Tkinter-based GUI for visualizing cryo-EM particles on micrographs.

Features:
- Left panel: Controls and file management
- Right panel: Large micrograph display with particle overlays
- Separate window: 3D projection preview
- ChimeraX integration for interactive structure viewing
- Background projection preloading with caching
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better integration
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import mrcfile
from pathlib import Path
import sys
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from particle_mapper import (
    load_cs_file, match_particles, load_volume, project_volume,
    get_particle_orientation_arrow, get_particle_axes,
    fractional_to_pixel_coords, load_pdb_structure, project_pdb_structure,
    calculate_custom_vector_from_pdb, project_custom_vector,
    calculate_structure_com, calculate_com_offset_correction,
    calculate_vector_from_two_points, simulate_em_projection_from_pdb
)
from scipy.ndimage import gaussian_filter


class ParticleMapperGUI:
    """Redesigned GUI for visualizing particles on micrographs."""
    
    @staticmethod
    def _get_default_button_color():
        """Get the default button background color (cross-platform)."""
        try:
            # Try to get the default button color from ttk style
            style = ttk.Style()
            default_color = style.lookup('TButton', 'background')
            if default_color:
                return default_color
        except:
            pass
        # Fallback: return empty string to use system default
        return ''
    
    def __init__(self, root, refinement_cs_path=None, passthrough_cs_path=None,
                 pdb_path=None, micrograph_dir=None):
        """Initialize the GUI."""
        self.root = root
        self.root.title("POSEMAP - Pose-Oriented Single-particle EM Micrograph Annotation & Projection")
        self.root.geometry("1400x900")
        # Store initial geometry to restore if window gets resized by external events
        self.saved_geometry = None
        
        # Auto-detect .cs files in parent directory if not provided
        self.auto_detected_cs = False
        if refinement_cs_path is None or passthrough_cs_path is None:
            # Use a static method approach to avoid needing self
            auto_detected = ParticleMapperGUI._auto_detect_cs_files_static()
            if auto_detected:
                self.auto_detected_cs = True
                if refinement_cs_path is None:
                    refinement_cs_path = auto_detected.get('refinement')
                if passthrough_cs_path is None:
                    passthrough_cs_path = auto_detected.get('passthrough')
        
        # Auto-detect PDB and micrograph directories if not provided
        self.auto_detected_pdb = False
        self.auto_detected_micrographs = False
        if pdb_path is None:
            auto_pdb = ParticleMapperGUI._auto_detect_pdb_static()
            if auto_pdb:
                pdb_path = auto_pdb
                self.auto_detected_pdb = True
        
        if micrograph_dir is None:
            auto_micrographs = ParticleMapperGUI._auto_detect_micrograph_dir_static()
            if auto_micrographs:
                micrograph_dir = auto_micrographs
                self.auto_detected_micrographs = True
        
        # Data paths
        self.refinement_cs_path = refinement_cs_path
        self.passthrough_cs_path = passthrough_cs_path
        self.pdb_path = pdb_path
        self.micrograph_dir = Path(micrograph_dir) if micrograph_dir else None
        
        # Data storage
        self.refinement_cs = None
        self.passthrough_cs = None
        self.matched_data = None
        self.pdb_data = None  # PDB structure data
        self.micrograph_paths = []
        self.micrograph_files = []  # Actual file paths
        self.current_micrograph_idx = 0
        self.current_micrograph = None
        self.current_micrograph_path = None
        self.current_particles = None
        self.current_projection = None  # For preview window
        
        # Visualization settings
        self.show_projections = True  # ALWAYS show per-particle projections by default
        self.show_orientations = True
        self.show_projection_preview = False
        self.show_outlines = False  # Toggle for outlining projection overlays
        self.projection_alpha = 1.0  # Default: fully opaque (user can adjust with slider)
        self.projection_size = 375  # Default: start at 375 pixels
        self.base_projection_size = None  # Store original size from ChimeraX generation
        self.pixel_size_angstroms = None  # Pixel size in Angstroms per pixel (for auto-scaling)
        self.arrow_length = 90  # 3x longer default (was 30)
        
        # Rotation correction angles (in degrees) - can be fine-tuned via GUI
        # Base corrections: Y=180°, Z=180° work well but may need fine-tuning
        self.rotation_correction_x = 0.0
        self.rotation_correction_y = 180.0  # 180° around Y axis
        self.rotation_correction_z = 180.0  # 180° around Z axis
        self.rotation_correction_x_var = None  # Will be set in GUI setup
        self.rotation_correction_y_var = None  # Will be set in GUI setup
        self.rotation_correction_z_var = None  # Will be set in GUI setup
        # Small offset adjustments (in pixels) to fine-tune projection position
        # User found these values work well: X=-16.9, Y=20.0
        self.projection_offset_x = -16.9  # Small X offset in pixels
        self.projection_offset_y = 20.0  # Small Y offset in pixels
        self.projection_offset_x_var = None  # Will be set in GUI setup
        self.projection_offset_y_var = None  # Will be set in GUI setup
        self.show_scale_bar = False  # Toggle for scale bar display
        self.scale_bar_length_angstroms = 50.0  # Default scale bar length in Angstroms
        self.scale_bar_linewidth = 40.0  # Default scale bar line width in pixels
        self.scale_bar_color = 'white'  # Default scale bar color
        
        # Second arrow (custom structural vector) settings
        self.show_custom_arrow = False
        self.custom_arrow_length = 180  # Doubled from 90 for better visibility
        self.custom_arrow_color = '#FFFFFF'  # Default white color
        self.custom_vector_3d = None  # 3D vector in model coordinate system (normalized)
        self.custom_vector_method = 'user_defined'  # 'user_defined', 'chain_com', 'atom_selection', 'chain_axis', 'from_markers'
        self.custom_vector_params = {}  # Parameters for vector calculation (chain_ids, etc.)
        self.marker_positions = None  # Store marker positions [point1, point2] when using from_markers method
        self.marker_x_offset = 0.0  # X offset in pixels for marker positions (for debugging coordinate system)
        self.marker_y_offset = 0.0  # Y offset in pixels for marker positions (for debugging coordinate system)
        self.marker_distance_angstroms = None  # Distance between markers in Angstroms (for default arrow length)
        self.show_arrows_at_markers = False  # Toggle to show arrows at marker positions
        
        # Cache for fast projection toggle
        self.cached_blank_image = None  # Cached image without projections
        self.cached_with_projections = None  # Cached image with projections
        self.cached_micrograph_idx = None  # Which micrograph the cache is for
        
        # Image enhancement
        self.lowpass_A = 2.0
        self.brightness = 0.0
        self.contrast = 1.0
        self.original_micrograph = None  # Store unprocessed micrograph
        
        # Chain coloring settings
        self.color_mode = 'bio_material'  # 'bio_material' or 'chain'
        self.chain_color_map = {}  # Dict mapping chain_id -> hex color
        self.default_protein_color = '#007CBE'  # Default color for protein chains
        self.default_nucleic_color = '#3B1F2B'  # Default color for nucleic acid chains
        self.available_chains = []  # List of chain IDs from loaded PDB structure
        
        # Projection cache - store pre-generated projections for each particle
        self.projection_cache = {}  # Key: (micrograph_idx, particle_idx), Value: projection array
        # Clear cache on startup
        self.projection_cache.clear()
        self.temp_projection_dir = None  # Temporary directory for projection images
        self.current_projection_files = []  # List of temporary projection PNG files for current micrograph
        self.projection_generation_limit = 1  # Only generate first 1 projection by default
        self.all_projections_generated = False  # Track if all projections have been generated
        self._generating_projections = False  # Flag to prevent multiple simultaneous generations
        
        # Zoom functionality
        self.zoom_mode = False  # Whether zoom box selection is active
        self.zoom_box_start = None  # Starting point of zoom box
        self.zoom_box_rect = None  # Rectangle patch for zoom box
        self.zoom_xlim = None  # Saved x limits before zoom
        self.zoom_ylim = None  # Saved y limits before zoom
        
        # Background preloading
        self.background_loading = False  # Flag to prevent multiple background threads
        self.background_thread = None  # Background thread for preloading
        self.idle_timer = None  # Timer for idle detection
        
        # Thread pool for parallel projection generation (up to 3 workers)
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ProjectionWorker")
        self.cache_lock = threading.Lock()  # Lock for thread-safe cache access
        self.generation_futures = {}  # Track ongoing generation tasks
        self.last_user_action = time.time()  # Track last user interaction
        self.default_projection_limit = 1  # Default limit (reset when loading new micrograph)
        self.background_preload_limit = 5  # How many to preload in background (more than default for smoother experience)
        
        # Create temporary directory for projection cache in working directory
        self.temp_projection_dir = Path.cwd() / 'particle_projections_temp'
        self.temp_projection_dir.mkdir(exist_ok=True)
        print(f"Created temporary projection cache directory: {self.temp_projection_dir}")
        
        # Setup GUI
        self.setup_gui()
        
        # Auto-load vector from vector_val.txt if present
        self._auto_load_vector_file()
        
        # Auto-load vector from vector_val.txt if present
        self._auto_load_vector_file()
        
        # Register cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Check if fields should be orange initially (AFTER GUI is set up)
        self.root.after(100, self.check_fields_filled)  # Use after() to ensure GUI is fully rendered
        
        # Load data if paths provided
        if all([refinement_cs_path, passthrough_cs_path, pdb_path, micrograph_dir]):
            self.load_all_data()
    
    def on_closing(self):
        """Clean up when window is closed."""
        # Cancel any background loading
        self._cancel_background_loading()
        
        # Shutdown thread pool executor
        if hasattr(self, 'executor'):
            print("Shutting down thread pool executor...")
            self.executor.shutdown(wait=False, cancel_futures=True)
        
        # Clean up projection cache
        self._cleanup_projection_cache()
        
        # Clean up projection files
        self._cleanup_projection_files()
        
        # Clean up temp directory
        if self.temp_projection_dir and self.temp_projection_dir.exists():
            import shutil
            try:
                shutil.rmtree(self.temp_projection_dir, ignore_errors=True)
                print(f"Cleaned up temp directory: {self.temp_projection_dir}")
            except:
                pass
        
        # Clean up any leftover ChimeraX script files in working directory
        try:
            for script_file in Path.cwd().glob('chimerax_view_*.cxc'):
                try:
                    script_file.unlink()
                except:
                    pass
        except:
            pass
        
        self.root.destroy()
    
    @staticmethod
    def _auto_detect_chimerax():
        """
        Auto-detect ChimeraX executable path.
        Tries: which chimerax, which ChimeraX, common macOS locations.
        Returns path string or None.
        """
        import shutil
        
        # Try common command names
        for cmd_name in ['chimerax', 'ChimeraX', 'chimerax-app']:
            path = shutil.which(cmd_name)
            if path:
                return path
        
        # Try common macOS app bundle locations
        common_paths = [
            '/Applications/ChimeraX.app/Contents/bin/ChimeraX',
            '/Applications/ChimeraX.app/Contents/MacOS/ChimeraX',
            '/Applications/ChimeraX-1.9.app/Contents/bin/ChimeraX',
            '/Applications/ChimeraX-1.9.app/Contents/MacOS/ChimeraX',
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
        
        # Try to find any ChimeraX.app in Applications
        apps_dir = Path('/Applications')
        if apps_dir.exists():
            for app in apps_dir.glob('ChimeraX*.app'):
                # Try bin first, then MacOS
                for subpath in ['Contents/bin/ChimeraX', 'Contents/MacOS/ChimeraX']:
                    full_path = app / subpath
                    if full_path.exists():
                        return str(full_path)
        
        return None
    
    @staticmethod
    def _auto_detect_cs_files_static():
        """
        Auto-detect .cs files in the current and parent directories.
        If exactly 2 .cs files exist and one contains 'passthrough', auto-assign them.
        Returns dict with 'refinement' and 'passthrough' keys, or None if not found.
        """
        try:
            # Get the current directory (where script is) and parent directory
            script_dir = Path(__file__).parent.absolute()
            parent_dir = script_dir.parent
            
            # Try current directory first, then parent directory
            for search_dir in [script_dir, parent_dir]:
                # Find all .cs files in directory
                cs_files = list(search_dir.glob('*.cs'))
                
                if len(cs_files) == 2:
                    # Check which one contains 'passthrough'
                    passthrough_file = None
                    refinement_file = None
                    
                    for cs_file in cs_files:
                        if 'passthrough' in cs_file.name.lower():
                            passthrough_file = cs_file
                        else:
                            refinement_file = cs_file
                    
                    # If we found a passthrough file, return both
                    if passthrough_file and refinement_file:
                        print(f"Auto-detected CS files in {search_dir}:")
                        print(f"  Refinement: {refinement_file}")
                        print(f"  Passthrough: {passthrough_file}")
                        return {
                            'refinement': str(refinement_file),
                            'passthrough': str(passthrough_file)
                        }
            
            return None
        except Exception as e:
            print(f"Error in auto-detection: {e}")
            return None
    
    @staticmethod
    def _auto_detect_pdb_static():
        """
        Auto-detect structure file (.pdb or .cif) in 'ref_volume' directory.
        Looks for .pdb or .cif files (especially _aligned.pdb or _aligned.cif) in ref_volume folder.
        Prefers .cif over .pdb if both are present.
        Returns path string to first structure file found, or None.
        """
        try:
            # Get the current directory (where script is)
            script_dir = Path(__file__).parent.absolute()
            
            # Check for ref_volume directory
            ref_volume_dir = script_dir / 'ref_volume'
            if ref_volume_dir.exists() and ref_volume_dir.is_dir():
                # Look for .cif files first (preferred), then .pdb files
                cif_files = list(ref_volume_dir.glob('*.cif'))
                pdb_files = list(ref_volume_dir.glob('*.pdb'))
                
                # Prefer .cif files
                if cif_files:
                    # Prefer _aligned.cif if available
                    aligned_cif = [f for f in cif_files if '_aligned.cif' in f.name]
                    if aligned_cif:
                        struct_path = str(aligned_cif[0])
                    else:
                        struct_path = str(cif_files[0])  # Use first .cif file found
                    print(f"Auto-detected structure file (CIF): {struct_path}")
                    return struct_path
                
                # Fall back to .pdb files
                if pdb_files:
                    # Prefer _aligned.pdb if available
                    aligned_pdb = [f for f in pdb_files if '_aligned.pdb' in f.name]
                    if aligned_pdb:
                        struct_path = str(aligned_pdb[0])
                    else:
                        struct_path = str(pdb_files[0])  # Use first .pdb file found
                    print(f"Auto-detected structure file (PDB): {struct_path}")
                    return struct_path
            
            return None
        except Exception as e:
            print(f"Error in structure file auto-detection: {e}")
            return None
    
    @staticmethod
    def _auto_detect_micrograph_dir_static():
        """
        Auto-detect micrograph directory.
        Looks for folders named 'ref_movies', 'ref_micrographs', or 'ref_mics'.
        Returns path string to first matching directory found, or None.
        """
        try:
            # Get the current directory (where script is)
            script_dir = Path(__file__).parent.absolute()
            
            # Check for common micrograph directory names
            dir_names = ['ref_movies', 'ref_micrographs', 'ref_mics']
            for dir_name in dir_names:
                micrograph_dir = script_dir / dir_name
                if micrograph_dir.exists() and micrograph_dir.is_dir():
                    # Check if it contains any .mrc files
                    mrc_files = list(micrograph_dir.glob('*.mrc'))
                    if mrc_files:
                        micrograph_path = str(micrograph_dir)
                        print(f"Auto-detected micrograph directory: {micrograph_path} ({len(mrc_files)} .mrc files)")
                        return micrograph_path
            
            return None
        except Exception as e:
            print(f"Error in micrograph directory auto-detection: {e}")
            return None
    
    def setup_gui(self):
        """Setup the GUI layout."""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls - increased width to prevent cutoff
        left_frame = ttk.Frame(main_paned, width=450)
        main_paned.add(left_frame, weight=0)
        # Set minimum width to prevent it from being resized too small
        left_frame.pack_propagate(False)
        
        # Right panel for micrograph display
        right_frame = ttk.Frame(main_paned)
        # Prevent the right frame from resizing when external events occur (like launching ChimeraX)
        right_frame.pack_propagate(False)
        main_paned.add(right_frame, weight=1)
        
        self.setup_left_panel(left_frame)
        self.setup_right_panel(right_frame)
        
        # Create projection preview window (initially hidden)
        self.setup_projection_preview()
    
    def setup_left_panel(self, parent):
        """Setup the left control panel."""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Files section
        files_frame = ttk.LabelFrame(scrollable_frame, text="Files", padding=10)
        files_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(files_frame, text="Refinement CS:").pack(anchor=tk.W)
        self.refinement_cs_entry = ttk.Entry(files_frame, width=40)
        self.refinement_cs_entry.pack(fill=tk.X, pady=2)
        if self.refinement_cs_path:
            self.refinement_cs_entry.insert(0, str(self.refinement_cs_path))
        ttk.Button(files_frame, text="Browse...", 
                  command=self.browse_refinement_cs).pack(pady=2)
        
        ttk.Label(files_frame, text="Passthrough CS:").pack(anchor=tk.W, pady=(10,0))
        self.passthrough_cs_entry = ttk.Entry(files_frame, width=40)
        self.passthrough_cs_entry.pack(fill=tk.X, pady=2)
        if self.passthrough_cs_path:
            self.passthrough_cs_entry.insert(0, str(self.passthrough_cs_path))
        ttk.Button(files_frame, text="Browse...", 
                  command=self.browse_passthrough_cs).pack(pady=2)
        
        # Show auto-detection status if files were auto-detected
        if hasattr(self, 'auto_detected_cs') and self.auto_detected_cs:
            auto_status = "✓ Auto-detected from parent directory"
            status_label = ttk.Label(files_frame, text=auto_status, 
                                    font=('TkDefaultFont', 8), foreground='green')
            status_label.pack(anchor=tk.W, pady=(2, 0))
        
        ttk.Label(files_frame, text="PDB Structure:").pack(anchor=tk.W, pady=(10,0))
        self.pdb_entry = ttk.Entry(files_frame, width=40)
        self.pdb_entry.pack(fill=tk.X, pady=2)
        if self.pdb_path:
            self.pdb_entry.insert(0, str(self.pdb_path))
        ttk.Button(files_frame, text="Browse...", 
                  command=self.browse_pdb).pack(pady=2)
        # Show auto-detection status if PDB was auto-detected
        if hasattr(self, 'auto_detected_pdb') and self.auto_detected_pdb:
            auto_status = "✓ Auto-detected from ref_volume/"
            status_label = ttk.Label(files_frame, text=auto_status, 
                                    font=('TkDefaultFont', 8), foreground='green')
            status_label.pack(anchor=tk.W, pady=(2, 0))
        
        ttk.Label(files_frame, text="Micrograph Directory:").pack(anchor=tk.W, pady=(10,0))
        self.micrograph_dir_entry = ttk.Entry(files_frame, width=40)
        self.micrograph_dir_entry.pack(fill=tk.X, pady=2)
        if self.micrograph_dir:
            self.micrograph_dir_entry.insert(0, str(self.micrograph_dir))
        ttk.Button(files_frame, text="Browse...", 
                  command=self.browse_micrograph_dir).pack(pady=2)
        
        # Pixel size input for automatic projection scaling
        ttk.Label(files_frame, text="Pixel Size (Å/pixel):").pack(anchor=tk.W, pady=(10,0))
        pixel_frame = ttk.Frame(files_frame)
        pixel_frame.pack(fill=tk.X, pady=2)
        self.pixel_size_entry = ttk.Entry(pixel_frame, width=15)
        self.pixel_size_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.pixel_size_entry.insert(0, "1.1")  # Default value
        ttk.Button(pixel_frame, text="Auto-scale", 
                  command=self.auto_scale_projection_size).pack(side=tk.LEFT)
        ttk.Label(pixel_frame, text="(for automatic projection sizing)", 
                 font=('TkDefaultFont', 8), foreground='gray').pack(side=tk.LEFT, padx=(5, 0))
        
        # Particle extraction box size for comparison
        ttk.Label(files_frame, text="Particle Box Size (px):").pack(anchor=tk.W, pady=(10,0))
        box_size_frame = ttk.Frame(files_frame)
        box_size_frame.pack(fill=tk.X, pady=2)
        self.particle_box_size_entry = ttk.Entry(box_size_frame, width=15)
        self.particle_box_size_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.particle_box_size_entry.insert(0, "256")  # Default 256 pixels
        ttk.Label(box_size_frame, text="(for particle extraction in comparison)", 
                 font=('TkDefaultFont', 8), foreground='gray').pack(side=tk.LEFT, padx=(5, 0))
        
        # Show auto-detection status if micrograph dir was auto-detected
        if hasattr(self, 'auto_detected_micrographs') and self.auto_detected_micrographs:
            auto_status = "✓ Auto-detected from ref_movies/ref_micrographs/ref_mics/"
            status_label = ttk.Label(files_frame, text=auto_status, 
                                    font=('TkDefaultFont', 8), foreground='green')
            status_label.pack(anchor=tk.W, pady=(2, 0))
        
        default_button_color = self._get_default_button_color()
        self.load_data_button = tk.Button(files_frame, text="Load All Data", 
                                          command=self.load_all_data,
                                          bg=default_button_color,
                                          relief=tk.RAISED)
        self.load_data_button.pack(pady=10)
        
        # Bind to entry changes to check if all fields are filled
        # Create a unified callback that works for all events
        def schedule_check(event=None):
            self.root.after_idle(self.check_fields_filled)
        
        # Bind to all entry widgets
        for entry in [self.refinement_cs_entry, self.passthrough_cs_entry, 
                     self.pdb_entry, self.micrograph_dir_entry]:
            entry.bind('<KeyRelease>', schedule_check)
            entry.bind('<FocusOut>', schedule_check)
            entry.bind('<Button-1>', schedule_check)  # Also check on click
        
        # Image Selection section
        selection_frame = ttk.LabelFrame(scrollable_frame, text="Image Selection", padding=10)
        selection_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(selection_frame, text="Sort by:").pack(anchor=tk.W)
        self.sort_var = tk.StringVar(value="Name")
        ttk.Combobox(selection_frame, textvariable=self.sort_var, 
                    values=["Name", "Particles", "Size"], state="readonly",
                    width=37).pack(fill=tk.X, pady=2)
        
        ttk.Label(selection_frame, text="Images:").pack(anchor=tk.W, pady=(10,0))
        
        # Listbox with scrollbar
        listbox_frame = ttk.Frame(selection_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        self.image_listbox = tk.Listbox(listbox_frame, height=8)
        image_scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", 
                                       command=self.image_listbox.yview)
        self.image_listbox.configure(yscrollcommand=image_scrollbar.set)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        image_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # Navigation buttons and index input
        nav_frame = ttk.Frame(selection_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        ttk.Button(nav_frame, text="Previous", 
                  command=self.prev_micrograph).pack(side=tk.LEFT, padx=2)
        
        # Micrograph index input
        index_frame = ttk.Frame(nav_frame)
        index_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(index_frame, text="Go to:").pack(side=tk.LEFT, padx=2)
        self.micrograph_index_entry = ttk.Entry(index_frame, width=8)
        self.micrograph_index_entry.pack(side=tk.LEFT, padx=2)
        self.micrograph_index_entry.bind('<Return>', self.jump_to_micrograph)
        ttk.Button(index_frame, text="Go", 
                  command=self.jump_to_micrograph).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next", 
                  command=self.next_micrograph).pack(side=tk.LEFT, padx=2)
        
        # Visualization Controls section
        viz_frame = ttk.LabelFrame(scrollable_frame, text="Visualization", padding=10)
        viz_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.show_projections_var = tk.BooleanVar(value=True)  # Always True - per-particle projections are the core feature
        ttk.Checkbutton(viz_frame, text="Show Projections (per-particle, based on pose)", 
                       variable=self.show_projections_var,
                       command=self.toggle_projections).pack(anchor=tk.W)
        
        self.show_orientations_var = tk.BooleanVar(value=self.show_orientations)
        ttk.Checkbutton(viz_frame, text="Show Orientations", 
                       variable=self.show_orientations_var,
                       command=self.toggle_orientations).pack(anchor=tk.W)
        
        self.show_preview_var = tk.BooleanVar(value=self.show_projection_preview)
        ttk.Checkbutton(viz_frame, text="Show Projection Preview", 
                       variable=self.show_preview_var,
                       command=self.toggle_preview).pack(anchor=tk.W)
        
        self.show_outlines_var = tk.BooleanVar(value=self.show_outlines)
        ttk.Checkbutton(viz_frame, text="Show Projection Outlines", 
                       variable=self.show_outlines_var,
                       command=self.toggle_outlines).pack(anchor=tk.W)
        
        # Projection count controls
        proj_count_frame = ttk.Frame(viz_frame)
        proj_count_frame.pack(fill=tk.X, pady=5)
        ttk.Label(proj_count_frame, text="Projections to show:").pack(side=tk.LEFT, padx=(0, 5))
        self.projection_count_var = tk.IntVar(value=self.projection_generation_limit)
        self.projection_count_label = ttk.Label(proj_count_frame, text=str(self.projection_generation_limit))
        self.projection_count_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(proj_count_frame, text="−", width=3, 
                  command=self.decrease_projection_count).pack(side=tk.LEFT, padx=2)
        ttk.Button(proj_count_frame, text="+", width=3, 
                  command=self.increase_projection_count).pack(side=tk.LEFT, padx=2)
        
        # Button to generate remaining projections
        self.generate_all_button = tk.Button(viz_frame, text="Generate All Projections", 
                                             command=self.generate_remaining_projections,
                                             state=tk.DISABLED)
        self.generate_all_button.pack(pady=5)
        
        ttk.Label(viz_frame, text="Projection Alpha:").pack(anchor=tk.W, pady=(10,0))
        self.alpha_var = tk.DoubleVar(value=self.projection_alpha)
        alpha_scale = ttk.Scale(viz_frame, from_=0.0, to=1.0, 
                               variable=self.alpha_var, orient=tk.HORIZONTAL,
                               command=lambda v: self.update_alpha(float(v)))
        alpha_scale.pack(fill=tk.X, pady=2)
        self.alpha_label = ttk.Label(viz_frame, text=f"{self.projection_alpha:.2f}")
        self.alpha_label.pack(anchor=tk.W)
        
        ttk.Label(viz_frame, text="Projection Size (px):").pack(anchor=tk.W, pady=(10,0))
        self.size_var = tk.IntVar(value=self.projection_size)
        size_scale = ttk.Scale(viz_frame, from_=50, to=400, 
                              variable=self.size_var, orient=tk.HORIZONTAL,
                              command=lambda v: self.update_size(int(float(v))))
        size_scale.pack(fill=tk.X, pady=2)
        self.size_label = ttk.Label(viz_frame, text=f"{self.projection_size} px")
        self.size_label.pack(anchor=tk.W)
        
        ttk.Label(viz_frame, text="Arrow Length (px):").pack(anchor=tk.W, pady=(10,0))
        self.arrow_var = tk.IntVar(value=self.arrow_length)
        arrow_scale = ttk.Scale(viz_frame, from_=10, to=100, 
                               variable=self.arrow_var, orient=tk.HORIZONTAL,
                               command=lambda v: self.update_arrow(int(float(v))))
        arrow_scale.pack(fill=tk.X, pady=2)
        self.arrow_label = ttk.Label(viz_frame, text=f"{self.arrow_length} px")
        self.arrow_label.pack(anchor=tk.W)
        
        # Projection fine-tuning offset controls
        ttk.Separator(viz_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(15, 10))
        
        # Collapsible fine-tuning section
        fine_tuning_frame = ttk.LabelFrame(viz_frame, text="Projection Fine-Tuning (Advanced)", padding=5)
        fine_tuning_frame.pack(fill=tk.X, pady=5)
        
        # Toggle button to show/hide fine-tuning controls
        self.fine_tuning_expanded = tk.BooleanVar(value=False)
        toggle_button = ttk.Checkbutton(fine_tuning_frame, text="Show Fine-Tuning Controls", 
                                        variable=self.fine_tuning_expanded,
                                        command=lambda: self._toggle_fine_tuning(fine_tuning_frame))
        toggle_button.pack(anchor=tk.W, pady=(0, 5))
        
        # Auto-correct COM offset checkbox (always visible)
        self.auto_correct_com_offset = False
        self.auto_correct_com_offset_var = tk.BooleanVar(value=self.auto_correct_com_offset)
        auto_correct_check = ttk.Checkbutton(fine_tuning_frame, 
                                             text="Auto-correct Center of Mass Offset",
                                             variable=self.auto_correct_com_offset_var,
                                             command=self._toggle_auto_correct_com)
        auto_correct_check.pack(anchor=tk.W, pady=(0, 5))
        
        # Container for fine-tuning controls (initially hidden)
        offset_frame = ttk.Frame(fine_tuning_frame)
        # Store reference for toggle function
        self.fine_tuning_offset_frame = offset_frame
        
        # X offset slider
        ttk.Label(offset_frame, text="X Offset (pixels):").pack(anchor=tk.W, pady=(5, 0))
        self.projection_offset_x_var = tk.DoubleVar(value=self.projection_offset_x)
        offset_x_scale = ttk.Scale(offset_frame, from_=-20.0, to=20.0, 
                                   variable=self.projection_offset_x_var, orient=tk.HORIZONTAL,
                                   command=lambda v: self.update_projection_offset('x', float(v)))
        offset_x_scale.pack(fill=tk.X, pady=2)
        self.offset_x_label = ttk.Label(offset_frame, text=f"{self.projection_offset_x:.1f} px")
        self.offset_x_label.pack(anchor=tk.W)
        
        # Y offset slider
        ttk.Label(offset_frame, text="Y Offset (pixels):").pack(anchor=tk.W, pady=(10, 0))
        self.projection_offset_y_var = tk.DoubleVar(value=self.projection_offset_y)
        offset_y_scale = ttk.Scale(offset_frame, from_=-20.0, to=20.0, 
                                   variable=self.projection_offset_y_var, orient=tk.HORIZONTAL,
                                   command=lambda v: self.update_projection_offset('y', float(v)))
        offset_y_scale.pack(fill=tk.X, pady=2)
        self.offset_y_label = ttk.Label(offset_frame, text=f"{self.projection_offset_y:.1f} px")
        self.offset_y_label.pack(anchor=tk.W)
        
        # Rotation fine-tuning controls
        ttk.Label(offset_frame, text="Rotation Corrections (degrees):", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, pady=(15, 5))
        
        # X rotation slider
        ttk.Label(offset_frame, text="Rotate X (degrees):").pack(anchor=tk.W, pady=(5, 0))
        self.rotation_correction_x_var = tk.DoubleVar(value=self.rotation_correction_x)
        rot_x_scale = ttk.Scale(offset_frame, from_=-180.0, to=180.0, 
                                variable=self.rotation_correction_x_var, orient=tk.HORIZONTAL,
                                command=lambda v: self.update_rotation_correction('x', float(v)))
        rot_x_scale.pack(fill=tk.X, pady=2)
        self.rot_x_label = ttk.Label(offset_frame, text=f"{self.rotation_correction_x:.1f}°")
        self.rot_x_label.pack(anchor=tk.W)
        
        # Y rotation slider
        ttk.Label(offset_frame, text="Rotate Y (degrees):").pack(anchor=tk.W, pady=(10, 0))
        self.rotation_correction_y_var = tk.DoubleVar(value=self.rotation_correction_y)
        rot_y_scale = ttk.Scale(offset_frame, from_=-180.0, to=180.0, 
                                variable=self.rotation_correction_y_var, orient=tk.HORIZONTAL,
                                command=lambda v: self.update_rotation_correction('y', float(v)))
        rot_y_scale.pack(fill=tk.X, pady=2)
        self.rot_y_label = ttk.Label(offset_frame, text=f"{self.rotation_correction_y:.1f}°")
        self.rot_y_label.pack(anchor=tk.W)
        
        # Z rotation slider
        ttk.Label(offset_frame, text="Rotate Z (degrees):").pack(anchor=tk.W, pady=(10, 0))
        self.rotation_correction_z_var = tk.DoubleVar(value=self.rotation_correction_z)
        rot_z_scale = ttk.Scale(offset_frame, from_=-180.0, to=180.0, 
                                variable=self.rotation_correction_z_var, orient=tk.HORIZONTAL,
                                command=lambda v: self.update_rotation_correction('z', float(v)))
        rot_z_scale.pack(fill=tk.X, pady=2)
        self.rot_z_label = ttk.Label(offset_frame, text=f"{self.rotation_correction_z:.1f}°")
        self.rot_z_label.pack(anchor=tk.W)
        
        # Apply button to regenerate projections with new rotation
        apply_rot_button = ttk.Button(offset_frame, text="Apply Rotation & Regenerate", 
                                     command=self.apply_rotation_correction_and_regenerate)
        apply_rot_button.pack(pady=(15, 5))
        
        # Initially hide the fine-tuning controls (they start collapsed)
        # They will be shown/hidden by the toggle function
        
        # Scale bar controls
        ttk.Separator(viz_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(15, 10))
        self.show_scale_bar_var = tk.BooleanVar(value=self.show_scale_bar)
        ttk.Checkbutton(viz_frame, text="Show Scale Bar", 
                       variable=self.show_scale_bar_var,
                       command=self.toggle_scale_bar).pack(anchor=tk.W, pady=(10,0))
        
        ttk.Label(viz_frame, text="Scale Bar Length (Å):").pack(anchor=tk.W, pady=(10,0))
        self.scale_bar_var = tk.DoubleVar(value=self.scale_bar_length_angstroms)
        scale_bar_scale = ttk.Scale(viz_frame, from_=10, to=1500, 
                                   variable=self.scale_bar_var, orient=tk.HORIZONTAL,
                                   command=lambda v: self.update_scale_bar(float(v)))
        scale_bar_scale.pack(fill=tk.X, pady=2)
        self.scale_bar_label = ttk.Label(viz_frame, text=f"{self.scale_bar_length_angstroms:.0f} Å")
        self.scale_bar_label.pack(anchor=tk.W)
        
        # Scale bar appearance controls
        ttk.Label(viz_frame, text="Scale Bar Width:").pack(anchor=tk.W, pady=(10,0))
        self.scale_bar_width_var = tk.DoubleVar(value=self.scale_bar_linewidth)
        # Maximum width will be set dynamically based on image height (0.25 * height)
        # For now, use a reasonable default max (will be updated when image is loaded)
        scale_bar_width_scale = ttk.Scale(viz_frame, from_=1.0, to=200.0,
                                         variable=self.scale_bar_width_var, orient=tk.HORIZONTAL,
                                         command=lambda v: self.update_scale_bar_width(float(v)))
        scale_bar_width_scale.pack(fill=tk.X, pady=2)
        self.scale_bar_width_scale = scale_bar_width_scale  # Store reference for dynamic max update
        self.scale_bar_width_label = ttk.Label(viz_frame, text=f"{self.scale_bar_linewidth:.1f} px")
        self.scale_bar_width_label.pack(anchor=tk.W)
        
        # Scale bar color
        color_frame = ttk.Frame(viz_frame)
        color_frame.pack(fill=tk.X, pady=2)
        ttk.Label(color_frame, text="Color:").pack(side=tk.LEFT, padx=(0, 5))
        self.scale_bar_color_entry = ttk.Entry(color_frame, width=10)
        self.scale_bar_color_entry.insert(0, self.scale_bar_color)
        self.scale_bar_color_entry.pack(side=tk.LEFT, padx=2)
        self.scale_bar_color_entry.bind('<Return>', self.on_scale_bar_color_change)
        ttk.Button(color_frame, text="Pick", width=8,
                  command=self.pick_scale_bar_color).pack(side=tk.LEFT, padx=2)
        
        # Separator for custom arrow section
        ttk.Separator(viz_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(15, 10))
        ttk.Label(viz_frame, text="Custom Structural Vector", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))
        
        # Custom arrow toggle
        self.show_custom_arrow_var = tk.BooleanVar(value=self.show_custom_arrow)
        ttk.Checkbutton(viz_frame, text="Show Custom Vector Arrow", 
                       variable=self.show_custom_arrow_var,
                       command=self.toggle_custom_arrow).pack(anchor=tk.W)
        
        # Custom arrow method selection
        method_frame = ttk.Frame(viz_frame)
        method_frame.pack(fill=tk.X, pady=(5, 5))
        ttk.Label(method_frame, text="Vector Method:").pack(side=tk.LEFT, padx=(0, 5))
        self.custom_vector_method_var = tk.StringVar(value='user_defined')
        method_combo = ttk.Combobox(method_frame, textvariable=self.custom_vector_method_var,
                                   values=['user_defined', 'from_markers', 'chain_com', 'atom_selection', 'chain_axis'],
                                   state='readonly', width=15)
        method_combo.pack(side=tk.LEFT, padx=2)
        method_combo.bind('<<ComboboxSelected>>', self.on_custom_vector_method_changed)
        
        # Custom arrow vector input (for user_defined method)
        self.custom_vector_frame = ttk.Frame(viz_frame)
        self.custom_vector_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.custom_vector_frame, text="Vector (x, y, z):").pack(anchor=tk.W)
        vector_input_frame = ttk.Frame(self.custom_vector_frame)
        vector_input_frame.pack(fill=tk.X, pady=2)
        self.custom_vector_x_entry = ttk.Entry(vector_input_frame, width=8)
        self.custom_vector_x_entry.insert(0, "0.0")
        self.custom_vector_x_entry.pack(side=tk.LEFT, padx=2)
        self.custom_vector_y_entry = ttk.Entry(vector_input_frame, width=8)
        self.custom_vector_y_entry.insert(0, "0.0")
        self.custom_vector_y_entry.pack(side=tk.LEFT, padx=2)
        self.custom_vector_z_entry = ttk.Entry(vector_input_frame, width=8)
        self.custom_vector_z_entry.insert(0, "1.0")
        self.custom_vector_z_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(vector_input_frame, text="Update", width=8,
                  command=self.update_custom_vector).pack(side=tk.LEFT, padx=5)
        
        # Marker positions input (for from_markers method)
        self.custom_markers_frame = ttk.Frame(viz_frame)
        # Don't pack initially - will be shown when from_markers method is selected
        ttk.Label(self.custom_markers_frame, text="ChimeraX Marker Positions:").pack(anchor=tk.W)
        ttk.Label(self.custom_markers_frame, text="Marker 1 (x, y, z):").pack(anchor=tk.W, pady=(5,0))
        marker1_frame = ttk.Frame(self.custom_markers_frame)
        marker1_frame.pack(fill=tk.X, pady=2)
        self.marker1_x_entry = ttk.Entry(marker1_frame, width=10)
        self.marker1_x_entry.pack(side=tk.LEFT, padx=2)
        self.marker1_y_entry = ttk.Entry(marker1_frame, width=10)
        self.marker1_y_entry.pack(side=tk.LEFT, padx=2)
        self.marker1_z_entry = ttk.Entry(marker1_frame, width=10)
        self.marker1_z_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(self.custom_markers_frame, text="Marker 2 (x, y, z):").pack(anchor=tk.W, pady=(5,0))
        marker2_frame = ttk.Frame(self.custom_markers_frame)
        marker2_frame.pack(fill=tk.X, pady=2)
        self.marker2_x_entry = ttk.Entry(marker2_frame, width=10)
        self.marker2_x_entry.pack(side=tk.LEFT, padx=2)
        self.marker2_y_entry = ttk.Entry(marker2_frame, width=10)
        self.marker2_y_entry.pack(side=tk.LEFT, padx=2)
        self.marker2_z_entry = ttk.Entry(marker2_frame, width=10)
        self.marker2_z_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(self.custom_markers_frame, text="Calculate Vector (Marker 1 → Marker 2)", width=30,
                  command=self.update_custom_vector_from_markers).pack(pady=5)
        
        # X and Y offset inputs for debugging marker position
        offset_frame = ttk.Frame(self.custom_markers_frame)
        offset_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(offset_frame, text="Marker X Offset (px):").pack(side=tk.LEFT, padx=(0, 5))
        self.marker_x_offset_entry = ttk.Entry(offset_frame, width=10)
        self.marker_x_offset_entry.insert(0, "0.0")
        self.marker_x_offset_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(offset_frame, text="Y Offset (px):").pack(side=tk.LEFT, padx=(5, 5))
        self.marker_y_offset_entry = ttk.Entry(offset_frame, width=10)
        self.marker_y_offset_entry.insert(0, "0.0")
        self.marker_y_offset_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(offset_frame, text="Apply", width=6, command=self.update_marker_offset).pack(side=tk.LEFT, padx=2)
        ttk.Label(offset_frame, text="(for debugging - adjust to find correct offset)").pack(side=tk.LEFT, padx=5)
        self.marker_x_offset_entry.bind('<Return>', self.update_marker_offset)
        self.marker_x_offset_entry.bind('<FocusOut>', self.update_marker_offset)
        self.marker_y_offset_entry.bind('<Return>', self.update_marker_offset)
        self.marker_y_offset_entry.bind('<FocusOut>', self.update_marker_offset)
        
        # Chain selection (for chain_com and chain_axis methods)
        self.custom_chain_frame = ttk.Frame(viz_frame)
        # Don't pack initially - will be shown when chain methods are selected
        ttk.Label(self.custom_chain_frame, text="Chain IDs (comma-separated):").pack(anchor=tk.W)
        self.custom_chain_entry = ttk.Entry(self.custom_chain_frame, width=20)
        self.custom_chain_entry.pack(fill=tk.X, pady=2)
        ttk.Button(self.custom_chain_frame, text="Update from Chains", width=15,
                  command=self.update_custom_vector_from_chains).pack(pady=2)
        
        # Custom arrow length (doubled range for better visibility)
        ttk.Label(viz_frame, text="Custom Arrow Length (px):").pack(anchor=tk.W, pady=(10,0))
        self.custom_arrow_var = tk.IntVar(value=self.custom_arrow_length)
        custom_arrow_scale = ttk.Scale(viz_frame, from_=20, to=200, 
                                      variable=self.custom_arrow_var, orient=tk.HORIZONTAL,
                                      command=lambda v: self.update_custom_arrow_length(int(float(v))))
        custom_arrow_scale.pack(fill=tk.X, pady=2)
        self.custom_arrow_label = ttk.Label(viz_frame, text=f"{self.custom_arrow_length} px")
        self.custom_arrow_label.pack(anchor=tk.W)
        
        # Custom arrow color
        color_frame = ttk.Frame(viz_frame)
        color_frame.pack(fill=tk.X, pady=5)
        ttk.Label(color_frame, text="Custom Arrow Color:").pack(side=tk.LEFT, padx=(0, 5))
        self.custom_arrow_color_entry = ttk.Entry(color_frame, width=10)
        self.custom_arrow_color_entry.insert(0, '#FFFFFF')  # Default white
        self.custom_arrow_color_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(color_frame, text="Pick", width=8,
                  command=self.pick_custom_arrow_color).pack(side=tk.LEFT, padx=2)
        
        # Option to render arrow in PyMOL (avoids coordinate transformation issues)
        self.render_arrow_in_pymol_var = tk.BooleanVar(value=False)
        def toggle_pymol_arrow():
            # Clear projection cache when toggling this option since it affects rendering
            with self.cache_lock:
                self.projection_cache.clear()
            self.update_display()
        ttk.Checkbutton(viz_frame, text="Render Arrow in PyMOL (avoids coordinate issues)", 
                       variable=self.render_arrow_in_pymol_var,
                       command=toggle_pymol_arrow).pack(anchor=tk.W, pady=(5,0))
        
        # Toggle for showing arrows at marker positions
        self.show_arrows_at_markers_var = tk.BooleanVar(value=self.show_arrows_at_markers)
        ttk.Checkbutton(viz_frame, text="Show Arrows at Marker Positions (longer, not from particle center)", 
                       variable=self.show_arrows_at_markers_var,
                       command=self.toggle_arrows_at_markers).pack(anchor=tk.W, pady=(5,0))
        
        # Image Enhancement section
        enhance_frame = ttk.LabelFrame(scrollable_frame, text="Image Enhancement", padding=10)
        enhance_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(enhance_frame, text="Low-pass (Å):").pack(anchor=tk.W)
        lowpass_input_frame = ttk.Frame(enhance_frame)
        lowpass_input_frame.pack(fill=tk.X, pady=2)
        self.lowpass_var = tk.DoubleVar(value=self.lowpass_A)
        lowpass_scale = ttk.Scale(lowpass_input_frame, from_=0.0, to=50.0, 
                                 variable=self.lowpass_var, orient=tk.HORIZONTAL,
                                 command=lambda v: self.update_lowpass(float(v)))
        lowpass_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.lowpass_entry = ttk.Entry(lowpass_input_frame, width=8)
        self.lowpass_entry.insert(0, f"{self.lowpass_A:.1f}")
        self.lowpass_entry.pack(side=tk.LEFT, padx=2)
        self.lowpass_entry.bind('<Return>', self.on_lowpass_entry_change)
        self.lowpass_label = ttk.Label(enhance_frame, text=f"{self.lowpass_A:.1f} Å")
        self.lowpass_label.pack(anchor=tk.W)
        
        ttk.Label(enhance_frame, text="Brightness:").pack(anchor=tk.W, pady=(10,0))
        self.brightness_var = tk.DoubleVar(value=self.brightness)
        brightness_scale = ttk.Scale(enhance_frame, from_=-1.0, to=1.0, 
                                    variable=self.brightness_var, orient=tk.HORIZONTAL,
                                    command=lambda v: self.update_brightness(float(v)))
        brightness_scale.pack(fill=tk.X, pady=2)
        self.brightness_label = ttk.Label(enhance_frame, text=f"{self.brightness:.2f}")
        self.brightness_label.pack(anchor=tk.W)
        
        ttk.Label(enhance_frame, text="Contrast:").pack(anchor=tk.W, pady=(10,0))
        self.contrast_var = tk.DoubleVar(value=self.contrast)
        contrast_scale = ttk.Scale(enhance_frame, from_=0.1, to=3.0, 
                                  variable=self.contrast_var, orient=tk.HORIZONTAL,
                                  command=lambda v: self.update_contrast(float(v)))
        contrast_scale.pack(fill=tk.X, pady=2)
        self.contrast_label = ttk.Label(enhance_frame, text=f"{self.contrast:.2f}")
        self.contrast_label.pack(anchor=tk.W)
        
        ttk.Button(enhance_frame, text="Reset Enhancements", 
                  command=self.reset_enhancements).pack(pady=10)
        
        # Chain Coloring section
        chain_frame = ttk.LabelFrame(scrollable_frame, text="Structure Coloring", padding=10)
        chain_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Color mode selector
        mode_frame = ttk.Frame(chain_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(mode_frame, text="Color Mode:").pack(side=tk.LEFT, padx=(0, 10))
        self.color_mode_var = tk.StringVar(value=self.color_mode)
        ttk.Radiobutton(mode_frame, text="Bio Material", variable=self.color_mode_var, 
                       value='bio_material', command=self.on_color_mode_changed).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Chain", variable=self.color_mode_var, 
                       value='chain', command=self.on_color_mode_changed).pack(side=tk.LEFT, padx=5)
        
        # Bio material color inputs (shown by default)
        self.bio_material_frame = ttk.Frame(chain_frame)
        self.bio_material_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.bio_material_frame, text="Protein Color:").pack(anchor=tk.W)
        protein_color_frame = ttk.Frame(self.bio_material_frame)
        protein_color_frame.pack(fill=tk.X, pady=2)
        self.protein_color_entry = ttk.Entry(protein_color_frame, width=15)
        self.protein_color_entry.insert(0, self.default_protein_color)
        self.protein_color_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(protein_color_frame, text="Pick", width=8,
                  command=self.pick_protein_color).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(self.bio_material_frame, text="Nucleic Acid Color:").pack(anchor=tk.W, pady=(10,0))
        nucleic_color_frame = ttk.Frame(self.bio_material_frame)
        nucleic_color_frame.pack(fill=tk.X, pady=2)
        self.nucleic_color_entry = ttk.Entry(nucleic_color_frame, width=15)
        self.nucleic_color_entry.insert(0, self.default_nucleic_color)
        self.nucleic_color_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(nucleic_color_frame, text="Pick", width=8,
                  command=self.pick_nucleic_color).pack(side=tk.LEFT, padx=2)
        
        # Chain color inputs (hidden by default, shown when chain mode selected)
        self.chain_colors_frame = ttk.Frame(chain_frame)
        # Don't pack initially - will be shown when chain mode is selected
        self.chain_color_entries = {}  # Dict mapping chain_id -> Entry widget
        self.chain_color_pick_buttons = {}  # Dict mapping chain_id -> Button widget
        
        # Update button at bottom
        ttk.Button(chain_frame, text="Update Colors", 
                  command=self.apply_chain_colors).pack(pady=(10, 0))
        
        # Zoom controls
        zoom_frame = ttk.LabelFrame(scrollable_frame, text="Zoom", padding=10)
        zoom_frame.pack(fill=tk.X, padx=5, pady=5)
        self.zoom_button = tk.Button(zoom_frame, text="Zoom to Box", 
                                     command=self.toggle_zoom_mode)
        self.zoom_button.pack(side=tk.LEFT, padx=5)
        self.reset_zoom_button = tk.Button(zoom_frame, text="Reset Zoom", 
                                           command=self.reset_zoom, state=tk.DISABLED)
        self.reset_zoom_button.pack(side=tk.LEFT, padx=5)
        
        # ChimeraX section
        chimerax_frame = ttk.LabelFrame(scrollable_frame, text="ChimeraX Integration", padding=10)
        chimerax_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(chimerax_frame, text="ChimeraX Path:").pack(anchor=tk.W)
        self.chimerax_entry = ttk.Entry(chimerax_frame, width=40)
        # Auto-detect ChimeraX path
        auto_chimerax = self._auto_detect_chimerax()
        if auto_chimerax:
            self.chimerax_entry.insert(0, auto_chimerax)
        self.chimerax_entry.pack(fill=tk.X, pady=2)
        ttk.Button(chimerax_frame, text="Browse...", 
                  command=self.browse_chimerax).pack(pady=2)
        
        ttk.Button(chimerax_frame, text="Open PDB in ChimeraX", 
                  command=self.open_chimerax).pack(pady=5)
        
        ttk.Label(chimerax_frame, text="Note:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, pady=(10,0))
        instructions = ("Each particle gets its own unique\n"
                       "PDB-generated projection based\n"
                       "on its 3D pose. Projections are\n"
                       "automatically generated when you\n"
                       "load a micrograph.")
        ttk.Label(chimerax_frame, text=instructions, justify=tk.LEFT, 
                 font=('TkDefaultFont', 8)).pack(anchor=tk.W, pady=2)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(scrollable_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=5, pady=5)
    
    def setup_right_panel(self, parent):
        """Setup the right panel for micrograph display."""
        # Create matplotlib figure with fixed size to prevent resizing
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Micrograph")
        # Set aspect ratio to equal to prevent distortion
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('off')
        
        # Embed in tkinter - create a container frame to lock size
        canvas_container = ttk.Frame(parent)
        canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas_container.pack_propagate(False)  # Prevent container from resizing
        
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Connect mouse events for zoom and particle interaction
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def setup_projection_preview(self):
        """Setup the projection preview window."""
        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("3D Projection Preview")
        self.preview_window.geometry("400x400")
        self.preview_window.withdraw()  # Hide initially
        
        # Handle window close event - uncheck the checkbox when window is closed
        self.preview_window.protocol("WM_DELETE_WINDOW", self._on_preview_window_close)
        
        preview_frame = ttk.Frame(self.preview_window)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.preview_fig = Figure(figsize=(4, 4), dpi=100)
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_ax.set_title("Current Projection")
        self.preview_ax.axis('off')
        
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, preview_frame)
        self.preview_canvas.draw()
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _on_preview_window_close(self):
        """Handle preview window being closed by user."""
        # Uncheck the checkbox to sync state
        self.show_projection_preview = False
        if hasattr(self, 'show_preview_var'):
            self.show_preview_var.set(False)
        # Destroy the window (it will be recreated if checkbox is checked again)
        if self.preview_window.winfo_exists():
            self.preview_window.destroy()
        self.preview_window = None
    
    # File browsing methods
    def browse_refinement_cs(self):
        filename = filedialog.askopenfilename(
            title="Select Refinement CS File",
            filetypes=[("CS files", "*.cs"), ("All files", "*.*")]
        )
        if filename:
            self.refinement_cs_entry.delete(0, tk.END)
            self.refinement_cs_entry.insert(0, filename)
            self.check_fields_filled()
    
    def browse_passthrough_cs(self):
        filename = filedialog.askopenfilename(
            title="Select Passthrough CS File",
            filetypes=[("CS files", "*.cs"), ("All files", "*.*")]
        )
        if filename:
            self.passthrough_cs_entry.delete(0, tk.END)
            self.passthrough_cs_entry.insert(0, filename)
            self.check_fields_filled()
    
    def browse_pdb(self):
        filename = filedialog.askopenfilename(
            title="Select Structure File (PDB or CIF)",
            filetypes=[("Structure files", "*.pdb *.cif"), ("PDB files", "*.pdb"), ("CIF files", "*.cif"), ("All files", "*.*")]
        )
        if filename:
            self.pdb_entry.delete(0, tk.END)
            self.pdb_entry.insert(0, filename)
            self.check_fields_filled()
    
    def browse_micrograph_dir(self):
        dirname = filedialog.askdirectory(title="Select Micrograph Directory")
        if dirname:
            self.micrograph_dir_entry.delete(0, tk.END)
            self.micrograph_dir_entry.insert(0, dirname)
            self.check_fields_filled()
    
    def browse_chimerax(self):
        filename = filedialog.askopenfilename(
            title="Select ChimeraX Executable",
            filetypes=[("Executable", "*"), ("All files", "*.*")]
        )
        if filename:
            self.chimerax_entry.delete(0, tk.END)
            self.chimerax_entry.insert(0, filename)
            self.chimerax_path = filename
    
    def check_fields_filled(self, event=None):
        """Check if all required fields are filled and update button color."""
        # Make sure button exists
        if not hasattr(self, 'load_data_button'):
            return
            
        refinement_path = self.refinement_cs_entry.get().strip()
        passthrough_path = self.passthrough_cs_entry.get().strip()
        pdb_path = self.pdb_entry.get().strip()
        micrograph_dir = self.micrograph_dir_entry.get().strip()
        
        all_filled = all([refinement_path, passthrough_path, pdb_path, micrograph_dir])
        
        if all_filled:
            # Soft burnt orange color
            self.load_data_button.config(bg='#D2691E', activebackground='#CD853F')
        else:
            default_button_color = self._get_default_button_color()
            self.load_data_button.config(bg=default_button_color, activebackground=default_button_color)
    
    def load_all_data(self):
        """Load all data files."""
        try:
            self.status_var.set("Loading data...")
            self.root.update()
            
            # Get paths
            refinement_path = self.refinement_cs_entry.get()
            passthrough_path = self.passthrough_cs_entry.get()
            pdb_path = self.pdb_entry.get()
            micrograph_dir = self.micrograph_dir_entry.get()
            
            if not all([refinement_path, passthrough_path, pdb_path, micrograph_dir]):
                messagebox.showerror("Error", "Please specify all file paths")
                return
            
            # Load CS files
            self.status_var.set("Loading CS files...")
            self.refinement_cs = load_cs_file(refinement_path)
            self.passthrough_cs = load_cs_file(passthrough_path)
            self.matched_data = match_particles(self.refinement_cs, self.passthrough_cs)
            
            # Load PDB structure
            self.status_var.set("Loading structure file...")
            self.pdb_data = load_pdb_structure(pdb_path)
            self.pdb_path = pdb_path
            print(f"Loaded structure file with {len(self.pdb_data['coords'])} atoms")
            
            # Extract available chains
            if 'chain_ids' in self.pdb_data:
                # Get unique chains, ensuring they're strings and stripped of whitespace
                unique_chains = set()
                for chain_id in self.pdb_data['chain_ids']:
                    # Convert to string and strip whitespace
                    chain_str = str(chain_id).strip()
                    if chain_str:  # Only add non-empty chains
                        unique_chains.add(chain_str)
                self.available_chains = sorted(list(unique_chains))
                print(f"Found {len(self.available_chains)} chains: {self.available_chains}")
                # Update chain color UI if it exists
                if hasattr(self, 'chain_colors_frame'):
                    self._update_chain_color_ui()
            
            # Try to auto-scale projection size if pixel size is available
            if self.matched_data is not None and 'pixel_size' in self.matched_data:
                try:
                    pixel_size = float(np.median(self.matched_data['pixel_size']))
                    self.pixel_size_entry.delete(0, tk.END)
                    self.pixel_size_entry.insert(0, f"{pixel_size:.3f}")
                    self.pixel_size_angstroms = pixel_size
                    # Auto-calculate projection size
                    model_size = self._calculate_model_dimensions()
                    if model_size is not None:
                        projection_size = int((model_size / pixel_size) * 1.2)
                        projection_size = max(100, min(800, projection_size))
                        if abs(projection_size - self.projection_size) > 20:  # Only update if significantly different
                            self.projection_size = projection_size
                            self.size_var.set(self.projection_size)
                            self.size_label.config(text=f"{self.projection_size} px")
                            print(f"Auto-scaled projection size to {projection_size} px based on model dimensions")
                except Exception as e:
                    print(f"Could not auto-scale projection size: {e}")
            
            # Get micrograph paths - START WITH ACTUAL FILES IN DIRECTORY
            self.micrograph_dir = Path(micrograph_dir)
            self.status_var.set("Finding available micrographs...")
            self.root.update()
            
            # Get list of actual .mrc files in the directory
            actual_mrc_files = sorted(list(self.micrograph_dir.glob('*.mrc')))
            actual_filenames = {f.name for f in actual_mrc_files}
            
            print(f"Found {len(actual_mrc_files)} micrograph files in directory")
            
            # Normalize path helper
            def normalize_path(p):
                if isinstance(p, bytes):
                    return p.decode('utf-8')
                return str(p)
            
            def get_filename_from_path(path_str):
                """Extract just the filename from a path."""
                return Path(normalize_path(path_str)).name
            
            # Now filter particles to ONLY those that match actual files
            # Match by filename (not full path) since CS file paths may differ
            particle_mask = np.array([
                get_filename_from_path(p) in actual_filenames 
                for p in self.matched_data['micrograph_paths']
            ])
            
            num_matching_particles = np.sum(particle_mask)
            print(f"Found {num_matching_particles} particles matching {len(actual_filenames)} micrograph files")
            
            # Filter matched_data to ONLY include particles from existing micrographs
            if num_matching_particles > 0:
                # Filter all matched data arrays
                self.matched_data = {
                    'refinement_indices': [self.matched_data['refinement_indices'][i] 
                                          for i in range(len(particle_mask)) if particle_mask[i]],
                    'passthrough_indices': [self.matched_data['passthrough_indices'][i] 
                                           for i in range(len(particle_mask)) if particle_mask[i]],
                    'uids': self.matched_data['uids'][particle_mask],
                    'poses': self.matched_data['poses'][particle_mask],
                    'shifts': self.matched_data['shifts'][particle_mask],
                    'pixel_size': self.matched_data['pixel_size'][particle_mask],
                    'micrograph_paths': self.matched_data['micrograph_paths'][particle_mask],
                    'center_x_frac': self.matched_data['center_x_frac'][particle_mask],
                    'center_y_frac': self.matched_data['center_y_frac'][particle_mask],
                    'micrograph_shapes': self.matched_data['micrograph_shapes'][particle_mask],
                    'micrograph_psize': self.matched_data['micrograph_psize'][particle_mask],
                }
            
            # Create micrograph_paths list from actual files, using the filename as the key
            # Store both the actual file path and the matching CS path for each micrograph
            
            # Pre-compute a dictionary mapping filenames to CS paths for O(1) lookup
            # This avoids O(n*m) nested loops
            self.status_var.set("Building micrograph index...")
            self.root.update()
            
            filename_to_cs_path = {}
            for cs_path in self.matched_data['micrograph_paths']:
                filename = get_filename_from_path(cs_path)
                if filename not in filename_to_cs_path:
                    filename_to_cs_path[filename] = normalize_path(cs_path)
            
            self.micrograph_files = []  # Actual file paths
            self.micrograph_paths = []  # CS file paths (for particle matching)
            
            for mrc_file in actual_mrc_files:
                self.micrograph_files.append(mrc_file)
                # Look up matching CS path from pre-computed dictionary
                matching_cs_path = filename_to_cs_path.get(mrc_file.name)
                if matching_cs_path:
                    self.micrograph_paths.append(matching_cs_path)
                else:
                    # If no match found, use filename as fallback
                    self.micrograph_paths.append(mrc_file.name)
            
            self.micrograph_paths = np.array(self.micrograph_paths)
            
            # Update image list
            self.update_image_list()
            
            # Reset button color after loading
            default_button_color = self._get_default_button_color()
            self.load_data_button.config(bg=default_button_color)
            
            # Don't automatically load first micrograph - let user select
            self.status_var.set(f"Loaded {len(self.matched_data['uids'])} particles, {len(self.micrograph_paths)} micrographs. Select an image to view.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Error loading data")
    
    def update_image_list(self):
        """Update the image listbox."""
        self.image_listbox.delete(0, tk.END)
        
        # Helper functions
        def normalize_path(p):
            if isinstance(p, bytes):
                return p.decode('utf-8')
            return str(p)
        
        def get_filename_from_path(path_str):
            return Path(normalize_path(path_str)).name
        
        num_files = len(self.micrograph_files)
        # Skip file size calculation if there are too many files (performance optimization)
        # File stat() calls can be slow on network filesystems
        show_file_sizes = num_files < 500
        
        # Update status
        if num_files > 100:
            self.status_var.set(f"Updating image list ({num_files} micrographs)...")
            self.root.update()
        
        # Pre-compute particle counts per micrograph using a dictionary
        # This avoids O(n*m) comparisons (3487 files * 137k particles = ~477M comparisons!)
        # Instead, we do a single pass through particles: O(m)
        filename_to_count = {}
        for cs_path in self.matched_data['micrograph_paths']:
            filename = get_filename_from_path(cs_path)
            filename_to_count[filename] = filename_to_count.get(filename, 0) + 1
        
        for idx, mg_file in enumerate(self.micrograph_files):
            mg_name = mg_file.name
            
            # Get particle count from pre-computed dictionary
            num_particles = filename_to_count.get(mg_name, 0)
            
            # Get file size (skip for large lists to avoid blocking)
            if show_file_sizes:
                if mg_file.exists():
                    try:
                        size_mb = mg_file.stat().st_size / (1024 * 1024)
                        size_str = f"{size_mb:.1f} MB"
                    except:
                        size_str = "N/A"
                else:
                    size_str = "N/A"
                display_text = f"{idx}: {mg_name[:50]}... Picks: {num_particles}, Size: {size_str}"
            else:
                display_text = f"{idx}: {mg_name[:50]}... Picks: {num_particles}"
            
            self.image_listbox.insert(tk.END, display_text)
            
            # Update GUI periodically for large lists to keep it responsive
            if num_files > 100 and idx % 100 == 0:
                self.root.update()
        
        # Clear status message
        if num_files > 100:
            self.status_var.set(f"Ready - {num_files} micrographs loaded")
    
    def find_micrograph_file(self, micrograph_path_str):
        """Find micrograph file by matching filename."""
        # Normalize path
        if isinstance(micrograph_path_str, bytes):
            micrograph_path_str = micrograph_path_str.decode('utf-8')
        
        micrograph_filename = Path(micrograph_path_str).name
        
        # Try exact match
        potential_file = self.micrograph_dir / micrograph_filename
        if potential_file.exists():
            return potential_file
        
        return None
    
    def load_micrograph(self, idx):
        """Load a micrograph by index."""
        if idx < 0 or idx >= len(self.micrograph_files):
            return
        
        # Clean up old projection cache for previous micrograph
        self._cleanup_projection_cache()
        
        self.current_micrograph_idx = idx
        mg_file = self.micrograph_files[idx]  # Use actual file path
        
        if not mg_file.exists():
            messagebox.showerror("Error", f"Micrograph file not found: {mg_file}")
            return
        
        try:
            with mrcfile.open(mg_file) as mrc:
                self.original_micrograph = mrc.data.astype(np.float32)
                self.current_micrograph = self.original_micrograph.copy()
            
            self.current_micrograph_path = str(mg_file)
            
            # Get particles for THIS SPECIFIC micrograph only
            # Match by filename since CS paths may differ from actual file paths
            def normalize_path(p):
                if isinstance(p, bytes):
                    return p.decode('utf-8')
                return str(p)
            
            def get_filename_from_path(path_str):
                return Path(normalize_path(path_str)).name
            
            target_filename = mg_file.name
            current_mg_cs_path = normalize_path(self.micrograph_paths[idx])
            
            # Match particles by filename (more reliable than full path)
            matched_paths_normalized = np.array([get_filename_from_path(p) for p in self.matched_data['micrograph_paths']])
            mask = matched_paths_normalized == target_filename
            
            # Also try matching by CS path for extra safety
            if not np.any(mask):
                matched_paths_str = np.array([normalize_path(p) for p in self.matched_data['micrograph_paths']])
                mask = matched_paths_str == current_mg_cs_path
            
            # Verify we have particles for this micrograph
            if not np.any(mask):
                print(f"Warning: No particles found for micrograph {idx}: {mg_file.name}")
                self.current_particles = {
                    'poses': np.array([]),
                    'shifts': np.array([]),
                    'center_x_frac': np.array([]),
                    'center_y_frac': np.array([]),
                    'micrograph_shape': None,
                }
            else:
                self.current_particles = {
                    'poses': self.matched_data['poses'][mask],
                    'shifts': self.matched_data['shifts'][mask],
                    'center_x_frac': self.matched_data['center_x_frac'][mask],
                    'center_y_frac': self.matched_data['center_y_frac'][mask],
                    'micrograph_shape': self.matched_data['micrograph_shapes'][mask][0] if np.any(mask) else None,
                }
            
            # Don't auto-estimate - use default size of 375
            # Update slider and label to match current size
            if hasattr(self, 'size_var'):
                self.size_var.set(self.projection_size)
            if hasattr(self, 'size_label'):
                self.size_label.config(text=f"{self.projection_size} px")
            
            # Clear display cache when loading new micrograph
            self.cached_blank_image = None
            self.cached_with_projections = None
            self.cached_micrograph_idx = None
            
            # Cancel any background loading for previous micrograph
            self._cancel_background_loading()
            
            # Generate projections for particles in this micrograph
            # Generate all in background (fast pseudo-surface rendering makes this feasible)
            if self.pdb_data is not None and len(self.current_particles['poses']) > 0:
                # Reset flag when loading new micrograph
                self.all_projections_generated = False
                # Use micrograph filename (without extension) as base for projection filenames
                base_name = mg_file.stem  # Filename without extension
                num_particles = len(self.current_particles['poses'])
                
                # Set projection limit to show all particles (they'll be generated in background)
                self.projection_generation_limit = num_particles
                if hasattr(self, 'projection_count_var'):
                    self.projection_count_var.set(self.projection_generation_limit)
                if hasattr(self, 'projection_count_label'):
                    self.projection_count_label.config(text=str(self.projection_generation_limit))
                
                # Generate all projections in background (non-blocking)
                self.status_var.set(f"Generating all {num_particles} projections in background...")
                self.root.update()
                # Start background generation - this returns immediately, doesn't block
                self._start_background_generation(idx, base_name, num_particles)
                
                # Update button state
                if hasattr(self, 'generate_all_button'):
                    self.generate_all_button.config(state=tk.DISABLED)  # All will be generated
            else:
                # No PDB data or particles - reset to default limit
                self.projection_generation_limit = self.default_projection_limit
                if hasattr(self, 'projection_count_var'):
                    self.projection_count_var.set(self.projection_generation_limit)
                if hasattr(self, 'projection_count_label'):
                    self.projection_count_label.config(text=str(self.projection_generation_limit))
            
            # Reset zoom state when loading new micrograph
            self.zoom_mode = False
            self.zoom_box_start = None
            if self.zoom_box_rect is not None:
                self.zoom_box_rect.remove()
                self.zoom_box_rect = None
            self.zoom_xlim = None
            self.zoom_ylim = None
            if hasattr(self, 'zoom_button'):
                self.zoom_button.config(relief=tk.RAISED)
            if hasattr(self, 'reset_zoom_button'):
                self.reset_zoom_button.config(state=tk.DISABLED)
            
            # Update selection in listbox
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(idx)
            self.image_listbox.see(idx)
            
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load micrograph: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _estimate_particle_size(self):
        """Estimate particle size from the micrograph by analyzing regions around particle centers.
        
        Returns estimated size in pixels, or None if estimation fails.
        """
        if self.original_micrograph is None or self.current_particles is None:
            return None
        
        if len(self.current_particles['poses']) == 0:
            return None
        
        try:
            from scipy.ndimage import gaussian_filter
            from scipy.signal import find_peaks
            
            mg_shape = self.current_particles.get('micrograph_shape')
            if mg_shape is None:
                mg_shape = self.original_micrograph.shape
            
            # Sample a few particles to estimate size
            num_samples = min(10, len(self.current_particles['poses']))
            sample_indices = np.linspace(0, len(self.current_particles['poses'])-1, num_samples, dtype=int)
            
            sizes = []
            for idx in sample_indices:
                x_frac = self.current_particles['center_x_frac'][idx]
                y_frac = self.current_particles['center_y_frac'][idx]
                x_pixel = int(x_frac * mg_shape[1])
                y_pixel = int(y_frac * mg_shape[0])
                
                # Extract a region around the particle (assume particles are roughly 50-300 pixels)
                region_size = 300
                y_start = max(0, y_pixel - region_size // 2)
                y_end = min(mg_shape[0], y_pixel + region_size // 2)
                x_start = max(0, x_pixel - region_size // 2)
                x_end = min(mg_shape[1], x_pixel + region_size // 2)
                
                if y_end - y_start < 50 or x_end - x_start < 50:
                    continue
                
                region = self.original_micrograph[y_start:y_end, x_start:x_end]
                
                # Center coordinates within region
                center_y = y_pixel - y_start
                center_x = x_pixel - x_start
                
                # Create radial profile from center
                y_coords, x_coords = np.ogrid[:region.shape[0], :region.shape[1]]
                distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                
                # Bin distances and compute mean intensity at each distance
                max_dist = min(region.shape[0], region.shape[1]) // 2
                bins = np.linspace(0, max_dist, 50)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Compute mean intensity at each distance
                intensities = []
                for i in range(len(bins) - 1):
                    mask = (distances >= bins[i]) & (distances < bins[i+1])
                    if mask.sum() > 0:
                        intensities.append(region[mask].mean())
                    else:
                        intensities.append(np.nan)
                
                intensities = np.array(intensities)
                valid = ~np.isnan(intensities)
                if valid.sum() < 10:
                    continue
                
                # Find the radius where intensity drops significantly (particle edge)
                # Particles are typically darker than background
                # Look for the point where intensity increases (background)
                if intensities[valid].min() < intensities[valid].max():
                    # Normalize intensities
                    int_norm = (intensities[valid] - intensities[valid].min()) / (intensities[valid].max() - intensities[valid].min() + 1e-6)
                    
                    # Find where intensity rises above 0.5 (transition from particle to background)
                    # Start from center and go outward
                    for i in range(len(int_norm)):
                        if int_norm[i] > 0.5:
                            radius = bin_centers[valid][i]
                            # Particle diameter is roughly 2 * radius
                            particle_size = int(2 * radius * 1.2)  # Add 20% margin
                            if 30 <= particle_size <= 400:  # Reasonable range
                                sizes.append(particle_size)
                            break
            
            if len(sizes) > 0:
                # Use median to be robust to outliers
                estimated = int(np.median(sizes))
                # Clamp to reasonable range
                estimated = max(50, min(300, estimated))
                return estimated
            
            # Fallback: use a default based on typical cryo-EM particle sizes
            # Typical 80S ribosome is ~250-300 Å, at ~1.1 Å/pixel = ~230-270 pixels
            # Use a conservative estimate
            return 200
            
        except Exception as e:
            print(f"Warning: Could not estimate particle size: {e}")
            # Fallback to default
            return 200
    
    def _calculate_model_dimensions(self):
        """Calculate the dimensions of the atomic model in Angstroms."""
        if self.pdb_data is None:
            return None
        
        coords = self.pdb_data['coords']
        if len(coords) == 0:
            return None
        
        # Calculate bounding box dimensions
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        dimensions = max_coords - min_coords
        
        # Return the maximum dimension (diameter of the model)
        max_dimension = np.max(dimensions)
        return max_dimension
    
    def auto_scale_projection_size(self):
        """Automatically calculate and set projection size based on model dimensions and pixel size."""
        if self.pdb_data is None:
            messagebox.showwarning("No Structure", "Please load a structure file first.")
            return
        
        # Get pixel size from user input
        try:
            pixel_size_str = self.pixel_size_entry.get().strip()
            if not pixel_size_str:
                # Try to get from matched_data if available
                if self.matched_data is not None and 'pixel_size' in self.matched_data:
                    pixel_size = float(np.median(self.matched_data['pixel_size']))
                    self.pixel_size_entry.delete(0, tk.END)
                    self.pixel_size_entry.insert(0, f"{pixel_size:.3f}")
                    self.pixel_size_angstroms = pixel_size
                    print(f"Using pixel size from data: {pixel_size:.3f} Å/pixel")
                else:
                    messagebox.showwarning("Pixel Size Required", 
                                         "Please enter the pixel size in Angstroms per pixel.\n"
                                         "This is typically found in your cryoSPARC job output.")
                    return
            else:
                pixel_size = float(pixel_size_str)
                self.pixel_size_angstroms = pixel_size
        except ValueError:
            messagebox.showerror("Invalid Input", "Pixel size must be a number (e.g., 1.1)")
            return
        
        if pixel_size <= 0:
            messagebox.showerror("Invalid Input", "Pixel size must be greater than 0")
            return
        
        # Calculate model dimensions
        model_size_angstroms = self._calculate_model_dimensions()
        if model_size_angstroms is None:
            messagebox.showerror("Error", "Could not calculate model dimensions.")
            return
        
        # Calculate projection size in pixels
        # Add 20% padding for better visualization
        projection_size_pixels = int((model_size_angstroms / pixel_size) * 1.2)
        
        # Clamp to reasonable range
        projection_size_pixels = max(100, min(800, projection_size_pixels))
        
        # Update projection size
        old_size = self.projection_size
        self.projection_size = projection_size_pixels
        self.size_var.set(self.projection_size)
        self.size_label.config(text=f"{self.projection_size} px")
        
        print(f"Auto-scaled projection size:")
        print(f"  Model size: {model_size_angstroms:.1f} Å")
        print(f"  Pixel size: {pixel_size:.3f} Å/pixel")
        print(f"  Calculated projection size: {projection_size_pixels} px (was {old_size} px)")
        
        # Update display if micrograph is loaded
        if self.current_micrograph_idx is not None:
            # Resize cached projections if size changed significantly
            if abs(old_size - self.projection_size) > 10:
                self.update_display()
    
    def _euler_to_chimerax_view(self, euler_angles, preserve_translation=True):
        """
        Convert Euler angles (ZYZ convention, radians) to ChimeraX view command.
        
        CRITICAL ANGLE CONVERSION LOGIC:
        =================================
        
        In cryo-EM (.cs files):
        - Euler angles [phi, theta, psi] in ZYZ convention (radians)
        - Describe how to rotate the VOLUME to match the 2D projection
        - Rotation matrix R rotates from volume space to view space
        - Projection looks down +Z axis after rotation
        
        In PyMOL (for projections):
        - We rotate the OBJECT by the Euler angles sequentially: rotate Z(phi), Y(theta), Z(psi)
        - Then look at it from +Z (default view)
        - This matches the 2D projection
        
        In ChimeraX (for viewing):
        - We need to rotate the CAMERA to see the same view as PyMOL
        - If object is rotated by R, camera must be rotated by R^-1 = R^T
        - ChimeraX view matrix rotates the camera coordinate system
        
        Coordinate systems:
        - Cryo-EM/PyMOL/ChimeraX all use: Z toward viewer, Y up, X right
        - No coordinate system conversion needed, only camera vs object rotation
        
        VERIFICATION:
        - PyMOL: object rotated by R, viewed from +Z → matches projection
        - ChimeraX: camera rotated by R^T, viewing object → should match PyMOL view
        
        ChimeraX view matrix format (12 numbers, row-major):
        r11,r12,r13,tx,r21,r22,r23,ty,r31,r32,r33,tz
        """
        from scipy.spatial.transform import Rotation
        
        # Convert ZYZ Euler angles to rotation matrix (same as PyMOL uses)
        # This is the rotation that transforms volume to view space
        rot = Rotation.from_euler('ZYZ', euler_angles, degrees=False)
        R_volume_to_view = rot.as_matrix()
        
        # CRITICAL: ChimeraX view matrix interpretation
        # After testing, we found that ChimeraX's "view matrix camera" actually rotates
        # the SCENE/OBJECT (not the camera), similar to PyMOL's object rotation.
        # Therefore, we use the DIRECT rotation matrix (same as PyMOL), not the transpose.
        #
        # PyMOL: object rotated by R, viewed from +Z → matches projection
        # ChimeraX: scene rotated by R (via view matrix), viewed from +Z → should match PyMOL
        #
        # CORRECTION: ChimeraX view appears rotated 180° in-plane compared to PyMOL projection
        # This is likely due to coordinate system convention differences (Y-axis flip or similar)
        # Apply 180° rotation around Z axis to correct: R_180_z = [[-1,0,0],[0,-1,0],[0,0,1]]
        R_180_z = np.array([[-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 0.0, 1.0]])
        R_chimerax = R_volume_to_view @ R_180_z  # Apply 180° Z rotation correction
        
        # Debug: Print rotation matrices for verification
        phi, theta, psi = euler_angles[0], euler_angles[1], euler_angles[2]
        print(f"DEBUG ChimeraX: Euler=[{phi:.6f}, {theta:.6f}, {psi:.6f}] rad, "
              f"[{phi*180/np.pi:.2f}, {theta*180/np.pi:.2f}, {psi*180/np.pi:.2f}] deg")
        print(f"  R_volume_to_view (PyMOL object rotation):\n{R_volume_to_view}")
        print(f"  R_chimerax (with 180° Z correction):\n{R_chimerax}")
        
        # Translation: use zeros (centering handled separately)
        translation = [0.0, 0.0, 0.0]
        
        # Build the 12-number matrix string: row1, row2, row3, translation
        # Format: r11,r12,r13,tx,r21,r22,r23,ty,r31,r32,r33,tz
        # ChimeraX expects: view matrix camera r11,r12,r13,tx,r21,r22,r23,ty,r31,r32,r33,tz
        matrix_values = []
        for i in range(3):
            for j in range(3):
                matrix_values.append(f"{R_chimerax[i,j]:.6f}")
            matrix_values.append(f"{translation[i]:.6f}")
        
        matrix_str = ','.join(matrix_values)
        
        # Return the view command - ChimeraX expects: view matrix camera r11,r12,r13,tx,r21,r22,r23,ty,r31,r32,r33,tz
        # Make sure all values are properly formatted as floats
        return f"view matrix camera {matrix_str}"
    
    def _generate_pdb_projection_for_particle(self, particle_idx, euler_angles, output_size=(500, 500)):
        """Generate a single PDB projection for one particle at its specific pose.
        
        Returns:
            (H, W, 4) RGBA array in [0, 1] range
        """
        if self.pdb_data is None:
            raise RuntimeError("PDB data not loaded")
        
        # Debug: verify different Euler angles for different particles
        print(f"  Particle {particle_idx+1}: Euler angles = [{euler_angles[0]:.6f}, {euler_angles[1]:.6f}, {euler_angles[2]:.6f}]")
        
        # Generate projection using PDB structure with PyMOL
        # CRITICAL: Don't access GUI elements from background threads
        # Use stored paths instead of accessing entry widgets
        pdb_path = self.pdb_path if hasattr(self, 'pdb_path') and self.pdb_path else None
        
        # Get chimerax_path from stored value, not GUI (thread-safe)
        chimerax_path = None
        if hasattr(self, 'chimerax_path') and self.chimerax_path:
            chimerax_path = self.chimerax_path
        elif hasattr(self, 'chimerax_entry'):
            # Only access GUI if we're in the main thread (not background)
            # For background threads, use None (will auto-detect)
            try:
                # Try to get value, but catch if we're in a background thread
                import threading
                if threading.current_thread() is threading.main_thread():
                    chimerax_path = self.chimerax_entry.get()
                    if not chimerax_path or not Path(chimerax_path).exists():
                        chimerax_path = None  # Will auto-detect
                else:
                    # Background thread - don't access GUI, will auto-detect
                    chimerax_path = None
            except:
                # If accessing GUI fails, just use None (will auto-detect)
                chimerax_path = None
        
        # Determine chain_color_map based on color mode
        chain_color_map = None
        if self.color_mode == 'chain':
            chain_color_map = self.chain_color_map if self.chain_color_map else None
        
        # Get marker coordinates if PyMOL arrow rendering is enabled
        marker1_coords = None
        marker2_coords = None
        render_arrow_in_pymol = getattr(self, 'render_arrow_in_pymol_var', None)
        if render_arrow_in_pymol and render_arrow_in_pymol.get():
            # Get marker coordinates from GUI entries
            try:
                marker1_x = float(self.marker1_x_entry.get())
                marker1_y = float(self.marker1_y_entry.get())
                marker1_z = float(self.marker1_z_entry.get())
                marker2_x = float(self.marker2_x_entry.get())
                marker2_y = float(self.marker2_y_entry.get())
                marker2_z = float(self.marker2_z_entry.get())
                marker1_coords = np.array([marker1_x, marker1_y, marker1_z])
                marker2_coords = np.array([marker2_x, marker2_y, marker2_z])
            except (ValueError, AttributeError):
                # If marker entries don't exist or have invalid values, skip
                marker1_coords = None
                marker2_coords = None
        
        projection = project_pdb_structure(
            self.pdb_data,
            euler_angles,  # CRITICAL: Each particle gets its unique Euler angles from .cs file
            output_size=output_size,
            chain_color_map=chain_color_map,
            default_protein_color=self.default_protein_color,
            default_nucleic_color=self.default_nucleic_color,
            rotation_correction_x=self.rotation_correction_x,
            rotation_correction_y=self.rotation_correction_y,
            rotation_correction_z=self.rotation_correction_z,
            pdb_path=pdb_path,
            chimerax_path=chimerax_path,
            marker1_coords=marker1_coords,
            marker2_coords=marker2_coords,
            render_vector_arrow=(marker1_coords is not None and marker2_coords is not None)
        )
        
        if particle_idx < 2:
            print(f"  Generated PDB projection for particle {particle_idx+1}: shape={projection.shape}, "
                  f"RGB_range=[{projection[:,:,:3].min():.3f}, {projection[:,:,:3].max():.3f}], "
                  f"Alpha_range=[{projection[:,:,3].min():.3f}, {projection[:,:,3].max():.3f}]")
        
        return projection
    
    def _generate_single_projection_worker(self, micrograph_idx, particle_idx, pose, output_size):
        """Worker function to generate a single projection (thread-safe).
        
        This is called from worker threads, so it must be thread-safe.
        """
        try:
            # Generate projection using PDB structure
            projection = self._generate_pdb_projection_for_particle(
                particle_idx, pose, output_size=output_size
            )
            
            # Resize if needed to match projection_size
            if projection.shape[0] != output_size[0] or projection.shape[1] != output_size[1]:
                from PIL import Image
                rgba_uint8 = (projection * 255.0).astype(np.uint8)
                pil_img = Image.fromarray(rgba_uint8)
                pil_img_resized = pil_img.resize(output_size, Image.Resampling.LANCZOS)
                projection = np.array(pil_img_resized).astype(np.float32) / 255.0
                projection[:,:,3] = np.clip(projection[:,:,3], 0, 1)
            
            return (micrograph_idx, particle_idx, projection, None)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return (micrograph_idx, particle_idx, None, (e, error_trace))
    
    def _generate_all_projections(self, micrograph_idx, base_filename="proj_v1", limit=None, background=False):
        """Generate and cache projections for particles using PDB structure.
        
        Uses parallel processing with up to 3 worker threads for faster generation.
        GUI remains responsive during generation.
        
        Args:
            limit: If specified, only generate projections for first N particles.
                   If None, generate for all particles.
            background: If True, runs in background mode (checks for cancellation, less verbose).
        """
        if self.pdb_data is None or self.current_particles is None:
            return
        
        num_particles = len(self.current_particles['poses'])
        if limit is not None:
            num_to_generate = min(limit, num_particles)
            print(f"Generating {num_to_generate} unique PDB projections (of {num_particles}) for micrograph {micrograph_idx} using {min(3, num_to_generate)} parallel workers...")
        else:
            num_to_generate = num_particles
            print(f"Generating {num_particles} unique PDB projections for micrograph {micrograph_idx} using {min(3, num_particles)} parallel workers...")
        
        # Clean up old projection cache for this micrograph (only if not in background mode)
        if not background:
            with self.cache_lock:
                keys_to_remove = [k for k in self.projection_cache.keys() if k[0] == micrograph_idx]
                for key in keys_to_remove:
                    del self.projection_cache[key]
        
        # Prepare list of particles to generate
        particles_to_generate = []
        for particle_idx in range(num_to_generate):
            cache_key = (micrograph_idx, particle_idx)
            
            # Skip if already cached
            with self.cache_lock:
                if cache_key in self.projection_cache:
                    if not background:
                        print(f"  Skipping particle {particle_idx+1} - already cached")
                    continue
            
            pose = self.current_particles['poses'][particle_idx]
            particles_to_generate.append((particle_idx, pose))
        
        if not particles_to_generate:
            if not background:
                print("All projections already cached")
            return
        
        # Submit all tasks to thread pool
        futures = {}
        output_size = (self.projection_size, self.projection_size)
        
        for particle_idx, pose in particles_to_generate:
            future = self.executor.submit(
                self._generate_single_projection_worker,
                micrograph_idx, particle_idx, pose, output_size
            )
            futures[future] = particle_idx
        
        # Process completed tasks as they finish
        completed = 0
        failed_particles = []
        total_to_generate = len(particles_to_generate)
        
        # Update status initially
        if not background:
            self.status_var.set(f"Generating {total_to_generate} projections in parallel...")
            self.root.update()
        
        # Process futures as they complete
        for future in as_completed(futures):
            # Check for cancellation
            if background and not self.background_loading:
                print(f"Background preloading cancelled (user switched images)")
                # Cancel remaining futures
                for f in futures:
                    if not f.done():
                        f.cancel()
                # Reset generation flag
                self._generating_projections = False
                return
            
            if background and self.current_micrograph_idx != micrograph_idx:
                print(f"Background preloading cancelled (micrograph changed)")
                for f in futures:
                    if not f.done():
                        f.cancel()
                # Reset generation flag
                self._generating_projections = False
                return
            
            particle_idx = futures[future]
            try:
                result_mg_idx, result_particle_idx, projection, error = future.result()
                
                if error is not None:
                    # Generation failed
                    e, traceback_str = error
                    print(f"ERROR: Failed to generate PDB projection for particle {particle_idx+1}: {e}")
                    if not background:
                        print(traceback_str)
                    failed_particles.append(particle_idx + 1)
                else:
                    # Success - cache the projection (thread-safe)
                    cache_key = (result_mg_idx, result_particle_idx)
                    with self.cache_lock:
                        self.projection_cache[cache_key] = projection
                    
                    completed += 1
                    
                    # Progress update
                    if not background:
                        if completed % 5 == 0 or completed <= 3:
                            print(f"  Generated projection {completed}/{total_to_generate} (particle {particle_idx+1})")
                            self.status_var.set(f"Generated {completed}/{total_to_generate} PDB projections...")
                            self.root.update()
                    elif completed % 10 == 0:
                        print(f"  Background: Generated projection {completed}/{total_to_generate}")
                    
                    # Update display periodically to show progress
                    if not background:
                        if completed <= 3 or completed % 5 == 0:
                            self.root.after(0, self.update_display)
                    # Note: In background mode, we don't update display here to avoid thread issues
                    # The final update will be handled by the caller
                        
            except Exception as e:
                print(f"ERROR: Exception processing future for particle {particle_idx+1}: {e}")
                failed_particles.append(particle_idx + 1)
        
        # Final status update
        with self.cache_lock:
            cached_count = len([k for k in self.projection_cache.keys() if k[0] == micrograph_idx])
        
        print(f"Generated and cached {cached_count} PDB projections for micrograph {micrograph_idx}")
        if failed_particles:
            print(f"WARNING: Failed to generate projections for particles: {failed_particles}")
            if not background:
                messagebox.showwarning("Some projections failed", 
                    f"Failed to generate PDB projections for {len(failed_particles)} particles.\n"
                    f"Particle indices: {failed_particles}\n\n"
                    f"Check console for details.")
        
        # Update button state and status (schedule on main thread)
        def update_ui_final():
            # Reset generation flag when generation completes
            self._generating_projections = False
            
            if limit is not None and num_to_generate < num_particles:
                remaining = num_particles - cached_count
                self.status_var.set(f"Ready - {cached_count}/{num_particles} projections cached ({remaining} remaining)")
                if hasattr(self, 'generate_all_button'):
                    self.generate_all_button.config(state=tk.NORMAL)
            else:
                self.status_var.set(f"Ready - {cached_count} PDB projections cached")
                self.all_projections_generated = (cached_count >= num_particles)
                if hasattr(self, 'generate_all_button'):
                    self.generate_all_button.config(state=tk.DISABLED if self.all_projections_generated else tk.NORMAL)
            
            # Force display update after generation completes
            self.update_display()
        
        # Schedule UI update on main thread (only if not already in background thread)
        if not background:
            # We're in the main thread, can call directly
            update_ui_final()
        else:
            # We're in a background thread - schedule on main thread
            # Also reset flag immediately since we're done generating
            self._generating_projections = False
            try:
                self.root.after(0, update_ui_final)
            except RuntimeError:
                # Main thread not in main loop - flag already reset above
                pass
    
    def _cleanup_projection_cache(self):
        """Clean up projection cache for the previous micrograph."""
        if self.current_micrograph_idx is not None:
            # Remove cached projections for the previous micrograph (thread-safe)
            with self.cache_lock:
                keys_to_remove = [k for k in self.projection_cache.keys() 
                                if k[0] == self.current_micrograph_idx]
                for key in keys_to_remove:
                    del self.projection_cache[key]
            if keys_to_remove:
                print(f"Cleaned up {len(keys_to_remove)} cached projections")
    
    def _cleanup_projection_files(self):
        """Clean up temporary projection PNG files for the previous micrograph."""
        for filepath in self.current_projection_files:
            try:
                if filepath.exists():
                    filepath.unlink()
            except Exception as e:
                print(f"Warning: Could not delete projection file {filepath}: {e}")
        if self.current_projection_files:
            print(f"Cleaned up {len(self.current_projection_files)} temporary projection PNG files")
        self.current_projection_files = []
    
    def _cancel_background_loading(self):
        """Cancel any ongoing background loading."""
        self.background_loading = False
        if self.idle_timer is not None:
            self.root.after_cancel(self.idle_timer)
            self.idle_timer = None
    
    def _reset_idle_timer(self):
        """Reset the idle timer - will trigger background loading after 7 seconds of inactivity."""
        # Cancel existing timer
        if self.idle_timer is not None:
            self.root.after_cancel(self.idle_timer)
        
        # Set new timer for 7 seconds (7000 ms)
        self.idle_timer = self.root.after(7000, self._on_idle_timeout)
    
    def _on_idle_timeout(self):
        """Called when user has been idle for 7 seconds - start background preloading."""
        self.idle_timer = None
        
        # Only start background loading if:
        # 1. Not already loading
        # 2. We have a current micrograph
        # 3. We have particles
        # 4. We have PDB loaded
        if (not self.background_loading and 
            self.current_micrograph_idx is not None and
            self.current_particles is not None and
            len(self.current_particles['poses']) > 0 and
            self.pdb_data is not None):
            
            # Check how many projections we have cached (thread-safe)
            with self.cache_lock:
                num_cached = len([k for k in self.projection_cache.keys() 
                                if k[0] == self.current_micrograph_idx])
            num_particles = len(self.current_particles['poses'])
            
            # Preload up to background_preload_limit (more than default for smoother experience)
            target_limit = min(self.background_preload_limit, num_particles)
            if num_cached < target_limit:
                print(f"Idle timeout: Starting background preloading (have {num_cached}, target {target_limit})")
                self._start_background_preloading()
            else:
                print(f"Idle timeout: Already have {num_cached} projections (target {target_limit}), skipping background preload")
    
    def _start_background_generation(self, micrograph_idx, base_name, num_particles):
        """Start background generation of all projections for a micrograph."""
        if self.background_loading:
            print("Background generation already in progress, skipping")
            return
        
        self.background_loading = True
        print(f"Starting background generation for micrograph {micrograph_idx} ({num_particles} particles)")
        
        def generate_in_background():
            try:
                self._generate_all_projections(micrograph_idx, base_filename=base_name, 
                                             limit=None, background=True)
                if self.current_micrograph_idx == micrograph_idx:
                    with self.cache_lock:
                        num_cached = len([k for k in self.projection_cache.keys() 
                                        if k[0] == micrograph_idx])
                    print(f"Background generation complete: {num_cached} projections cached")
                    # Schedule UI update on main thread - wrap in try/except to handle thread issues
                    try:
                        self.root.after(0, self.update_display)
                    except RuntimeError as e:
                        # Main thread is not in main loop - this can happen in some edge cases
                        # Just skip the UI update, it will happen on next user interaction
                        print(f"Note: Could not schedule UI update: {e}")
            except Exception as e:
                print(f"Error in background generation: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.background_loading = False
        
        # Start background thread
        import threading
        self.background_thread = threading.Thread(target=generate_in_background, daemon=True)
        self.background_thread.start()
    
    def _start_background_preloading(self):
        """Start background thread to preload projections for current micrograph."""
        if self.background_loading:
            print("Background preloading already in progress, skipping")
            return  # Already loading
        
        if (self.current_micrograph_idx is None or 
            self.current_particles is None or
            len(self.current_particles['poses']) == 0):
            print("Cannot start background preloading - missing data")
            return
        
        self.background_loading = True
        self.status_var.set("Background: Preloading projections...")
        print(f"Starting background preloading thread for micrograph {self.current_micrograph_idx}")
        
        # Start background thread
        self.background_thread = threading.Thread(
            target=self._background_preload_projections,
            daemon=True
        )
        self.background_thread.start()
    
    def _background_preload_projections(self):
        """Background thread function to preload projections up to background_preload_limit."""
        try:
            micrograph_idx = self.current_micrograph_idx
            if micrograph_idx is None or micrograph_idx >= len(self.micrograph_files):
                print(f"Background preload: Invalid micrograph_idx {micrograph_idx}")
                return
            
            mg_file = self.micrograph_files[micrograph_idx]
            base_name = mg_file.stem
            
            # Check how many we already have (thread-safe)
            with self.cache_lock:
                num_cached = len([k for k in self.projection_cache.keys() 
                                if k[0] == micrograph_idx])
            num_particles = len(self.current_particles['poses'])
            
            # Preload up to background_preload_limit (but only generate what's missing)
            target_limit = min(self.background_preload_limit, num_particles)
            
            print(f"Background preload: micrograph {micrograph_idx}, cached: {num_cached}, target: {target_limit}, total particles: {num_particles}")
            
            if num_cached < target_limit:
                # Generate missing projections in background
                num_to_generate = target_limit - num_cached
                print(f"Background: Preloading {num_to_generate} projections for micrograph {micrograph_idx}...")
                # Use the existing function - it will skip already-cached particles
                self._generate_all_projections(micrograph_idx, base_filename=base_name, 
                                             limit=target_limit, background=True)
                
                # Don't update display automatically - only update status
                if self.current_micrograph_idx == micrograph_idx:
                    # Check final count
                    final_cached = len([k for k in self.projection_cache.keys() 
                                      if k[0] == micrograph_idx])
                    print(f"Background preload complete: {final_cached} projections now cached (display not updated)")
                    # Only update status, NOT the display
                    self.root.after(0, lambda: self.status_var.set(
                        f"Ready - {final_cached} projections cached (background preloading complete)"))
            else:
                print(f"Background preload: Already have {num_cached} projections (target {target_limit}), nothing to preload")
        except Exception as e:
            print(f"Background preloading error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.background_loading = False
            print("Background preloading thread finished")
    
    def apply_enhancements(self, image):
        """Apply image enhancements."""
        if image is None:
            return None
        
        enhanced = image.copy()
        
        # Low-pass filter
        if self.lowpass_A > 0:
            # Convert Angstroms to pixels (assuming ~1.1 Å/pixel)
            # Use a default pixel size if not available
            pixel_size = getattr(self, 'volume_pixel_size', 1.1)
            sigma_pixels = self.lowpass_A / pixel_size
            enhanced = gaussian_filter(enhanced, sigma=sigma_pixels)
        
        # Brightness and contrast
        enhanced = enhanced * self.contrast + self.brightness
        
        return enhanced
    
    def update_display(self, use_cache=False):
        """Update the micrograph display.
        
        Args:
            use_cache: If True and cache is valid, use cached images for fast toggle.
                      If False, regenerate and cache the images.
        """
        if self.original_micrograph is None:
            return
        
        # Save current zoom state before any updates
        # Only save if we have a valid zoom (not default 0-1 normalized)
        current_xlim = None
        current_ylim = None
        if hasattr(self, 'ax') and self.ax is not None and self.original_micrograph is not None:
            try:
                temp_xlim = list(self.ax.get_xlim())
                temp_ylim = list(self.ax.get_ylim())
                # Only save if limits are reasonable (not normalized 0-1)
                x_range = temp_xlim[1] - temp_xlim[0]
                y_range = temp_ylim[1] - temp_ylim[0]
                img_height, img_width = self.original_micrograph.shape[:2]
                # Save only if range is > 10 pixels and within reasonable bounds
                if (x_range > 10 and y_range > 10 and
                    temp_xlim[0] >= -100 and temp_xlim[1] <= img_width + 100 and
                    temp_ylim[0] >= -100 and temp_ylim[1] <= img_height + 100):
                    current_xlim = temp_xlim
                    current_ylim = temp_ylim
            except:
                pass
        
        # Check if we can use cached images for fast toggle
        # Only use cache if explicitly requested AND cache is valid
        use_cached = False
        if use_cache and self.cached_micrograph_idx == self.current_micrograph_idx:
            # CRITICAL: Always use original micrograph dimensions, not cached image dimensions
            # Cached images are rendered at canvas resolution, not original image resolution
            if self.original_micrograph is not None:
                img_height, img_width = self.original_micrograph.shape[:2]
            else:
                # No original micrograph - can't use cache properly
                use_cached = False
            
            if self.show_projections and self.cached_with_projections is not None:
                # Use cached image with projections
                # Cached images are at canvas resolution, need to map to original image coordinates
                cached_shape = self.cached_with_projections.shape
                cached_height, cached_width = cached_shape[0], cached_shape[1]
                
                self.ax.clear()
                # Use extent to map cached image pixels to original image coordinate space
                self.ax.imshow(self.cached_with_projections, origin='lower', aspect='equal',
                              extent=[-0.5, img_width - 0.5, -0.5, img_height - 0.5])
                self.ax.set_title(f"Micrograph {self.current_micrograph_idx+1}/{len(self.micrograph_paths)} - "
                                 f"{len(self.current_particles['poses'])} particles")
                # Set axes to full image dimensions
                self.ax.set_xlim(-0.5, img_width - 0.5)
                self.ax.set_ylim(-0.5, img_height - 0.5)
                self.ax.set_aspect('equal', adjustable='box')
                self.canvas.draw()
                use_cached = True
            elif not self.show_projections and self.cached_blank_image is not None:
                # Use cached blank image
                # Cached images are at canvas resolution, need to map to original image coordinates
                cached_shape = self.cached_blank_image.shape
                cached_height, cached_width = cached_shape[0], cached_shape[1]
                
                self.ax.clear()
                # Use extent to map cached image pixels to original image coordinate space
                self.ax.imshow(self.cached_blank_image, origin='lower', aspect='equal',
                              extent=[-0.5, img_width - 0.5, -0.5, img_height - 0.5])
                self.ax.set_title(f"Micrograph {self.current_micrograph_idx+1}/{len(self.micrograph_paths)} - "
                                 f"{len(self.current_particles['poses'])} particles")
                # Set axes to full image dimensions
                self.ax.set_xlim(-0.5, img_width - 0.5)
                self.ax.set_ylim(-0.5, img_height - 0.5)
                self.ax.set_aspect('equal', adjustable='box')
                self.canvas.draw()
                use_cached = True
        
        # If we used cached image, return early
        if use_cached:
            return
        
        # Save zoom limits if they exist (before clearing)
        # Use current limits if available, otherwise use saved zoom limits
        if current_xlim is not None and current_ylim is not None:
            # Use current limits (preserves any zoom/pan state)
            restore_zoom = True
            saved_xlim = current_xlim
            saved_ylim = current_ylim
        elif self.zoom_xlim is not None and self.zoom_ylim is not None:
            # Fall back to saved zoom limits
            restore_zoom = True
            saved_xlim = self.zoom_xlim
            saved_ylim = self.zoom_ylim
        else:
            restore_zoom = False
            saved_xlim = None
            saved_ylim = None
        
        self.ax.clear()
        
        # Apply enhancements
        display_image = self.apply_enhancements(self.original_micrograph)
        
        # Get image dimensions
        img_height, img_width = display_image.shape[:2]
        
        # Display micrograph with better contrast
        vmin, vmax = np.percentile(display_image, [1, 99])
        
        # Display image WITHOUT extent - let matplotlib use pixel coordinates
        im = self.ax.imshow(display_image, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
        self.ax.set_title(f"Micrograph {self.current_micrograph_idx+1}/{len(self.micrograph_paths)} - "
                         f"{len(self.current_particles['poses'])} particles")
        
        # CRITICAL: Set axes limits to match image dimensions IMMEDIATELY (pixel coordinates)
        # This must be done right after imshow to ensure the image displays correctly
        self.ax.set_xlim(-0.5, img_width - 0.5)
        self.ax.set_ylim(-0.5, img_height - 0.5)
        self.ax.set_aspect('equal', adjustable='box')
        
        # Disable auto-scaling to prevent matplotlib from adjusting limits when projections are drawn
        self.ax.set_autoscale_on(False)
        
        # Cache the blank image (without projections)
        if self.cached_micrograph_idx != self.current_micrograph_idx:
            # New micrograph - clear old cache
            self.cached_blank_image = None
            self.cached_with_projections = None
        
        # Store blank image for caching (we'll update this after drawing)
        # We'll capture the final rendered image after all drawing is complete
        
        # Don't restore zoom here - we'll do it AFTER all drawing is complete
        # This ensures projections don't cause axis limit shifts
        
        if self.current_particles is None or len(self.current_particles['poses']) == 0:
            # Cache blank image (no particles)
            self.canvas.draw()
            # Capture the rendered image for caching
            self._cache_current_display()
            return
        
        # Draw particles
        mg_shape = self.current_particles['micrograph_shape']
        if mg_shape is None:
            mg_shape = self.original_micrograph.shape
        
        particles_drawn = 0
        projections_drawn = 0
        print(f"\n=== Drawing {len(self.current_particles['poses'])} particles ===")
        print(f"Show projections: {self.show_projections}")
        with self.cache_lock:
            cache_size = len(self.projection_cache)
        print(f"Projection cache size: {cache_size}")
        print(f"Current micrograph idx: {self.current_micrograph_idx}")
        
        for i in range(len(self.current_particles['poses'])):
            x_frac = self.current_particles['center_x_frac'][i]
            y_frac = self.current_particles['center_y_frac'][i]
            x_pixel = x_frac * mg_shape[1]
            y_pixel = y_frac * mg_shape[0]
            
            # Apply 2D shifts from refinement (in Angstroms, convert to pixels)
            # cryoSPARC's alignments3D/shift contains [shift_x, shift_y] in Angstroms
            # Apply shifts to particle center position
            if 'shifts' in self.current_particles and len(self.current_particles['shifts']) > i:
                shift = self.current_particles['shifts'][i]  # [shift_x, shift_y] in Angstroms
                # Get pixel size from particle data or GUI entry field
                if 'pixel_size' in self.current_particles and len(self.current_particles['pixel_size']) > i:
                    pixel_size = self.current_particles['pixel_size'][i]
                else:
                    # Fallback to GUI entry field
                    try:
                        pixel_size = float(self.pixel_size_entry.get().strip())
                    except (ValueError, AttributeError):
                        pixel_size = 1.0  # Default fallback
                # Convert shifts from Angstroms to pixels
                # Apply shifts directly (not negated) - shifts represent how particle moved during refinement
                x_pixel += shift[0] / pixel_size
                y_pixel += shift[1] / pixel_size
            
            # Keep pixel coordinates as float for sub-pixel accuracy
            # Only convert to int for bounds checking
            x_pixel_int = int(round(x_pixel))
            y_pixel_int = int(round(y_pixel))
            
            if x_pixel_int < 0 or x_pixel_int >= display_image.shape[1] or \
               y_pixel_int < 0 or y_pixel_int >= display_image.shape[0]:
                continue
            
            pose = self.current_particles['poses'][i]
            
            # ALWAYS draw particle marker (star) FIRST - this is essential for identifying particles
            # Draw a star marker at the particle center with fill color #F42C04 and outline #4d4d4f
            marker_size = 12  # Size of star marker in pixels
            self.ax.plot(x_pixel, y_pixel, marker='*', markersize=marker_size, 
                        markerfacecolor='#F42C04', markeredgecolor='#4d4d4f', 
                        markeredgewidth=1.5, alpha=1.0, zorder=15, linestyle='None')
            
            # Load projection data if needed (for either projection display or outline)
            # We need the projection data even if only drawing outline
            projection = None
            rgba = None
            extent = None
            
            # Draw projection overlay - ALWAYS show per-particle projections (this is the core feature)
            # Only skip if explicitly disabled by user
            # Show all projections that are cached (check cache, not limit)
            if self.show_projections or self.show_outlines:
                try:
                    # CRITICAL: Each particle gets its own unique PDB-generated projection based on its 3D pose
                    cache_key = (self.current_micrograph_idx, i)
                    # Thread-safe cache access
                    with self.cache_lock:
                        if cache_key in self.projection_cache:
                            # Use cached PDB-generated projection
                            projection = self.projection_cache[cache_key].copy()  # Copy to avoid race conditions
                        else:
                            # Projection not in cache - skip drawing this one
                            if i < 5:  # Only warn for first few
                                print(f"Note: Projection not cached for particle {i} (will be generated when 'Generate All' is clicked)")
                            continue  # Skip this particle's projection, but still draw the marker
                        if i < 3:  # Debug first few particles
                            if len(projection.shape) == 3 and projection.shape[2] == 4:
                                print(f"Particle {i}: Using cached RGBA projection, shape={projection.shape}, "
                                      f"RGB_range=[{projection[:,:,:3].min():.3f}, {projection[:,:,:3].max():.3f}], "
                                      f"Alpha_range=[{projection[:,:,3].min():.3f}, {projection[:,:,3].max():.3f}], "
                                      f"at pixel ({x_pixel:.2f}, {y_pixel:.2f})")
                            else:
                                print(f"Particle {i}: Using cached projection, shape={projection.shape}, "
                                      f"at pixel ({x_pixel:.2f}, {y_pixel:.2f})")
                    # Store first particle's projection for preview (convert RGBA to grayscale for preview)
                    if i == 0:
                        if len(projection.shape) == 3 and projection.shape[2] == 4:
                            # Convert RGBA to grayscale for preview
                            preview = (0.299 * projection[:,:,0] + 
                                      0.587 * projection[:,:,1] + 
                                      0.114 * projection[:,:,2])
                            # Apply alpha mask
                            preview = preview * projection[:,:,3]
                        else:
                            preview = projection.copy()
                        self.current_projection = preview
                        if self.show_projection_preview:
                            self.update_preview()
                    
                    # Overlay the unique projection for this particle
                    # Calculate extent in data coordinates using float precision for sub-pixel accuracy
                    # 
                    # CRITICAL CENTERING ISSUE:
                    # The projection is rendered centered in its image (structure centered at image center).
                    # We place the projection image centered at the particle center from cryoSPARC (x_pixel, y_pixel).
                    # However, if the structure's center of mass (used by PyMOL's cmd.center()) doesn't exactly
                    # match the particle center from cryoSPARC, there will be a small offset. This is typically
                    # negligible (< 1 pixel) but can cause slight misalignment, especially if:
                    # 1. The structure's center of mass differs from the particle center
                    # 2. The rotation center (structure COM) doesn't match the particle center
                    # 3. There are small coordinate system differences between cryoSPARC and PyMOL
                    #
                    # The projection_offset_x/y sliders allow manual fine-tuning to correct for these small offsets.
                    
                    # Calculate automatic COM offset correction if enabled
                    auto_offset_x = 0.0
                    auto_offset_y = 0.0
                    if hasattr(self, 'auto_correct_com_offset') and self.auto_correct_com_offset:
                        if self.pdb_data is not None:
                            try:
                                euler_angles = self.current_particles['poses'][i]
                                if 'pixel_size' in self.current_particles and len(self.current_particles['pixel_size']) > i:
                                    pixel_size = self.current_particles['pixel_size'][i]
                                else:
                                    try:
                                        pixel_size = float(self.pixel_size_entry.get().strip())
                                    except (ValueError, AttributeError):
                                        pixel_size = 1.0
                                
                                # Get shifts if available
                                shifts_angstroms = None
                                if 'shifts' in self.current_particles and len(self.current_particles['shifts']) > i:
                                    shift = self.current_particles['shifts'][i]
                                    shifts_angstroms = (shift[0], shift[1])
                                
                                auto_offset_x, auto_offset_y = calculate_com_offset_correction(
                                    self.pdb_data, euler_angles,
                                    (x_pixel, y_pixel), pixel_size, self.projection_size,
                                    shifts_angstroms=shifts_angstroms
                                )
                                
                                # Debug output for first few particles
                                if i < 3:
                                    print(f"Particle {i}: Auto COM offset = ({auto_offset_x:.2f}, {auto_offset_y:.2f}) pixels")
                            except Exception as e:
                                # If calculation fails, use zero offset
                                if i < 3:  # Only warn for first few
                                    print(f"Warning: Failed to calculate COM offset for particle {i}: {e}")
                                    import traceback
                                    traceback.print_exc()
                    
                    half_size = self.projection_size / 2.0  # Use float division
                    center_x = x_pixel + self.projection_offset_x + auto_offset_x
                    center_y = y_pixel + self.projection_offset_y + auto_offset_y
                    extent = [center_x - half_size, center_x + half_size,
                             center_y - half_size, center_y + half_size]
                    
                    # CRITICAL: projection is now an RGBA array (0-1 range) directly from PDB rendering
                    # It already has the correct colors and transparency - just apply user's alpha setting
                    if len(projection.shape) == 3 and projection.shape[2] == 4:
                        # It's an RGBA array - use it directly
                        rgba = projection.copy()
                        
                        # CRITICAL: Use PDB images directly - they already have all the detail we need
                        # Do NOT apply any contrast enhancement that would destroy fine detail
                        # Determine which pixels have structure based on original alpha channel
                        original_alpha = rgba[:, :, 3].copy()
                        structure_mask = original_alpha > 0.01  # Pixels with any structure
                        
                        # Preserve the original RGB values from PDB rendering - they already have perfect detail
                        # Just ensure values are in valid [0, 1] range (they should already be)
                        rgba[:, :, :3] = np.clip(rgba[:, :, :3], 0, 1)
                        
                        # Apply user's projection_alpha setting to the alpha channel
                        # Where there's structure, set alpha to user's setting directly (not multiplied)
                        # This ensures alpha=1.0 makes overlays fully opaque, regardless of PDB rendering's original alpha
                        # Force to exactly 1.0 if user wants fully opaque (>= 0.99), otherwise use exact user value
                        if self.projection_alpha >= 0.99:
                            # Force fully opaque for structure pixels - use exactly 1.0
                            rgba[:, :, 3] = np.where(structure_mask, 1.0, 0.0)
                        else:
                            # Use user's alpha value directly for structure pixels
                            rgba[:, :, 3] = np.where(structure_mask, float(self.projection_alpha), 0.0)
                        
                        # Ensure alpha is in valid range [0, 1] (should already be, but clip to be safe)
                        rgba[:, :, 3] = np.clip(rgba[:, :, 3], 0, 1)
                        
                        # Debug: check RGBA values
                        if i < 3:
                            alpha_min = rgba[:,:,3].min()
                            alpha_max = rgba[:,:,3].max()
                            alpha_mean = rgba[:,:,3].mean()
                            structure_pixels = (rgba[:,:,3] > 0.01).sum()
                            fully_opaque_pixels = (rgba[:,:,3] >= 0.99).sum()
                            print(f"    Drawing RGBA projection: shape={rgba.shape}, "
                                  f"RGB_range=[{rgba[:,:,:3].min():.3f}, {rgba[:,:,:3].max():.3f}], "
                                  f"Alpha_range=[{alpha_min:.3f}, {alpha_max:.3f}], Alpha_mean={alpha_mean:.3f}, "
                                  f"structure_pixels={structure_pixels}, fully_opaque_pixels={fully_opaque_pixels}, "
                                  f"user_alpha={self.projection_alpha:.2f}, "
                                  f"extent={extent}")
                            if self.projection_alpha >= 0.99 and alpha_max < 0.99:
                                print(f"    ERROR: Alpha max is {alpha_max:.3f}, should be 1.0 for fully opaque!")
                            elif self.projection_alpha < 0.99 and abs(alpha_max - self.projection_alpha) > 0.01:
                                print(f"    WARNING: Alpha max is {alpha_max:.3f}, expected {self.projection_alpha:.2f}")
                        
                        # Draw projection if enabled (but prepare rgba for outline too)
                        if self.show_projections:
                            # Draw with proper transparency - use the original RGBA directly
                            # Note: extent is [left, right, bottom, top] in data coordinates
                            # origin='lower' means y-axis increases upward (standard for images)
                            self.ax.imshow(rgba, extent=extent, origin='lower', 
                                          interpolation='bilinear', zorder=10, aspect='auto')
                            projections_drawn += 1
                    else:
                        # Fallback: if it's not RGBA, skip it
                        if i < 3:
                            print(f"    Warning: Projection is not RGBA format, shape={projection.shape}, skipping")
                        continue
                except Exception as e:
                    print(f"Error loading projection for particle {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Draw outline if enabled (independent of projection display)
            if self.show_outlines and projection is not None and extent is not None:
                try:
                    # Use matplotlib's contour function directly - this ensures perfect alignment
                    # with imshow since it uses the same coordinate system
                    from scipy.ndimage import binary_erosion
                    
                    # Create binary mask from alpha channel (structure pixels)
                    # Use original alpha from projection for structure detection (not modified by projection_alpha)
                    if len(projection.shape) == 3 and projection.shape[2] == 4:
                        original_alpha = projection[:, :, 3].copy()
                    else:
                        # Fallback if projection is not RGBA
                        continue
                    alpha_mask = original_alpha > 0.01
                    
                    if alpha_mask.any():
                        # Erode to find edge (boundary between structure and background)
                        eroded = binary_erosion(alpha_mask)
                        edge = alpha_mask & ~eroded
                        
                        # Create coordinate arrays matching the extent used by imshow
                        # extent = [left, right, bottom, top]
                        # With origin='lower', array row 0 is at bottom, row H-1 is at top
                        # So we create y coordinates from bottom to top
                        x = np.linspace(extent[0], extent[1], projection.shape[1])
                        y = np.linspace(extent[2], extent[3], projection.shape[0])
                        X, Y = np.meshgrid(x, y)
                        
                        # Draw contour at edge - matplotlib's contour automatically handles
                        # the coordinate system matching with imshow when using the same X, Y grids
                        # Note: edge array is indexed as [row, col] where row=0 is bottom (with origin='lower')
                        # and contour expects the same indexing, so this should align perfectly
                        self.ax.contour(X, Y, edge.astype(float), levels=[0.5], 
                                       colors=['#F2570C'], linewidths=2, zorder=11)
                except Exception as e:
                    if i < 3:
                        print(f"    Warning: Could not draw outline: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Draw orientation arrow if enabled (markers already drawn above)
            if self.show_orientations:
                try:
                    dx, dy = get_particle_orientation_arrow(pose, length=self.arrow_length)
                    self.ax.arrow(x_pixel, y_pixel, dx, dy,
                                 head_width=self.arrow_length*0.3,
                                 head_length=self.arrow_length*0.2,
                                 fc='#0F1A20', ec='#0F1A20', alpha=0.8, linewidth=1.5, zorder=16)
                except Exception as e:
                    print(f"Error drawing orientation {i}: {e}")
            
            # Draw custom structural vector arrow if enabled
            if self.show_custom_arrow and self.custom_vector_3d is not None:
                try:
                    from scipy.spatial.transform import Rotation
                    rot = Rotation.from_euler('ZYZ', pose, degrees=False)
                    R = rot.as_matrix()
                    
                    # Get pixel size for proper conversion
                    pixel_size = self.pixel_size_angstroms if self.pixel_size_angstroms is not None else 1.0
                    
                    # CRITICAL: For 'from_markers' method, draw arrow from marker 1 to marker 2
                    # For other methods, draw from particle center
                    if self.custom_vector_method == 'from_markers' and self.marker_positions is not None and len(self.marker_positions) >= 2:
                        # Get structure center (same as PyMOL uses via cmd.center())
                        # PyMOL centers on COM, which is the mean of all atom coordinates
                        if hasattr(self, 'pdb_data') and self.pdb_data is not None:
                            # Use the same calculation as PyMOL's cmd.center()
                            structure_center = np.mean(self.pdb_data['coords'], axis=0)
                        else:
                            structure_center = np.array([0.0, 0.0, 0.0])
                        
                        # Get marker positions (in ChimeraX/PDB absolute coordinates, Angstroms)
                        # CONFIRMED: According to ChimeraX documentation, markers use the model's local
                        # coordinate system (same as the PDB file) unless coordinateSystem is specified.
                        # "x,y,z values are interpreted in scene coordinates unless a different 
                        # coordinateSystem (reference model number) is given."
                        # Since we're placing markers on the structure model, they're in the PDB coordinate system.
                        marker1 = self.marker_positions[0].copy()
                        marker2 = self.marker_positions[1].copy()
                        
                        # CRITICAL: Transform markers exactly as PyMOL transforms the structure:
                        # 1. Center markers at structure COM (same as PyMOL's cmd.center())
                        #    Markers are in the same coordinate system as the PDB file, so this is correct
                        centered_marker1 = marker1 - structure_center
                        centered_marker2 = marker2 - structure_center
                        
                        # 2. Apply particle rotation (same rotation matrix R that PyMOL uses)
                        # R rotates from model space to view space
                        rotated_marker1 = R @ centered_marker1
                        rotated_marker2 = R @ centered_marker2
                        
                        # 3. Project onto XY plane (drop Z coordinate) - this is the projection plane
                        # PyMOL renders the projection with Z pointing out of screen
                        # The projection image shows X (right) and Y (up) axes
                        # Convert from Angstroms to pixels
                        # CRITICAL: The projection image is placed with extent and origin='lower'
                        # This means the projection's coordinate system matches the micrograph:
                        # - X: left to right (matches)
                        # - Y: bottom to top (matches, because origin='lower')
                        # The projection is centered at (center_x, center_y) in micrograph coordinates
                        # So a point at (0, 0) in projection space (center) maps to (center_x, center_y)
                        # 
                        # IMPORTANT: Coordinate system considerations:
                        # - PyMOL images are loaded with PIL (top-left origin, Y down)
                        # - matplotlib with origin='lower' interprets them (bottom-left origin, Y up)
                        # - ChimeraX marker coordinates may be in a different coordinate system than PDB
                        # 
                        # CRITICAL: PyMOL image coordinate system transformation
                        # PyMOL renders images with top-left origin (Y down in image coordinates)
                        # matplotlib with origin='lower' interprets them with bottom-left origin (Y up)
                        # This means the Y coordinate needs to be negated to match the display
                        #
                        # When PyMOL projects 3D coordinates to 2D image:
                        # - Image has top-left origin: Y increases downward
                        # - matplotlib with origin='lower': Y increases upward
                        # - So we need to negate Y to account for the flip
                        # CRITICAL: Coordinate system transformation for PyMOL images
                        # PyMOL images are saved with top-left origin (Y down), but matplotlib
                        # with origin='lower' interprets them with bottom-left origin (Y up)
                        # 
                        # When we project 3D coordinates to 2D:
                        # - rotated_marker[0] is X in view space (right = positive)
                        # - rotated_marker[1] is Y in view space (up = positive)
                        # 
                        # In PyMOL's image coordinate system (before display):
                        # - X: right (matches)
                        # - Y: down (opposite of view space Y)
                        # 
                        # When displayed with origin='lower':
                        # - X: right (matches)
                        # - Y: up (flipped from PyMOL's image Y)
                        # 
                        # CRITICAL: Coordinate system transformation for PyMOL images
                        # Based on ChimeraX geometry documentation, markers are in the model's
                        # local coordinate system (PDB coordinates), which should match our PDB file.
                        # 
                        # PyMOL's viewport uses OpenGL-style rendering:
                        # - In 3D view space: X right, Y up, Z out of screen
                        # - When rendered to image: X right, Y down (top-left origin)
                        # - When displayed with matplotlib origin='lower': X right, Y up (bottom-left origin)
                        # 
                        # The rotation matrix R transforms from model space to view space.
                        # After rotation, we project to 2D by taking X and Y coordinates.
                        # 
                        # However, PyMOL's image coordinate system may differ from the view space
                        # coordinate system. We need to account for how PyMOL maps view space
                        # coordinates to image pixels.
                        # 
                        # CRITICAL: The ChimeraX COM transformation is CORRECT (pink circle aligns with white star)
                        # This means the coordinate system transformation is correct:
                        # - Use rotated[1] for X pixel coordinate
                        # - Use -rotated[0] for Y pixel coordinate
                        # 
                        # The user says it's pointing the right direction and so close, just needs tiny X translation.
                        # Current transformation: rotated[1] for X, rotated[0] for Y (correct direction, slightly off in X)
                        # Use the user-adjustable X offset from the GUI to fine-tune the position
                        # This will help us determine the exact offset needed and figure out why it's needed
                        # CRITICAL INVESTIGATION: Why do we need -70 X and -30 Y offset?
                        # 
                        # ROOT CAUSE ANALYSIS:
                        # The projection size is calculated as: (model_size / pixel_size) * 1.2
                        # This means the projection is scaled with a 1.2x factor for padding.
                        # When PyMOL renders with cmd.zoom(complete=1), it scales the structure to fit
                        # the viewport. The actual scale factor in the rendered image is:
                        #   actual_scale = projection_size / (model_size * 1.2 / pixel_size)
                        #   = projection_size * pixel_size / (model_size * 1.2)
                        #
                        # But we're converting marker coordinates using pixel_size directly, which assumes
                        # the scale is exactly pixel_size. We should use the actual scale factor from
                        # the projection rendering.
                        #
                        # Calculate the actual scale factor from projection size and model dimensions:
                        # CRITICAL: PyMOL's zoom(complete=1) scales based on the ROTATED bounding box,
                        # not the original model size. We need to calculate the bounding box size
                        # after rotation to get the correct scale factor.
                        model_size_angstroms = None
                        rotated_bbox_size = None
                        if hasattr(self, 'pdb_data') and self.pdb_data is not None:
                            coords = self.pdb_data['coords']
                            if len(coords) > 0:
                                # Calculate original bounding box size
                                min_coords = coords.min(axis=0)
                                max_coords = coords.max(axis=0)
                                model_size_angstroms = np.max(max_coords - min_coords)
                                
                                # Calculate bounding box size AFTER rotation
                                # PyMOL's zoom(complete=1) likely uses the 3D bounding box diagonal
                                # or the maximum extent in any dimension after rotation
                                # Center coordinates first
                                center = coords.mean(axis=0)
                                coords_centered = coords - center
                                # Apply rotation
                                coords_rotated = (R @ coords_centered.T).T
                                # Calculate rotated bounding box
                                min_rotated = coords_rotated.min(axis=0)
                                max_rotated = coords_rotated.max(axis=0)
                                
                                # Try multiple methods to see which matches PyMOL:
                                # Method 1: 2D projection extent (XY plane)
                                rotated_2d_extent_xy = np.max(max_rotated[:2] - min_rotated[:2])
                                # Method 2: Maximum extent in any single dimension
                                rotated_max_extent = np.max(max_rotated - min_rotated)
                                # Method 3: 3D bounding box diagonal
                                rotated_3d_diagonal = np.linalg.norm(max_rotated - min_rotated)
                                # Method 4: Maximum distance from center in XY plane
                                rotated_max_dist_xy = np.max(np.linalg.norm(coords_rotated[:, :2], axis=1)) * 2
                                
                                # Try using "Max dist from center (XY)" - this might better match PyMOL's zoom
                                # PyMOL's zoom(complete=1) likely uses the maximum distance from center
                                # in the projection plane, not the bounding box extent
                                # This gives us the radius, so we double it for diameter
                                rotated_bbox_size = rotated_max_dist_xy  # Already doubled in calculation above
                                
                                # Alternative: Use 2D XY extent (previous method)
                                # rotated_bbox_size = rotated_2d_extent_xy * 2  # Full extent (diameter)
                                
                                # Store alternative calculations for debugging
                                if i == 0:  # Only print for first particle to avoid spam
                                    print(f"DEBUG: Rotated bbox calculations:")
                                    print(f"  2D XY extent: {rotated_2d_extent_xy:.2f} Å (diameter: {rotated_bbox_size:.2f} Å)")
                                    print(f"  Max extent (any dim): {rotated_max_extent:.2f} Å")
                                    print(f"  3D diagonal: {rotated_3d_diagonal:.2f} Å")
                                    print(f"  Max dist from center (XY): {rotated_max_dist_xy:.2f} Å")
                        
                        # Calculate effective pixel size based on projection scaling
                        # CRITICAL INSIGHT: 
                        # - projection_size is calculated from ORIGINAL model_size: (model_size / pixel_size) * 1.2
                        # - But PyMOL's zoom(complete=1) scales based on ROTATED bbox_size
                        # - This mismatch causes a systematic offset!
                        #
                        # The rotated bbox is typically larger than the original model (e.g., 299.76 Å vs 232.11 Å)
                        # So PyMOL has to fit a larger structure into the same projection_size
                        # This causes the structure to be scaled down more than expected
                        #
                        # Solution: Calculate what the projection_size SHOULD be for the rotated bbox,
                        # then use that to determine the correct pixel size
                        if rotated_bbox_size is not None and rotated_bbox_size > 0:
                            # Calculate what projection_size should be for this rotated bbox
                            correct_projection_size = (rotated_bbox_size / pixel_size) * 1.2
                            
                            # The effective pixel size is what PyMOL actually uses
                            # PyMOL fits rotated_bbox_size into (projection_size / 1.2) pixels
                            # But it SHOULD fit into (correct_projection_size / 1.2) pixels
                            # The ratio tells us how much the scale is off
                            scale_ratio = (self.projection_size / 1.2) / (correct_projection_size / 1.2)
                            scale_ratio = self.projection_size / correct_projection_size
                            
                            # The actual pixel size PyMOL uses
                            actual_scale_pixels = self.projection_size / 1.2
                            effective_pixel_size = rotated_bbox_size / actual_scale_pixels
                            
                            # CRITICAL: The offset is because projection_size is too small
                            # The structure is scaled by scale_ratio, which is < 1.0
                            # This means coordinates are scaled down more than they should be
                            # We need to account for this in the coordinate transformation
                            
                            if i == 0:  # Debug output
                                print(f"DEBUG: Projection size mismatch:")
                                print(f"  Original model_size: {model_size_angstroms:.2f} Å")
                                print(f"  Rotated bbox_size: {rotated_bbox_size:.2f} Å")
                                print(f"  Current projection_size: {self.projection_size} px")
                                print(f"  Correct projection_size: {correct_projection_size:.1f} px")
                                print(f"  Scale ratio: {scale_ratio:.4f}")
                                print(f"  Effective pixel size: {effective_pixel_size:.4f} Å/pixel")
                        elif model_size_angstroms is not None and model_size_angstroms > 0:
                            # Fallback to original model size if rotated bbox calculation failed
                            effective_pixel_size = model_size_angstroms / (self.projection_size / 1.2)
                        else:
                            effective_pixel_size = pixel_size  # Final fallback to alignment pixel size
                        
                        # Check for micrograph pixel size mismatch
                        micrograph_pixel_size = pixel_size  # Default to alignment pixel size
                        if 'micrograph_psize' in self.current_particles and len(self.current_particles['micrograph_psize']) > i:
                            micrograph_pixel_size = self.current_particles['micrograph_psize'][i]
                        
                        # Use user-adjustable X and Y offsets from GUI for debugging
                        # Get the current offset values (default to 0.0 if not set)
                        x_offset = getattr(self, 'marker_x_offset', 0.0)
                        y_offset = getattr(self, 'marker_y_offset', 0.0)
                        
                        # CRITICAL: The -70 pixel X offset suggests a systematic coordinate system issue.
                        # The effective pixel size is very close to alignment pixel size (1.1097 vs 1.1060),
                        # so the 1.2x scaling factor doesn't explain the large offset.
                        #
                        # Possible causes:
                        # 1. PyMOL's zoom(complete=1) might not scale exactly as we calculate
                        # 2. There might be a coordinate system offset in PyMOL's image rendering
                        # 3. The projection extent calculation might not match the actual image center
                        #
                        # CRITICAL: Coordinate transformation for markers
                        # 
                        # Coordinate system conventions:
                        # 1. Model space: PDB coordinates in Angstroms
                        # 2. View space: After rotation R (from model to view)
                        # 3. Projection plane: XY plane in view space (Z=0)
                        # 4. Image coordinates: X right, Y up (Matplotlib convention)
                        #
                        # Transformation steps:
                        # - Center marker: marker_centered = marker_model - structure_center
                        # - Rotate to view space: marker_rotated = R @ marker_centered
                        # - Project to 2D: Use X and Y components (Z is depth, ignored)
                        # - Convert to pixels: Divide by effective_pixel_size
                        #
                        # IMPORTANT: The coordinate swap (rotated[1] for X, rotated[0] for Y) is correct
                        # and verified by tests. This accounts for PyMOL's coordinate system.
                        #
                        # CRITICAL: The effective_pixel_size already accounts for PyMOL's zoom scaling
                        # based on the rotated_bbox_size. 
                        #
                        # The fact that Y works correctly (0 offset) but X needs -70 pixels suggests
                        # a coordinate system issue specific to X, not a general scaling problem.
                        #
                        # TODO: Consider rendering the arrow directly in PyMOL to avoid coordinate
                        # transformation issues. This would ensure the arrow is in the same coordinate
                        # system and scale as the structure projection.
                        #
                        # For now, use effective_pixel_size directly (it already accounts for rotated bbox)
                        marker1_x_pixels = rotated_marker1[1] / effective_pixel_size + x_offset
                        marker1_y_pixels = rotated_marker1[0] / effective_pixel_size + y_offset
                        marker2_x_pixels = rotated_marker2[1] / effective_pixel_size + x_offset
                        marker2_y_pixels = rotated_marker2[0] / effective_pixel_size + y_offset
                        
                        # DEBUG: Print scaling information
                        if i == 0:
                            print(f"DEBUG: Marker coordinate conversion:")
                            print(f"  Alignment pixel size: {pixel_size:.4f} Å/pixel")
                            print(f"  Micrograph pixel size: {micrograph_pixel_size:.4f} Å/pixel")
                            print(f"  Model size (original): {model_size_angstroms:.2f} Å" if model_size_angstroms else "  Model size: unknown")
                            print(f"  Rotated bbox size (2D extent): {rotated_bbox_size:.2f} Å" if rotated_bbox_size else "  Rotated bbox size: unknown")
                            print(f"  Projection size: {self.projection_size} pixels")
                            print(f"  Effective pixel size (using rotated bbox): {effective_pixel_size:.4f} Å/pixel")
                            if rotated_bbox_size is not None and rotated_bbox_size > 0 and model_size_angstroms is not None:
                                correct_projection_size = (rotated_bbox_size / pixel_size) * 1.2
                                scale_ratio = self.projection_size / correct_projection_size
                                scale_correction = 1.0 / scale_ratio
                                print(f"  Scale correction factor: {scale_correction:.4f}")
                            print(f"  User offsets - X: {x_offset}, Y: {y_offset} pixels")
                        
                        # 4. Position markers on micrograph
                        # CRITICAL: Use the EXACT same center calculation as projection placement
                        # The projection is placed with: center_x = x_pixel + projection_offset_x + auto_offset_x
                        # We must use the same calculation for markers to ensure they align with the projection
                        
                        # Calculate auto_offset if enabled (same as projection placement code)
                        auto_offset_x = 0.0
                        auto_offset_y = 0.0
                        if hasattr(self, 'auto_correct_com_offset') and self.auto_correct_com_offset:
                            if self.pdb_data is not None:
                                try:
                                    if 'pixel_size' in self.current_particles and len(self.current_particles['pixel_size']) > i:
                                        ps = self.current_particles['pixel_size'][i]
                                    else:
                                        try:
                                            ps = float(self.pixel_size_entry.get().strip())
                                        except (ValueError, AttributeError):
                                            ps = 1.0
                                    shifts_angstroms = None
                                    if 'shifts' in self.current_particles and len(self.current_particles['shifts']) > i:
                                        shift = self.current_particles['shifts'][i]
                                        shifts_angstroms = (shift[0], shift[1])
                                    auto_offset_x, auto_offset_y = calculate_com_offset_correction(
                                        self.pdb_data, pose,
                                        (x_pixel, y_pixel), ps, self.projection_size,
                                        shifts_angstroms=shifts_angstroms
                                    )
                                except Exception:
                                    pass  # Use zero offset if calculation fails
                        
                        # Calculate projection center (EXACT same as projection placement at line 2551-2552)
                        center_x = x_pixel + self.projection_offset_x + auto_offset_x
                        center_y = y_pixel + self.projection_offset_y + auto_offset_y
                        
                        # Markers are offset from projection center by their projected positions
                        marker1_x = center_x + marker1_x_pixels
                        marker1_y = center_y + marker1_y_pixels
                        marker2_x = center_x + marker2_x_pixels
                        marker2_y = center_y + marker2_y_pixels
                        
                        # Draw visual markers at calculated positions for debugging
                        # Draw a small circle at marker 1 position
                        self.ax.plot(marker1_x, marker1_y, 'o', color='yellow', markersize=8, 
                                   markeredgecolor='black', markeredgewidth=1, zorder=25, label='Marker 1' if i == 0 else '')
                        # Draw a small circle at marker 2 position
                        self.ax.plot(marker2_x, marker2_y, 's', color='cyan', markersize=8, 
                                   markeredgecolor='black', markeredgewidth=1, zorder=25, label='Marker 2' if i == 0 else '')
                        
                        # Draw a star at the projection center (particle center)
                        self.ax.plot(center_x, center_y, '*', color='white', markersize=10, 
                                   markeredgecolor='black', markeredgewidth=1, zorder=26, 
                                   label='Projection Center' if i == 0 else '')
                        
                        # DEBUG: Print marker positions for first particle only
                        if i == 0:
                            print(f"\n=== DEBUG Marker Projection (particle {i}) ===")
                            print(f"Structure center (COM, calculated): {structure_center}")
                            print(f"Structure center (COM, from ChimeraX): [246.62, 222.54, 307.65]")
                            print(f"COM difference: {structure_center - np.array([246.62, 222.54, 307.65])}")
                            print(f"Marker 1 (absolute, from ChimeraX): {marker1}")
                            print(f"Marker 2 (absolute, from ChimeraX): {marker2}")
                            print(f"Marker 1 (centered): {centered_marker1}")
                            print(f"Marker 2 (centered): {centered_marker2}")
                            print(f"Marker 1 (rotated): {rotated_marker1}")
                            print(f"Marker 2 (rotated): {rotated_marker2}")
                            print(f"Marker 1 (projected, pixels from center): ({marker1_x_pixels:.2f}, {marker1_y_pixels:.2f})")
                            print(f"Marker 2 (projected, pixels from center): ({marker2_x_pixels:.2f}, {marker2_y_pixels:.2f})")
                            print(f"Particle center (pixels): ({x_pixel:.2f}, {y_pixel:.2f})")
                            print(f"Projection center (with offsets): ({center_x:.2f}, {center_y:.2f})")
                            print(f"Projection extent: left={center_x - self.projection_size/2:.2f}, right={center_x + self.projection_size/2:.2f}, bottom={center_y - self.projection_size/2:.2f}, top={center_y + self.projection_size/2:.2f}")
                            print(f"Marker 1 (micrograph coords): ({marker1_x:.2f}, {marker1_y:.2f})")
                            print(f"Marker 2 (micrograph coords): ({marker2_x:.2f}, {marker2_y:.2f})")
                            print(f"Marker 1 offset from projection center: ({marker1_x - center_x:.2f}, {marker1_y - center_y:.2f})")
                            print(f"Marker 2 offset from projection center: ({marker2_x - center_x:.2f}, {marker2_y - center_y:.2f})")
                            print(f"Applied offsets: X={x_offset:.2f}, Y={y_offset:.2f} pixels")
                            print(f"Projection size: {self.projection_size} pixels")
                            print(f"Pixel size: {pixel_size:.4f} Angstroms/pixel")
                            print(f"Marker 1 rotated coords: {rotated_marker1}")
                            print(f"Marker 1 rotated coords (Angstroms): X={rotated_marker1[0]:.2f}, Y={rotated_marker1[1]:.2f}, Z={rotated_marker1[2]:.2f}")
                            print(f"Marker 1 using alignment pixel_size (pixels): X={rotated_marker1[1] / pixel_size:.2f}, Y={rotated_marker1[0] / pixel_size:.2f}")
                            print(f"Marker 1 using effective_pixel_size (pixels): X={rotated_marker1[1] / effective_pixel_size:.2f}, Y={rotated_marker1[0] / effective_pixel_size:.2f}")
                            print(f"Marker 1 after offset (pixels): X={marker1_x_pixels:.2f}, Y={marker1_y_pixels:.2f}")
                            print(f"Required offset: X={x_offset:.2f}, Y={y_offset:.2f} pixels")
                            print(f"Offset as % of projection_size: X={x_offset/self.projection_size*100:.1f}%, Y={y_offset/self.projection_size*100:.1f}%")
                            print(f"ANALYSIS: The -70 pixel X offset is {abs(x_offset)/self.projection_size*100:.1f}% of projection size")
                            print(f"  This suggests PyMOL's zoom(complete=1) may scale differently than expected,")
                            print(f"  or there's a coordinate system offset in PyMOL's image rendering.")
                            
                            # Test: Place a marker at the ChimeraX COM to verify coordinate system
                            chimerax_com = np.array([246.62, 222.54, 307.65])
                            centered_com = chimerax_com - structure_center
                            rotated_com = R @ centered_com
                            # Use the CORRECT COM transformation (this is what works for the pink circle)
                            com_x_pixels = rotated_com[1] / pixel_size   # COM uses rotated[1] for X
                            com_y_pixels = -rotated_com[0] / pixel_size  # COM uses -rotated[0] for Y
                            com_x = center_x + com_x_pixels
                            com_y = center_y + com_y_pixels
                            # Draw a big pink circle at the ChimeraX COM position
                            from matplotlib.patches import Circle
                            circle = Circle((com_x, com_y), 15, color='pink', fill=True, 
                                          edgecolor='black', linewidth=2, zorder=27, 
                                          label='ChimeraX COM' if i == 0 else '')
                            self.ax.add_patch(circle)
                            print(f"ChimeraX COM (centered): {centered_com}")
                            print(f"ChimeraX COM (rotated): {rotated_com}")
                            print(f"ChimeraX COM (projected, pixels from center): ({com_x_pixels:.2f}, {com_y_pixels:.2f})")
                            print(f"ChimeraX COM (micrograph coords): ({com_x:.2f}, {com_y:.2f})")
                            print(f"ChimeraX COM should be at projection center: ({center_x:.2f}, {center_y:.2f})")
                            print(f"ChimeraX COM offset from projection center: ({com_x - center_x:.2f}, {com_y - center_y:.2f})")
                            print(f"=== END DEBUG ===\n")
                        
                        # Vector from marker 1 to marker 2 in 2D (projected)
                        vec_2d_x = marker2_x - marker1_x
                        vec_2d_y = marker2_y - marker1_y
                        vec_2d_length = np.sqrt(vec_2d_x**2 + vec_2d_y**2)
                        
                        if vec_2d_length > 1e-6:
                            # Normalize 2D vector
                            vec_2d_x_norm = vec_2d_x / vec_2d_length
                            vec_2d_y_norm = vec_2d_y / vec_2d_length
                            
                            # Arrow length in pixels (user can scale it)
                            arrow_length_pixels = self.custom_arrow_length
                            
                            # Arrow direction (from marker 1 toward marker 2)
                            dx = vec_2d_x_norm * arrow_length_pixels
                            dy = vec_2d_y_norm * arrow_length_pixels
                            
                            # Calculate out-of-plane component for styling
                            # Use the 3D vector's Z component after rotation
                            rotated_vector = R @ self.custom_vector_3d
                            z_component = rotated_vector[2]
                            
                            # Visual indicators for in/out of plane
                            abs_z = abs(z_component)
                            if abs_z < 0.3:
                                linestyle = '-'
                                linewidth = 1.5
                                alpha = 0.6
                            else:
                                linestyle = '--' if z_component > 0 else ':'
                                linewidth = 2.0 + abs_z * 1.0
                                alpha = 0.8 + abs_z * 0.2
                            
                            # Draw arrow from marker 1 position
                            self.ax.arrow(marker1_x, marker1_y, dx, dy,
                                         head_width=arrow_length_pixels*0.3,
                                         head_length=arrow_length_pixels*0.2,
                                         fc=self.custom_arrow_color, ec=self.custom_arrow_color,
                                         alpha=alpha, linewidth=linewidth, linestyle=linestyle, zorder=17)
                            
                            # Add perpendicular indicator for depth
                            if abs_z > 0.1:
                                perp_length = 5.0 * abs_z
                                arrow_angle = np.arctan2(dy, dx)
                                perp_angle = arrow_angle + np.pi/2
                                perp_dx = np.cos(perp_angle) * perp_length
                                perp_dy = np.sin(perp_angle) * perp_length
                                mid_x = marker1_x + dx/2
                                mid_y = marker1_y + dy/2
                                self.ax.plot([mid_x - perp_dx/2, mid_x + perp_dx/2],
                                            [mid_y - perp_dy/2, mid_y + perp_dy/2],
                                            color=self.custom_arrow_color, linewidth=2, alpha=0.9, zorder=18)
                    else:
                        # Original behavior: draw from particle center
                        # CRITICAL: The custom vector is already a direction vector (normalized)
                        # It's defined in the model's coordinate system (centered at origin)
                        # So we can directly rotate and project it
                        rotated_vector = R @ self.custom_vector_3d
                        
                        # Project onto XY plane and convert to pixels
                        dx, dy = project_custom_vector(self.custom_vector_3d, pose, length=self.custom_arrow_length)
                        
                        z_component = rotated_vector[2]  # Z-component: positive = out of plane, negative = into plane
                        
                        # Visual indicators for in/out of plane:
                        # - Line style: solid for in-plane (|z| < 0.3), dashed for out-of-plane
                        # - Line width: thicker for more out-of-plane
                        # - Alpha: brighter for more out-of-plane
                        abs_z = abs(z_component)
                        if abs_z < 0.3:
                            linestyle = '-'  # Solid line for mostly in-plane
                            linewidth = 1.5
                            alpha = 0.6
                        else:
                            linestyle = '--' if z_component > 0 else ':'  # Dashed for out, dotted for in
                            linewidth = 2.0 + abs_z * 1.0  # Thicker for more out-of-plane
                            alpha = 0.8 + abs_z * 0.2  # Brighter for more out-of-plane
                        
                        # Draw arrow with style based on out-of-plane component
                        self.ax.arrow(x_pixel, y_pixel, dx, dy,
                                     head_width=self.custom_arrow_length*0.3,
                                     head_length=self.custom_arrow_length*0.2,
                                     fc=self.custom_arrow_color, ec=self.custom_arrow_color, 
                                     alpha=alpha, linewidth=linewidth, linestyle=linestyle, zorder=17)
                        
                        # Add a small perpendicular indicator line to show depth
                        # Draw a short line perpendicular to the arrow direction
                        if abs_z > 0.1:  # Only show if significantly out-of-plane
                            perp_length = 5.0 * abs_z  # Length proportional to out-of-plane component
                            # Perpendicular direction (rotate arrow direction by 90 degrees)
                            arrow_angle = np.arctan2(dy, dx)
                            perp_angle = arrow_angle + np.pi/2
                            perp_dx = np.cos(perp_angle) * perp_length
                            perp_dy = np.sin(perp_angle) * perp_length
                            # Draw perpendicular line at arrow midpoint
                            mid_x = x_pixel + dx/2
                            mid_y = y_pixel + dy/2
                            self.ax.plot([mid_x - perp_dx/2, mid_x + perp_dx/2],
                                        [mid_y - perp_dy/2, mid_y + perp_dy/2],
                                        color=self.custom_arrow_color, linewidth=2, alpha=0.9, zorder=18)
                    
                except Exception as e:
                    print(f"Error drawing custom arrow {i}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Draw arrows at marker positions if enabled (independent of regular custom arrow)
            if self.show_arrows_at_markers and self.marker_positions is not None and self.custom_vector_3d is not None:
                try:
                    from scipy.spatial.transform import Rotation
                    rot = Rotation.from_euler('ZYZ', pose, degrees=False)
                    R = rot.as_matrix()
                    
                    # CRITICAL: Marker positions are in ChimeraX/PDB absolute coordinates
                    # The structure is centered during projection, so we need to center the markers too
                    # Calculate structure center from PDB data
                    if hasattr(self, 'pdb_data') and self.pdb_data is not None:
                        structure_center = np.mean(self.pdb_data['coords'], axis=0)
                    else:
                        structure_center = np.array([0.0, 0.0, 0.0])
                    
                    # Rotate the custom vector for this particle
                    rotated_vector = R @ self.custom_vector_3d
                    
                    # Use a longer arrow length for marker positions
                    marker_arrow_length = self.custom_arrow_length * 2.0  # 2x longer
                    
                    # Get pixel size for coordinate conversion (use default if not set)
                    pixel_size = self.pixel_size_angstroms if self.pixel_size_angstroms is not None else 1.0
                    
                    # Project each marker position onto the micrograph plane
                    for marker_pos in self.marker_positions:
                        # CRITICAL: Marker positions are in ChimeraX/PDB absolute coordinates (Angstroms)
                        # The structure is centered during projection (see project_pdb_structure: coords - center)
                        # So we MUST center the marker positions the same way
                        centered_marker = marker_pos - structure_center
                        
                        # Rotate centered marker position by particle rotation (same as structure)
                        rotated_marker = R @ centered_marker
                        
                        # Project onto XY plane (drop Z coordinate) and convert from Angstroms to pixels
                        marker_offset_x_pixels = rotated_marker[0] / pixel_size
                        marker_offset_y_pixels = rotated_marker[1] / pixel_size
                        
                        # Marker position on micrograph (offset from particle center)
                        marker_x = x_pixel + marker_offset_x_pixels
                        marker_y = y_pixel + marker_offset_y_pixels
                        
                        # Arrow direction: project the custom vector (already rotated) onto XY plane
                        # marker_arrow_length is in pixels, rotated_vector is unit vector
                        # So we multiply by marker_arrow_length to get pixel offset
                        marker_vec_dx = rotated_vector[0] * marker_arrow_length
                        marker_vec_dy = rotated_vector[1] * marker_arrow_length
                        
                        # Draw arrow from marker position along the custom vector direction
                        self.ax.arrow(marker_x, marker_y, marker_vec_dx, marker_vec_dy,
                                     head_width=marker_arrow_length*0.3,
                                     head_length=marker_arrow_length*0.2,
                                     fc=self.custom_arrow_color, ec=self.custom_arrow_color,
                                     alpha=0.9, linewidth=2.5, linestyle='-', zorder=19)
                except Exception as e:
                    print(f"Error drawing marker position arrow {i}: {e}")
                    import traceback
                    traceback.print_exc()
            
            particles_drawn += 1
        
        print(f"=== Drawn {particles_drawn} particles, {projections_drawn} projections ===\n")
        
        # CRITICAL: After all drawing is complete, explicitly reset axis limits to image dimensions
        # This prevents matplotlib from auto-adjusting limits when projections extend beyond boundaries
        # We do this BEFORE restoring zoom to ensure consistent viewport size
        self.ax.set_xlim(-0.5, img_width - 0.5)
        self.ax.set_ylim(-0.5, img_height - 0.5)
        self.ax.set_aspect('equal', adjustable='box')
        
        # NOW restore zoom limits if they were saved (only if reasonable)
        if restore_zoom and saved_xlim is not None and saved_ylim is not None:
            # Only restore if limits are significantly different from normalized 0-1
            # and within reasonable bounds of the image
            x_range = saved_xlim[1] - saved_xlim[0]
            y_range = saved_ylim[1] - saved_ylim[0]
            # If ranges are > 10 pixels and within image bounds, restore zoom
            if (x_range > 10 and y_range > 10 and 
                saved_xlim[0] >= -100 and saved_xlim[1] <= img_width + 100 and
                saved_ylim[0] >= -100 and saved_ylim[1] <= img_height + 100):
                self.ax.set_xlim(saved_xlim)
                self.ax.set_ylim(saved_ylim)
                self.ax.set_aspect('equal', adjustable='box')
        
        # Draw scale bar AFTER zoom is restored, so it appears in the visible viewport
        if self.show_scale_bar and self.pixel_size_angstroms is not None:
            self._draw_scale_bar(img_width, img_height)
        
        self.canvas.draw()
        
        # Cache the current display after all drawing is complete
        self._cache_current_display()
    
    def _cache_current_display(self):
        """Cache the current display for fast projection toggle."""
        try:
            # Get the rendered image from the canvas
            # Use renderer buffer for better compatibility
            renderer = self.canvas.get_renderer()
            if renderer is None:
                return  # Can't cache without renderer
            
            # Get the buffer as RGB array
            buf = np.asarray(self.canvas.buffer_rgba())
            # Convert RGBA to RGB (drop alpha channel)
            if buf.shape[2] == 4:
                buf = buf[:, :, :3]
            
            # Store in cache based on whether projections are shown
            self.cached_micrograph_idx = self.current_micrograph_idx
            if self.show_projections:
                self.cached_with_projections = buf.copy()
            else:
                self.cached_blank_image = buf.copy()
        except Exception as e:
            # If caching fails, just continue - it's not critical
            pass  # Silently fail - caching is optional for performance
    
    def update_preview(self):
        """Update the projection preview window."""
        if self.current_projection is None:
            return
        
        # Only update if preview window is visible and exists
        if not self.show_projection_preview:
            return
        
        # Ensure window exists, recreate if needed
        if not hasattr(self, 'preview_window') or self.preview_window is None or not self.preview_window.winfo_exists():
            self.setup_projection_preview()
            self.preview_window.deiconify()
        
        self.preview_ax.clear()
        self.preview_ax.imshow(self.current_projection, cmap='hot', origin='lower')
        self.preview_ax.set_title("Current Projection")
        self.preview_ax.axis('off')
        self.preview_canvas.draw()
    
    # Event handlers
    def on_image_select(self, event):
        selection = self.image_listbox.curselection()
        if selection:
            idx = selection[0]
            self.load_micrograph(idx)
    
    def prev_micrograph(self):
        if self.current_micrograph_idx > 0:
            self.load_micrograph(self.current_micrograph_idx - 1)
    
    def next_micrograph(self):
        if self.current_micrograph_idx < len(self.micrograph_paths) - 1:
            self.load_micrograph(self.current_micrograph_idx + 1)
    
    def toggle_projections(self):
        """Toggle projections on/off - regenerate display to avoid coordinate issues."""
        self.show_projections = self.show_projections_var.get()
        # Always regenerate instead of using cache to avoid coordinate mismatches
        # The fast rendering makes this acceptable
        self.update_display(use_cache=False)
    
    def toggle_orientations(self):
        """Toggle orientations on/off while preserving zoom."""
        # Save current zoom state
        current_xlim = None
        current_ylim = None
        if hasattr(self, 'ax') and self.ax is not None:
            try:
                current_xlim = list(self.ax.get_xlim())
                current_ylim = list(self.ax.get_ylim())
            except:
                pass
        
        self.show_orientations = self.show_orientations_var.get()
        self.update_display()
        
        # Restore zoom state if we had one
        if current_xlim is not None and current_ylim is not None and self.original_micrograph is not None:
            img_height, img_width = self.original_micrograph.shape[:2]
            tol = 50
            was_zoomed = (abs(current_xlim[0]) > tol or 
                         abs(current_xlim[1] - img_width) > tol or
                         abs(current_ylim[0]) > tol or
                         abs(current_ylim[1] - img_height) > tol)
            if was_zoomed:
                self.ax.set_xlim(current_xlim)
                self.ax.set_ylim(current_ylim)
                self.ax.set_aspect('equal', adjustable='box')
                self.canvas.draw()
    
    def toggle_preview(self):
        self.show_projection_preview = self.show_preview_var.get()
        if self.show_projection_preview:
            # Check if window exists, recreate if it was destroyed
            if not hasattr(self, 'preview_window') or self.preview_window is None or not self.preview_window.winfo_exists():
                self.setup_projection_preview()
            self.preview_window.deiconify()
            if self.current_projection is not None:
                self.update_preview()
        else:
            # Hide window if it exists, but don't destroy it
            if hasattr(self, 'preview_window') and self.preview_window is not None and self.preview_window.winfo_exists():
                self.preview_window.withdraw()
    
    def toggle_outlines(self):
        """Toggle outlines on/off while preserving zoom."""
        # Save current zoom state
        current_xlim = None
        current_ylim = None
        if hasattr(self, 'ax') and self.ax is not None:
            try:
                current_xlim = list(self.ax.get_xlim())
                current_ylim = list(self.ax.get_ylim())
            except:
                pass
        
        self.show_outlines = self.show_outlines_var.get()
        self.update_display()
        
        # Restore zoom state if we had one
        if current_xlim is not None and current_ylim is not None:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
            self.ax.set_aspect('equal', adjustable='box')
            self.canvas.draw()
    
    def generate_remaining_projections(self):
        """Generate projections for remaining particles."""
        if self.current_micrograph_idx is None or self.current_particles is None:
            return
        
        num_particles = len(self.current_particles['poses'])
        with self.cache_lock:
            num_cached = len([k for k in self.projection_cache.keys() if k[0] == self.current_micrograph_idx])
        
        if num_cached >= num_particles:
            messagebox.showinfo("Info", "All projections have already been generated.")
            # Set flag and disable button
            self.all_projections_generated = True
            if hasattr(self, 'generate_all_button'):
                self.generate_all_button.config(state=tk.DISABLED)
            return
        
        # Check if already generating to prevent multiple simultaneous generations
        if hasattr(self, '_generating_projections') and self._generating_projections:
            print("Projection generation already in progress, skipping...")
            return
        
        # Set flag to prevent multiple generations
        self._generating_projections = True
        
        # Generate all remaining projections
        if self.current_micrograph_idx < len(self.micrograph_files):
            mg_file = self.micrograph_files[self.current_micrograph_idx]
            base_name = mg_file.stem
        else:
            base_name = "proj_v1"
        
        self.status_var.set(f"Generating remaining {num_particles - num_cached} projections...")
        self.root.update()
        
        # Generate in foreground (not background) to ensure completion and display update
        # When user explicitly clicks "Apply" or "Generate All", we want it to complete
        self._generate_all_projections(self.current_micrograph_idx, base_filename=base_name, limit=None, background=False)
        
        # Reset flag after a delay to allow generation to complete
        def reset_generating_flag():
            self._generating_projections = False
            # Check if all projections are now generated
            with self.cache_lock:
                num_cached_after = len([k for k in self.projection_cache.keys() if k[0] == self.current_micrograph_idx])
            if num_cached_after >= num_particles:
                self.all_projections_generated = True
                if hasattr(self, 'generate_all_button'):
                    self.generate_all_button.config(state=tk.DISABLED)
                self.status_var.set(f"Generated all {num_particles} projections.")
        
        # Wait a bit longer for background generation to complete
        self.root.after(2000, reset_generating_flag)
    
    def update_alpha(self, val):
        self._reset_idle_timer()  # Reset idle timer on user interaction
        self.projection_alpha = val
        self.alpha_label.config(text=f"{val:.2f}")
        self.update_display()
    
    def update_projection_offset(self, axis, value):
        """Update projection offset from slider."""
        if axis == 'x':
            self.projection_offset_x = value
            if hasattr(self, 'offset_x_label'):
                self.offset_x_label.config(text=f"{value:.1f} px")
        elif axis == 'y':
            self.projection_offset_y = value
            if hasattr(self, 'offset_y_label'):
                self.offset_y_label.config(text=f"{value:.1f} px")
        # Update display immediately
        self.update_display()
    
    def update_rotation_correction(self, axis, value):
        """Update rotation correction angle from slider (without regenerating)."""
        if axis == 'x':
            self.rotation_correction_x = value
            if hasattr(self, 'rot_x_label'):
                self.rot_x_label.config(text=f"{value:.1f}°")
        elif axis == 'y':
            self.rotation_correction_y = value
            if hasattr(self, 'rot_y_label'):
                self.rot_y_label.config(text=f"{value:.1f}°")
        elif axis == 'z':
            self.rotation_correction_z = value
            if hasattr(self, 'rot_z_label'):
                self.rot_z_label.config(text=f"{value:.1f}°")
        # Don't update display immediately - user clicks "Apply" button to regenerate
    
    def apply_rotation_correction_and_regenerate(self):
        """Apply rotation corrections and regenerate projections."""
        # Update angles from sliders
        if hasattr(self, 'rotation_correction_x_var'):
            self.rotation_correction_x = self.rotation_correction_x_var.get()
        if hasattr(self, 'rotation_correction_y_var'):
            self.rotation_correction_y = self.rotation_correction_y_var.get()
        if hasattr(self, 'rotation_correction_z_var'):
            self.rotation_correction_z = self.rotation_correction_z_var.get()
        
        print(f"Applying rotation corrections: X={self.rotation_correction_x:.1f}°, Y={self.rotation_correction_y:.1f}°, Z={self.rotation_correction_z:.1f}°")
        
        # Reset generation flag to allow new generation
        self._generating_projections = False
        
        # Clear cache for current micrograph
        if self.current_micrograph_idx is not None:
            with self.cache_lock:
                keys_to_remove = [k for k in self.projection_cache.keys() if k[0] == self.current_micrograph_idx]
                for key in keys_to_remove:
                    del self.projection_cache[key]
                print(f"Cleared {len(keys_to_remove)} cached projections")
        
        # Reset the all_projections_generated flag so Generate All button is enabled
        self.all_projections_generated = False
        if hasattr(self, 'generate_all_button'):
            self.generate_all_button.config(state=tk.NORMAL)
        
        # Regenerate projections for current micrograph
        if self.current_micrograph_idx is not None and self.current_particles is not None:
            # Generate all projections for current micrograph in foreground
            self.generate_remaining_projections()
        else:
            # Just update display if no micrograph loaded
            self.update_display(use_cache=False)
    
    def update_size(self, val):
        """Update projection size - FAST: just resize cached arrays, don't regenerate from PDB."""
        old_size = self.projection_size
        self.projection_size = int(val)
        self.size_label.config(text=f"{self.projection_size} px")
        
        # FAST: Just resize cached projections, don't regenerate from ChimeraX
        if old_size != self.projection_size and self.current_micrograph_idx is not None:
            # Resize all cached projections for current micrograph using high-quality resampling
            from PIL import Image
            resized_count = 0
            # Thread-safe cache access - get list of keys first
            with self.cache_lock:
                cache_keys_to_resize = [k for k in self.projection_cache.keys() 
                                       if k[0] == self.current_micrograph_idx]
            
            # Process each projection (resize outside lock to avoid blocking)
            for cache_key in cache_keys_to_resize:
                # Get projection copy (thread-safe)
                with self.cache_lock:
                    if cache_key not in self.projection_cache:
                        continue  # Skip if removed by another thread
                    projection = self.projection_cache[cache_key].copy()
                
                if len(projection.shape) == 3 and projection.shape[2] == 4:  # RGBA
                    # Convert to 0-255 range for PIL
                    rgba_uint8 = (projection * 255.0).astype(np.uint8)
                    # Create PIL Image
                    pil_img = Image.fromarray(rgba_uint8)
                    # Resize using LANCZOS (best quality for preserving detail)
                    pil_img_resized = pil_img.resize((self.projection_size, self.projection_size), 
                                                     Image.Resampling.LANCZOS)
                    # Convert back to numpy array and normalize
                    resized = np.array(pil_img_resized).astype(np.float32) / 255.0
                    # When resizing, we need to reapply the alpha setting
                    # First, determine structure mask from resized alpha
                    resized_structure_mask = resized[:,:,3] > 0.01
                    # Apply current alpha setting (force to 1.0 if >= 0.99)
                    if self.projection_alpha >= 0.99:
                        resized[:,:,3] = np.where(resized_structure_mask, 1.0, 0.0)
                    else:
                        resized[:,:,3] = np.where(resized_structure_mask, float(self.projection_alpha), 0.0)
                    # Ensure alpha stays in valid range
                    resized[:,:,3] = np.clip(resized[:,:,3], 0, 1)
                    # Update cache (thread-safe)
                    with self.cache_lock:
                        if cache_key in self.projection_cache:  # Check again in case removed
                            self.projection_cache[cache_key] = resized
                    resized_count += 1
            
            if resized_count > 0:
                print(f"Fast-resized {resized_count} cached projections to {self.projection_size}x{self.projection_size}")
        
        self.update_display()
    
    def update_arrow(self, val):
        self.arrow_length = val
        self.arrow_label.config(text=f"{val} px")
        self.update_display()
    
    def toggle_scale_bar(self):
        """Toggle scale bar display."""
        self.show_scale_bar = self.show_scale_bar_var.get()
        self.update_display(use_cache=False)
    
    def update_scale_bar(self, val):
        """Update scale bar length."""
        self.scale_bar_length_angstroms = val
        self.scale_bar_label.config(text=f"{val:.0f} Å")
        if self.show_scale_bar:
            self.update_display(use_cache=False)
    
    def update_scale_bar_width(self, val):
        """Update scale bar line width."""
        self.scale_bar_linewidth = val
        self.scale_bar_width_label.config(text=f"{val:.1f} px")
        if self.show_scale_bar:
            self.update_display(use_cache=False)
    
    def on_scale_bar_color_change(self, event=None):
        """Handle manual entry of scale bar color."""
        color = self.scale_bar_color_entry.get().strip()
        # Validate color (basic check)
        if color and (color.startswith('#') and len(color) == 7 or color in ['white', 'black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta']):
            self.scale_bar_color = color
            if self.show_scale_bar:
                self.update_display(use_cache=False)
        else:
            # Invalid color, restore previous
            self.scale_bar_color_entry.delete(0, tk.END)
            self.scale_bar_color_entry.insert(0, self.scale_bar_color)
    
    def pick_scale_bar_color(self):
        """Open color picker for scale bar color."""
        from tkinter import colorchooser
        color = colorchooser.askcolor(title="Pick Scale Bar Color", color=self.scale_bar_color)[1]
        if color:
            self.scale_bar_color = color
            self.scale_bar_color_entry.delete(0, tk.END)
            self.scale_bar_color_entry.insert(0, color)
            if self.show_scale_bar:
                self.update_display(use_cache=False)
    
    def jump_to_micrograph(self, event=None):
        """Jump to a specific micrograph by index."""
        try:
            idx_str = self.micrograph_index_entry.get().strip()
            if not idx_str:
                return
            idx = int(idx_str) - 1  # Convert from 1-based to 0-based
            if 0 <= idx < len(self.micrograph_paths):
                self.load_micrograph(idx)
            else:
                self.status_var.set(f"Invalid micrograph index. Must be between 1 and {len(self.micrograph_paths)}")
        except ValueError:
            self.status_var.set("Invalid micrograph index. Please enter a number.")
    
    def _auto_load_vector_file(self):
        """Auto-load vector and marker positions from vector_val.txt if present in current directory."""
        try:
            vector_file = Path.cwd() / 'vector_val.txt'
            if vector_file.exists():
                with open(vector_file, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
                    if len(lines) >= 3:
                        # First 3 lines are the vector
                        x = float(lines[0])
                        y = float(lines[1])
                        z = float(lines[2])
                        vec = np.array([x, y, z], dtype=float)
                        norm = np.linalg.norm(vec)
                        if norm > 1e-10:
                            self.custom_vector_3d = vec / norm
                            # Set method to from_markers if marker positions are available
                            if len(lines) >= 9:
                                self.custom_vector_method = 'from_markers'
                                # Update method dropdown if it exists
                                if hasattr(self, 'custom_vector_method_var'):
                                    self.custom_vector_method_var.set('from_markers')
                                    # Trigger method change handler to show marker frame
                                    self.on_custom_vector_method_changed()
                            else:
                                self.custom_vector_method = 'user_defined'
                            
                            # Try to load marker positions if available (lines 4-9)
                            if len(lines) >= 9:
                                try:
                                    m1_x = float(lines[3])
                                    m1_y = float(lines[4])
                                    m1_z = float(lines[5])
                                    m2_x = float(lines[6])
                                    m2_y = float(lines[7])
                                    m2_z = float(lines[8])
                                    point1 = np.array([m1_x, m1_y, m1_z])
                                    point2 = np.array([m2_x, m2_y, m2_z])
                                    self.marker_positions = [point1, point2]
                                    
                                    # Calculate distance between markers
                                    self.marker_distance_angstroms = np.linalg.norm(point2 - point1)
                                    
                                    # Update marker entry fields if they exist
                                    if hasattr(self, 'marker1_x_entry'):
                                        self.marker1_x_entry.delete(0, tk.END)
                                        self.marker1_x_entry.insert(0, f"{m1_x:.1f}")
                                        self.marker1_y_entry.delete(0, tk.END)
                                        self.marker1_y_entry.insert(0, f"{m1_y:.1f}")
                                        self.marker1_z_entry.delete(0, tk.END)
                                        self.marker1_z_entry.insert(0, f"{m1_z:.1f}")
                                        self.marker2_x_entry.delete(0, tk.END)
                                        self.marker2_x_entry.insert(0, f"{m2_x:.1f}")
                                        self.marker2_y_entry.delete(0, tk.END)
                                        self.marker2_y_entry.insert(0, f"{m2_y:.1f}")
                                        self.marker2_z_entry.delete(0, tk.END)
                                        self.marker2_z_entry.insert(0, f"{m2_z:.1f}")
                                    
                                    # Set default arrow length to distance between markers (in pixels)
                                    if self.pixel_size_angstroms is not None and self.pixel_size_angstroms > 0:
                                        default_length_pixels = self.marker_distance_angstroms / self.pixel_size_angstroms
                                        self.custom_arrow_length = int(default_length_pixels)
                                        # Update slider if it exists
                                        if hasattr(self, 'custom_arrow_var'):
                                            self.custom_arrow_var.set(self.custom_arrow_length)
                                        if hasattr(self, 'custom_arrow_label'):
                                            self.custom_arrow_label.config(text=f"{self.custom_arrow_length} px")
                                    
                                    print(f"Auto-loaded vector and markers from {vector_file}")
                                except (ValueError, IndexError) as e:
                                    print(f"Auto-loaded vector from {vector_file} (marker positions not found or invalid: {e})")
                            
                            # Update UI fields
                            if hasattr(self, 'custom_vector_x_entry'):
                                self.custom_vector_x_entry.delete(0, tk.END)
                                self.custom_vector_x_entry.insert(0, f"{self.custom_vector_3d[0]:.6f}")
                                self.custom_vector_y_entry.delete(0, tk.END)
                                self.custom_vector_y_entry.insert(0, f"{self.custom_vector_3d[1]:.6f}")
                                self.custom_vector_z_entry.delete(0, tk.END)
                                self.custom_vector_z_entry.insert(0, f"{self.custom_vector_3d[2]:.6f}")
                            print(f"Auto-loaded vector: [{self.custom_vector_3d[0]:.6f}, {self.custom_vector_3d[1]:.6f}, {self.custom_vector_3d[2]:.6f}]")
        except Exception as e:
            print(f"Could not auto-load vector from vector_val.txt: {e}")
    
    def toggle_custom_arrow(self):
        """Toggle custom structural vector arrow display."""
        self.show_custom_arrow = self.show_custom_arrow_var.get()
        if self.show_custom_arrow and self.custom_vector_3d is None:
            # Try to initialize with default user-defined vector if not set
            try:
                x = float(self.custom_vector_x_entry.get())
                y = float(self.custom_vector_y_entry.get())
                z = float(self.custom_vector_z_entry.get())
                self.custom_vector_3d = np.array([x, y, z], dtype=float)
                norm = np.linalg.norm(self.custom_vector_3d)
                if norm > 1e-10:
                    self.custom_vector_3d = self.custom_vector_3d / norm
                else:
                    messagebox.showwarning("Warning", "Custom vector has zero length. Please set a valid vector.")
                    self.show_custom_arrow = False
                    self.show_custom_arrow_var.set(False)
                    return
            except ValueError:
                messagebox.showwarning("Warning", "Invalid vector values. Please enter numeric values.")
                self.show_custom_arrow = False
                self.show_custom_arrow_var.set(False)
                return
        self.update_display(use_cache=False)
    
    def on_custom_vector_method_changed(self, event=None):
        """Handle change in custom vector calculation method."""
        method = self.custom_vector_method_var.get()
        self.custom_vector_method = method
        
        # Show/hide appropriate input frames
        if method == 'user_defined':
            self.custom_vector_frame.pack(fill=tk.X, pady=5)
            self.custom_markers_frame.pack_forget()
            self.custom_chain_frame.pack_forget()
        elif method == 'from_markers':
            self.custom_vector_frame.pack_forget()
            self.custom_markers_frame.pack(fill=tk.X, pady=5)
            self.custom_chain_frame.pack_forget()
        elif method in ['chain_com', 'chain_axis']:
            self.custom_vector_frame.pack_forget()
            self.custom_markers_frame.pack_forget()
            self.custom_chain_frame.pack(fill=tk.X, pady=5)
        elif method == 'atom_selection':
            # For now, just show chain frame (can be extended later)
            self.custom_vector_frame.pack_forget()
            self.custom_markers_frame.pack_forget()
            self.custom_chain_frame.pack(fill=tk.X, pady=5)
    
    def update_custom_vector(self):
        """Update custom vector from user-defined input."""
        try:
            x = float(self.custom_vector_x_entry.get())
            y = float(self.custom_vector_y_entry.get())
            z = float(self.custom_vector_z_entry.get())
            vec = np.array([x, y, z], dtype=float)
            norm = np.linalg.norm(vec)
            if norm < 1e-10:
                messagebox.showerror("Error", "Vector has zero length")
                return
            self.custom_vector_3d = vec / norm
            self.custom_vector_method = 'user_defined'
            messagebox.showinfo("Success", f"Custom vector updated: [{x:.3f}, {y:.3f}, {z:.3f}] (normalized)")
            if self.show_custom_arrow:
                self.update_display(use_cache=False)
        except ValueError:
            messagebox.showerror("Error", "Invalid vector values. Please enter numeric values.")
    
    def update_marker_offset(self, event=None):
        """Update the X and Y offsets for marker positions from the GUI input."""
        try:
            if hasattr(self, 'marker_x_offset_entry'):
                self.marker_x_offset = float(self.marker_x_offset_entry.get().strip())
            if hasattr(self, 'marker_y_offset_entry'):
                self.marker_y_offset = float(self.marker_y_offset_entry.get().strip())
            print(f"DEBUG: Updated marker offsets - X: {self.marker_x_offset}, Y: {self.marker_y_offset}")
            # Force redraw to update marker positions
            if hasattr(self, 'current_micrograph_idx') and self.current_micrograph_idx is not None:
                # Just call update_display - it will redraw everything
                self.update_display(use_cache=True)
        except (ValueError, AttributeError) as e:
            print(f"DEBUG: Error updating marker offsets: {e}")
            import traceback
            traceback.print_exc()
            self.marker_x_offset = 0.0
            self.marker_y_offset = 0.0
    
    def update_custom_vector_from_markers(self):
        """Update custom vector from two ChimeraX marker positions."""
        try:
            # Get marker positions
            m1_x = float(self.marker1_x_entry.get())
            m1_y = float(self.marker1_y_entry.get())
            m1_z = float(self.marker1_z_entry.get())
            m2_x = float(self.marker2_x_entry.get())
            m2_y = float(self.marker2_y_entry.get())
            m2_z = float(self.marker2_z_entry.get())
            
            point1 = np.array([m1_x, m1_y, m1_z])
            point2 = np.array([m2_x, m2_y, m2_z])
            
            # Store marker positions for drawing arrows at marker locations
            self.marker_positions = [point1, point2]
            
            # Calculate distance between markers (for default arrow length)
            self.marker_distance_angstroms = np.linalg.norm(point2 - point1)
            
            # Calculate vector from marker 1 to marker 2 (normalized)
            self.custom_vector_3d = calculate_vector_from_two_points(point1, point2)
            self.custom_vector_method = 'from_markers'
            
            # Set default arrow length to distance between markers (in pixels)
            if self.pixel_size_angstroms is not None and self.pixel_size_angstroms > 0:
                default_length_pixels = self.marker_distance_angstroms / self.pixel_size_angstroms
                self.custom_arrow_length = int(default_length_pixels)
                # Update slider if it exists
                if hasattr(self, 'custom_arrow_var'):
                    self.custom_arrow_var.set(self.custom_arrow_length)
                if hasattr(self, 'custom_arrow_label'):
                    self.custom_arrow_label.config(text=f"{self.custom_arrow_length} px")
            
            # Print vector value
            vector_str = f"[{self.custom_vector_3d[0]:.6f}, {self.custom_vector_3d[1]:.6f}, {self.custom_vector_3d[2]:.6f}]"
            print(f"Custom vector from markers: {vector_str}")
            print(f"  Marker 1: ({m1_x:.1f}, {m1_y:.1f}, {m1_z:.1f})")
            print(f"  Marker 2: ({m2_x:.1f}, {m2_y:.1f}, {m2_z:.1f})")
            
            # Save to vector_val.txt file with marker positions
            try:
                vector_file = Path.cwd() / 'vector_val.txt'
                with open(vector_file, 'w') as f:
                    f.write(f"# Custom vector calculated from ChimeraX markers\n")
                    f.write(f"# Marker 1 (x, y, z): {m1_x:.6f}, {m1_y:.6f}, {m1_z:.6f}\n")
                    f.write(f"# Marker 2 (x, y, z): {m2_x:.6f}, {m2_y:.6f}, {m2_z:.6f}\n")
                    f.write(f"# Vector (normalized): {vector_str}\n")
                    f.write(f"# Format: vector_x, vector_y, vector_z (one per line)\n")
                    f.write(f"{self.custom_vector_3d[0]:.10f}\n")
                    f.write(f"{self.custom_vector_3d[1]:.10f}\n")
                    f.write(f"{self.custom_vector_3d[2]:.10f}\n")
                    f.write(f"# Marker positions (for reference):\n")
                    f.write(f"# marker1_x, marker1_y, marker1_z\n")
                    f.write(f"{m1_x:.10f}\n")
                    f.write(f"{m1_y:.10f}\n")
                    f.write(f"{m1_z:.10f}\n")
                    f.write(f"# marker2_x, marker2_y, marker2_z\n")
                    f.write(f"{m2_x:.10f}\n")
                    f.write(f"{m2_y:.10f}\n")
                    f.write(f"{m2_z:.10f}\n")
                print(f"Saved vector and marker positions to {vector_file}")
            except Exception as e:
                print(f"Warning: Could not save vector to file: {e}")
            
            messagebox.showinfo("Success", 
                              f"Vector calculated from markers:\n"
                              f"Marker 1: ({m1_x:.1f}, {m1_y:.1f}, {m1_z:.1f})\n"
                              f"Marker 2: ({m2_x:.1f}, {m2_y:.1f}, {m2_z:.1f})\n"
                              f"Vector: {vector_str}\n\n"
                              f"Saved to vector_val.txt")
            
            # Also update the user_defined fields for reference
            self.custom_vector_x_entry.delete(0, tk.END)
            self.custom_vector_x_entry.insert(0, f"{self.custom_vector_3d[0]:.6f}")
            self.custom_vector_y_entry.delete(0, tk.END)
            self.custom_vector_y_entry.insert(0, f"{self.custom_vector_3d[1]:.6f}")
            self.custom_vector_z_entry.delete(0, tk.END)
            self.custom_vector_z_entry.insert(0, f"{self.custom_vector_3d[2]:.6f}")
            
            if self.show_custom_arrow:
                self.update_display(use_cache=False)
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid marker positions: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate vector: {str(e)}")
    
    def update_custom_vector_from_chains(self):
        """Update custom vector from chain selection."""
        if not hasattr(self, 'pdb_data') or self.pdb_data is None:
            messagebox.showerror("Error", "No PDB structure loaded. Please load a structure first.")
            return
        
        method = self.custom_vector_method_var.get()
        if method not in ['chain_com', 'chain_axis', 'atom_selection']:
            messagebox.showerror("Error", "Chain-based methods require 'chain_com' or 'chain_axis' method")
            return
        
        try:
            chain_str = self.custom_chain_entry.get().strip()
            if not chain_str:
                messagebox.showerror("Error", "Please enter chain IDs (comma-separated)")
                return
            
            chain_ids = [c.strip() for c in chain_str.split(',')]
            
            if method == 'chain_com' and len(chain_ids) < 2:
                messagebox.showerror("Error", "chain_com method requires at least 2 chain IDs")
                return
            
            # Calculate vector
            if method == 'chain_com':
                self.custom_vector_3d = calculate_custom_vector_from_pdb(
                    self.pdb_data,
                    method='chain_com',
                    chain_ids=chain_ids,
                    from_center_to='first_to_second'
                )
            elif method == 'chain_axis':
                self.custom_vector_3d = calculate_custom_vector_from_pdb(
                    self.pdb_data,
                    method='chain_axis',
                    chain_ids=chain_ids[:1]  # Use first chain only
                )
            elif method == 'atom_selection':
                # For atom selection, we'd need more parameters - for now, use chain axis
                self.custom_vector_3d = calculate_custom_vector_from_pdb(
                    self.pdb_data,
                    method='chain_axis',
                    chain_ids=chain_ids[:1]
                )
            
            self.custom_vector_method = method
            self.custom_vector_params = {'chain_ids': chain_ids}
            messagebox.showinfo("Success", f"Custom vector calculated from chains: {chain_ids}\nVector: [{self.custom_vector_3d[0]:.3f}, {self.custom_vector_3d[1]:.3f}, {self.custom_vector_3d[2]:.3f}]")
            if self.show_custom_arrow:
                self.update_display(use_cache=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate vector: {str(e)}")
    
    def update_custom_arrow_length(self, val):
        """Update custom arrow length."""
        self.custom_arrow_length = val
        self.custom_arrow_label.config(text=f"{val} px")
        if self.show_custom_arrow:
            self.update_display(use_cache=False)
    
    def _toggle_fine_tuning(self, parent_frame):
        """Toggle visibility of fine-tuning controls."""
        if self.fine_tuning_expanded.get():
            # Show controls
            self.fine_tuning_offset_frame.pack(fill=tk.X, pady=5)
        else:
            # Hide controls
            self.fine_tuning_offset_frame.pack_forget()
    
    def _toggle_auto_correct_com(self):
        """Toggle automatic COM offset correction."""
        self.auto_correct_com_offset = self.auto_correct_com_offset_var.get()
        if self.auto_correct_com_offset:
            print("Auto-correct COM offset enabled - will calculate and apply offset for each particle")
        self.update_display(use_cache=False)
    
    def toggle_arrows_at_markers(self):
        """Toggle showing arrows at marker positions."""
        self.show_arrows_at_markers = self.show_arrows_at_markers_var.get()
        if self.show_arrows_at_markers and (self.marker_positions is None or self.custom_vector_3d is None):
            messagebox.showwarning("Warning", "No marker positions available. Please calculate vector from markers first.")
            self.show_arrows_at_markers = False
            self.show_arrows_at_markers_var.set(False)
            return
        self.update_display(use_cache=False)
    
    def pick_custom_arrow_color(self):
        """Open color picker for custom arrow color."""
        from tkinter import colorchooser
        color = colorchooser.askcolor(title="Pick Custom Arrow Color", color=self.custom_arrow_color)
        if color[1]:  # color[1] is the hex string
            self.custom_arrow_color = color[1]
            self.custom_arrow_color_entry.delete(0, tk.END)
            self.custom_arrow_color_entry.insert(0, '#FFFFFF')  # Default white
            if self.show_custom_arrow:
                self.update_display(use_cache=False)
    
    def _draw_scale_bar(self, img_width, img_height):
        """Draw a scale bar on the micrograph display.
        
        The scale bar is positioned in the bottom-left corner of the image,
        4% away from the left and bottom edges, regardless of zoom level.
        
        Args:
            img_width: Width of the full image in pixels
            img_height: Height of the full image in pixels
        """
        if self.pixel_size_angstroms is None or self.pixel_size_angstroms <= 0:
            return
        
        # Calculate scale bar length in pixels
        scale_bar_length_pixels = self.scale_bar_length_angstroms / self.pixel_size_angstroms
        
        # Clamp scale bar width to maximum of 25% of image height
        max_width = img_height * 0.25
        scale_bar_width = min(self.scale_bar_linewidth, max_width)
        
        # Update scale bar width slider max if needed
        if hasattr(self, 'scale_bar_width_scale'):
            current_max = self.scale_bar_width_scale.cget('to')
            if current_max < max_width:
                self.scale_bar_width_scale.config(to=max_width)
        
        # Position scale bar in bottom-left corner, 3-5% from edges
        margin_x = img_width * 0.04  # 4% from left edge
        margin_y = img_height * 0.04  # 4% from bottom edge
        
        bar_x_start = margin_x
        bar_x_end = bar_x_start + scale_bar_length_pixels
        
        # Ensure scale bar doesn't exceed image bounds
        if bar_x_end > img_width - margin_x:
            bar_x_end = img_width - margin_x
            bar_x_start = bar_x_end - scale_bar_length_pixels
            if bar_x_start < margin_x:
                bar_x_start = margin_x
                # Adjust length if needed
                scale_bar_length_pixels = bar_x_end - bar_x_start
        
        # Y position: 4% from bottom
        bar_y = margin_y
        
        # Draw the scale bar with user-configurable width and color
        from matplotlib.patches import Rectangle
        scale_bar_rect = Rectangle(
            (bar_x_start, bar_y), 
            scale_bar_length_pixels, 
            scale_bar_width,
            facecolor=self.scale_bar_color, 
            edgecolor=self.scale_bar_color,
            linewidth=0,
            zorder=20
        )
        self.ax.add_patch(scale_bar_rect)
        
        # Add text label above the scale bar (plain text, no box)
        label_text = f"{self.scale_bar_length_angstroms:.0f} Å"
        text_y = bar_y + scale_bar_width + 8
        # Ensure text is within image bounds
        if text_y < img_height:
            self.ax.text(
                (bar_x_start + bar_x_end) / 2,  # Center of bar
                text_y,  # Above the bar
                label_text,
                color=self.scale_bar_color,
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='bottom',
                zorder=21
            )
    
    def update_lowpass(self, val):
        self.lowpass_A = val
        self.lowpass_label.config(text=f"{val:.1f} Å")
        if hasattr(self, 'lowpass_entry'):
            self.lowpass_entry.delete(0, tk.END)
            self.lowpass_entry.insert(0, f"{val:.1f}")
        self.update_display()
    
    def on_lowpass_entry_change(self, event=None):
        """Handle manual entry of low-pass filter value."""
        try:
            val = float(self.lowpass_entry.get().strip())
            val = max(0.0, min(50.0, val))  # Clamp to 0-50
            self.lowpass_A = val
            self.lowpass_var.set(val)
            self.lowpass_label.config(text=f"{val:.1f} Å")
            self.lowpass_entry.delete(0, tk.END)
            self.lowpass_entry.insert(0, f"{val:.1f}")
            self.update_display()
        except ValueError:
            # Invalid input, restore previous value
            self.lowpass_entry.delete(0, tk.END)
            self.lowpass_entry.insert(0, f"{self.lowpass_A:.1f}")
    
    def update_brightness(self, val):
        self.brightness = val
        self.brightness_label.config(text=f"{val:.2f}")
        self.update_display()
    
    def update_contrast(self, val):
        self.contrast = val
        self.contrast_label.config(text=f"{val:.2f}")
        self.update_display()
    
    def reset_enhancements(self):
        self.lowpass_A = 2.0
        self.brightness = 0.0
        self.contrast = 1.0
        self.lowpass_var.set(self.lowpass_A)
        self.brightness_var.set(self.brightness)
        self.contrast_var.set(self.contrast)
        self.lowpass_label.config(text=f"{self.lowpass_A:.1f} Å")
        if hasattr(self, 'lowpass_entry'):
            self.lowpass_entry.delete(0, tk.END)
            self.lowpass_entry.insert(0, f"{self.lowpass_A:.1f}")
        self.brightness_label.config(text=f"{self.brightness:.2f}")
        self.contrast_label.config(text=f"{self.contrast:.2f}")
        self.update_display()
    
    def pick_protein_color(self):
        """Open color picker for default protein color."""
        import tkinter.colorchooser
        color = tkinter.colorchooser.askcolor(title="Pick Protein Color", 
                                             color=self.default_protein_color)[1]
        if color:
            self.default_protein_color = color
            self.protein_color_entry.delete(0, tk.END)
            self.protein_color_entry.insert(0, color)
    
    def pick_nucleic_color(self):
        """Open color picker for default nucleic acid color."""
        import tkinter.colorchooser
        color = tkinter.colorchooser.askcolor(title="Pick Nucleic Acid Color", 
                                             color=self.default_nucleic_color)[1]
        if color:
            self.default_nucleic_color = color
            self.nucleic_color_entry.delete(0, tk.END)
            self.nucleic_color_entry.insert(0, color)
    
    def on_color_mode_changed(self):
        """Handle color mode change (bio_material vs chain)."""
        self.color_mode = self.color_mode_var.get()
        
        if self.color_mode == 'bio_material':
            # Show bio material frame, hide chain colors frame
            self.bio_material_frame.pack(fill=tk.X, pady=5)
            self.chain_colors_frame.pack_forget()
        else:  # chain mode
            # Hide bio material frame, show chain colors frame
            self.bio_material_frame.pack_forget()
            self.chain_colors_frame.pack(fill=tk.X, pady=5)
            # Update chain color UI if chains are available
            if self.available_chains:
                self._update_chain_color_ui()
    
    def _update_chain_color_ui(self):
        """Update the chain color input UI based on available chains."""
        # Clear existing widgets
        for widget in self.chain_colors_frame.winfo_children():
            widget.destroy()
        self.chain_color_entries = {}
        self.chain_color_pick_buttons = {}
        
        if not self.available_chains:
            ttk.Label(self.chain_colors_frame, text="No chains found in structure.").pack(anchor=tk.W)
            return
        
        # Determine default colors for each chain based on bio_material
        # We need to check if each chain is protein or nucleic
        chain_types = {}
        if self.pdb_data and 'chain_ids' in self.pdb_data and 'residue_names' in self.pdb_data:
            from particle_mapper import is_nucleic_acid
            # Convert chain_ids to strings for proper comparison
            chain_ids_str = np.array([str(cid).strip() for cid in self.pdb_data['chain_ids']])
            for chain_id in self.available_chains:
                # Find residues in this chain - compare as strings
                chain_id_clean = str(chain_id).strip()
                chain_mask = chain_ids_str == chain_id_clean
                if chain_mask.any():
                    # Check first residue to determine type (most chains are uniform)
                    first_resname = self.pdb_data['residue_names'][chain_mask][0]
                    chain_types[chain_id] = 'nucleic' if is_nucleic_acid(first_resname) else 'protein'
        
        # Create color input for each chain
        for chain_id in self.available_chains:
            chain_frame = ttk.Frame(self.chain_colors_frame)
            chain_frame.pack(fill=tk.X, pady=2)
            
            # Determine default color for this chain
            if chain_id in chain_types:
                default_color = self.default_nucleic_color if chain_types[chain_id] == 'nucleic' else self.default_protein_color
            else:
                default_color = self.default_protein_color  # Default to protein color
            
            # Initialize chain color in map if not present
            if chain_id not in self.chain_color_map:
                self.chain_color_map[chain_id] = default_color
            
            ttk.Label(chain_frame, text=f"Chain {chain_id}:").pack(side=tk.LEFT, padx=(0, 5))
            entry = ttk.Entry(chain_frame, width=15)
            entry.insert(0, self.chain_color_map[chain_id])
            entry.pack(side=tk.LEFT, padx=2)
            self.chain_color_entries[chain_id] = entry
            
            pick_btn = ttk.Button(chain_frame, text="Pick", width=8,
                                command=lambda cid=chain_id: self.pick_chain_color(cid))
            pick_btn.pack(side=tk.LEFT, padx=2)
            self.chain_color_pick_buttons[chain_id] = pick_btn
    
    def pick_chain_color(self, chain_id):
        """Open color picker for a specific chain color."""
        import tkinter.colorchooser
        current_color = self.chain_color_map.get(chain_id, self.default_protein_color)
        color = tkinter.colorchooser.askcolor(title=f"Pick Color for Chain {chain_id}", 
                                             color=current_color)[1]
        if color:
            self.chain_color_map[chain_id] = color
            if chain_id in self.chain_color_entries:
                self.chain_color_entries[chain_id].delete(0, tk.END)
                self.chain_color_entries[chain_id].insert(0, color)
    
    def apply_chain_colors(self):
        """Apply color settings and regenerate projections."""
        # Update color mode
        self.color_mode = self.color_mode_var.get()
        
        # Update bio material colors
        self.default_protein_color = self.protein_color_entry.get().strip()
        self.default_nucleic_color = self.nucleic_color_entry.get().strip()
        
        # Update chain colors if in chain mode
        if self.color_mode == 'chain':
            # Clear existing map and rebuild with cleaned chain IDs
            self.chain_color_map = {}
            for chain_id, entry in self.chain_color_entries.items():
                color = entry.get().strip()
                if color:
                    # Ensure chain_id is a clean string
                    chain_id_clean = str(chain_id).strip()
                    self.chain_color_map[chain_id_clean] = color
            print(f"Chain color map updated: {self.chain_color_map}")
        else:
            # Clear chain color map when in bio_material mode
            self.chain_color_map = {}
        
        # Clear ALL projection cache to force regeneration with new colors
        with self.cache_lock:
            self.projection_cache.clear()
        
        # Clear display cache
        self.cached_blank_image = None
        self.cached_with_projections = None
        self.cached_micrograph_idx = None
        
        # Regenerate projections for current micrograph if loaded
        if (self.pdb_data is not None and 
            self.current_micrograph_idx is not None and 
            self.current_particles is not None and
            len(self.current_particles['poses']) > 0):
            
            mg_file = self.micrograph_files[self.current_micrograph_idx]
            base_name = mg_file.stem
            num_particles = len(self.current_particles['poses'])
            
            self.status_var.set(f"Regenerating {num_particles} projections with new colors in background...")
            self.root.update()
            
            # Regenerate in background to keep GUI responsive
            # Use the same background generation approach as load_micrograph
            def regenerate_in_background():
                try:
                    self._generate_all_projections(self.current_micrograph_idx, 
                                                 base_filename=base_name, 
                                                 limit=None,  # Generate all
                                                 background=True)
                    # Update display when done
                    self.root.after(0, self.update_display)
                    self.root.after(0, lambda: self.status_var.set("Colors applied - projections regenerated"))
                except Exception as e:
                    print(f"Error regenerating projections: {e}")
                    import traceback
                    traceback.print_exc()
                    self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            
            # Start background thread
            import threading
            thread = threading.Thread(target=regenerate_in_background, daemon=True)
            thread.start()
            
            # Update display immediately to show cleared cache
            self.update_display()
        else:
            messagebox.showinfo("Info", "Colors updated. Projections will use new colors when generated.")
    
    def open_chimerax(self, particle_idx=None):
        """Open the PDB structure in ChimeraX, optionally with a specific particle's view."""
        # Add confirmation dialog
        if particle_idx is not None:
            confirm_msg = f"Open particle {particle_idx+1} in ChimeraX?"
        else:
            confirm_msg = "Open structure in ChimeraX?"
        
        if not messagebox.askyesno("Confirm", confirm_msg):
            return
        
        if self.pdb_path is None:
            pdb_path = self.pdb_entry.get()
        else:
            pdb_path = self.pdb_path
        
        if not pdb_path or not Path(pdb_path).exists():
            messagebox.showerror("Error", "PDB file not found")
            return
        
        # Try to find ChimeraX in common locations
        import shutil
        chimerax_path = shutil.which('chimerax')
        if not chimerax_path:
            # Try macOS app bundle
            for app_path in ['/Applications/ChimeraX.app/Contents/bin/ChimeraX',
                           '/Applications/ChimeraX-1.9.app/Contents/bin/ChimeraX']:
                if Path(app_path).exists():
                    chimerax_path = app_path
                    break
        
        if not chimerax_path:
            messagebox.showerror("Error", "ChimeraX not found. Please install ChimeraX to view structures.")
            return
        
        try:
            # Use absolute path
            pdb_abs = str(Path(pdb_path).absolute())
            
            # Escape path for ChimeraX
            def escape_path(path):
                path_str = str(path).replace('\\', '/')
                if ' ' in path_str:
                    return f'"{path_str}"'
                return path_str
            
            pdb_escaped = escape_path(pdb_abs)
            
            # Build command script
            if particle_idx is not None and self.current_particles is not None and particle_idx < len(self.current_particles['poses']):
                # Open with specific particle's view
                pose = self.current_particles['poses'][particle_idx]
                view_cmd = self._euler_to_chimerax_view(pose)
                
                # Add custom vector arrow visualization if defined
                vector_arrow_cmd = ""
                if self.show_custom_arrow and self.custom_vector_3d is not None:
                    # Calculate the rotated vector in world space
                    from scipy.spatial.transform import Rotation
                    rot = Rotation.from_euler('ZYZ', pose, degrees=False)
                    R = rot.as_matrix()
                    rotated_vector = R @ self.custom_vector_3d
                    
                    # Draw arrow from center of structure
                    # Arrow length: 50 Angstroms
                    arrow_length = 50.0
                    start_point = [0.0, 0.0, 0.0]  # Center of structure
                    end_point = [
                        start_point[0] + rotated_vector[0] * arrow_length,
                        start_point[1] + rotated_vector[1] * arrow_length,
                        start_point[2] + rotated_vector[2] * arrow_length
                    ]
                    
                    # Convert custom arrow color from hex to RGB (0-1 range)
                    hex_color = self.custom_arrow_color.lstrip('#')
                    r = int(hex_color[0:2], 16) / 255.0
                    g = int(hex_color[2:4], 16) / 255.0
                    b = int(hex_color[4:6], 16) / 255.0
                    
                    # Draw arrow using ChimeraX graphics
                    # Format: graphics arrow x1,y1,z1 x2,y2,z2 radius color
                    vector_arrow_cmd = f"""
graphics arrow {start_point[0]:.3f},{start_point[1]:.3f},{start_point[2]:.3f} {end_point[0]:.3f},{end_point[1]:.3f},{end_point[2]:.3f} radius 2 color {r:.3f},{g:.3f},{b:.3f}
"""
                
                # Set window size multiple times to prevent resizing
                # ChimeraX resizes during various operations, so we set it at key points
                # Try to disable auto-resize behavior if possible
                # Build script - ensure proper centering and view application
                # CRITICAL: Center the object AFTER applying the view matrix to ensure it's centered in the view
                script_content = f"""open {pdb_escaped}
windowsize 800 600
show #1
lighting soft
graphics silhouettes true width 4
lighting depthCue true depthCueStart 0.4 depthCueEnd 0.55
set bgColor white
cofr centerOfView
{view_cmd}
cofr centerOfView
view
color #1 #48B8D0
color #1 & nucleic #62466B
{vector_arrow_cmd}
"""
                # Write temporary script in working directory
                import uuid
                script_file = Path.cwd() / f'chimerax_view_{uuid.uuid4().hex[:8]}.cxc'
                with open(script_file, 'w') as f:
                    f.write(script_content)
                
                # Save current window geometry AND axes limits before launching ChimeraX
                saved_root_geometry = self.root.geometry()
                saved_xlim = self.ax.get_xlim()
                saved_ylim = self.ax.get_ylim()
                was_resizable = (self.root.resizable()[0], self.root.resizable()[1])
                
                # Temporarily disable window resizing to prevent ChimeraX from affecting it
                self.root.resizable(False, False)
                
                # Open ChimeraX with script - launch in completely detached process
                cmd = [chimerax_path, '--script', str(script_file)]
                import platform
                if platform.system() == 'Windows':
                    subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    # Completely detach from parent process - don't use preexec_fn to avoid errors
                    subprocess.Popen(cmd, start_new_session=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Restore window geometry, axes limits, and re-enable resizing
                def restore_all():
                    if saved_root_geometry:
                        self.root.geometry(saved_root_geometry)
                    # Restore axes limits to prevent micrograph from resizing
                    self.ax.set_xlim(saved_xlim)
                    self.ax.set_ylim(saved_ylim)
                    self.canvas.draw_idle()
                    # Re-enable resizing after a delay
                    self.root.after(1000, lambda: self.root.resizable(was_resizable[0], was_resizable[1]))
                
                # Restore multiple times to catch any delayed resize events
                self.root.after_idle(restore_all)
                for delay in [10, 50, 100, 200, 500, 1000]:
                    self.root.after(delay, restore_all)
                
                self.status_var.set(f"Opened ChimeraX with particle {particle_idx+1} view")
            else:
                # Save current window geometry AND axes limits before launching ChimeraX
                saved_root_geometry = self.root.geometry()
                saved_xlim = self.ax.get_xlim()
                saved_ylim = self.ax.get_ylim()
                was_resizable = (self.root.resizable()[0], self.root.resizable()[1])
                
                # Temporarily disable window resizing to prevent ChimeraX from affecting it
                self.root.resizable(False, False)
                
                # Just open the PDB - launch in completely detached process
                cmd = [chimerax_path, pdb_abs]
                import platform
                if platform.system() == 'Windows':
                    subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    # Completely detach from parent process - don't use preexec_fn to avoid errors
                    subprocess.Popen(cmd, start_new_session=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Restore window geometry, axes limits, and re-enable resizing
                def restore_all():
                    if saved_root_geometry:
                        self.root.geometry(saved_root_geometry)
                    # Restore axes limits to prevent micrograph from resizing
                    self.ax.set_xlim(saved_xlim)
                    self.ax.set_ylim(saved_ylim)
                    self.canvas.draw_idle()
                    # Re-enable resizing after a delay
                    self.root.after(1000, lambda: self.root.resizable(was_resizable[0], was_resizable[1]))
                
                # Restore multiple times to catch any delayed resize events
                self.root.after_idle(restore_all)
                for delay in [10, 50, 100, 200, 500, 1000]:
                    self.root.after(delay, restore_all)
                
                self.status_var.set("Opened PDB in ChimeraX")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open ChimeraX: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _restore_window_geometry(self):
        """Restore the saved window geometry to prevent resizing from external events."""
        if self.saved_geometry:
            self.root.geometry(self.saved_geometry)
            # Try again after a longer delay in case ChimeraX takes time to launch
            self.root.after(500, lambda: self.root.geometry(self.saved_geometry) if self.saved_geometry else None)
    
    def on_canvas_click(self, event):
        """Handle mouse clicks on the canvas."""
        self._reset_idle_timer()  # Reset idle timer on user interaction
        if event.inaxes != self.ax:
            return
        
        # Get click coordinates in data space
        x_click = event.xdata
        y_click = event.ydata
        
        if x_click is None or y_click is None:
            return
        
        # Handle zoom mode - left click starts box selection
        if self.zoom_mode and event.button == 1:  # Left click in zoom mode
            self.zoom_box_start = (x_click, y_click)
            # Remove any existing zoom box
            if self.zoom_box_rect is not None:
                self.zoom_box_rect.remove()
                self.zoom_box_rect = None
            return
        
        # Handle right-click - show context menu to add projection
        if event.button == 3:  # Right click
            if self.current_particles is None:
                return
            
            # Find the closest particle
            mg_shape = self.current_particles.get('micrograph_shape')
            if mg_shape is None:
                mg_shape = self.original_micrograph.shape if self.original_micrograph is not None else None
            
            if mg_shape is None:
                return
            
            min_dist = float('inf')
            closest_particle_idx = None
            
            for i in range(len(self.current_particles['poses'])):
                x_frac = self.current_particles['center_x_frac'][i]
                y_frac = self.current_particles['center_y_frac'][i]
                x_pixel = x_frac * mg_shape[1]
                y_pixel = y_frac * mg_shape[0]
                
                # Calculate distance
                dist = np.sqrt((x_click - x_pixel)**2 + (y_click - y_pixel)**2)
                
                # Check if within projection size radius
                click_radius = max(self.projection_size, 50)
                if dist < click_radius and dist < min_dist:
                    min_dist = dist
                    closest_particle_idx = i
            
            # Show context menu if clicked near a particle
            if closest_particle_idx is not None:
                self.show_particle_context_menu(event, closest_particle_idx)
            return
        
        # Handle normal left click - open ChimeraX with particle view (only if not in zoom mode)
        if event.button == 1 and not self.zoom_mode:
            if self.current_particles is None:
                return
            
            # Find the closest particle
            mg_shape = self.current_particles.get('micrograph_shape')
            if mg_shape is None:
                mg_shape = self.original_micrograph.shape if self.original_micrograph is not None else None
            
            if mg_shape is None:
                return
            
            min_dist = float('inf')
            closest_particle_idx = None
            
            for i in range(len(self.current_particles['poses'])):
                x_frac = self.current_particles['center_x_frac'][i]
                y_frac = self.current_particles['center_y_frac'][i]
                x_pixel = x_frac * mg_shape[1]
                y_pixel = y_frac * mg_shape[0]
                
                # Calculate distance
                dist = np.sqrt((x_click - x_pixel)**2 + (y_click - y_pixel)**2)
                
                # Check if within projection size radius
                click_radius = max(self.projection_size, 50)
                if dist < click_radius and dist < min_dist:
                    min_dist = dist
                    closest_particle_idx = i
            
            # If clicked near a particle, open PDB in ChimeraX with that view
            if closest_particle_idx is not None:
                self.open_chimerax(particle_idx=closest_particle_idx)
    
    def on_canvas_motion(self, event):
        """Handle mouse motion for zoom box drawing."""
        if not self.zoom_mode or self.zoom_box_start is None:
            return
        
        if event.inaxes != self.ax:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        # Remove old rectangle
        if self.zoom_box_rect is not None:
            self.zoom_box_rect.remove()
        
        # Draw new rectangle
        x0, y0 = self.zoom_box_start
        x1, y1 = event.xdata, event.ydata
        
        width = x1 - x0
        height = y1 - y0
        
        from matplotlib.patches import Rectangle
        self.zoom_box_rect = Rectangle((x0, y0), width, height, 
                                       linewidth=2, edgecolor='red', 
                                       facecolor='none', linestyle='--')
        self.ax.add_patch(self.zoom_box_rect)
        self.canvas.draw_idle()
    
    def on_canvas_release(self, event):
        """Handle mouse release for zoom box completion."""
        if not self.zoom_mode or self.zoom_box_start is None:
            return
        
        if event.inaxes != self.ax or event.button != 1:
            return
        
        if event.xdata is None or event.ydata is None:
            self.zoom_box_start = None
            if self.zoom_box_rect is not None:
                self.zoom_box_rect.remove()
                self.zoom_box_rect = None
                self.canvas.draw_idle()
            return
        
        # Get box coordinates
        x0, y0 = self.zoom_box_start
        x1, y1 = event.xdata, event.ydata
        
        # Ensure x0 < x1 and y0 < y1
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        
        # Save original limits before first zoom (if not already saved)
        if self.zoom_xlim is None:
            # Get current limits before zooming
            self.zoom_xlim = list(self.ax.get_xlim())
            self.zoom_ylim = list(self.ax.get_ylim())
        
        # Get the image dimensions to calculate proper aspect ratio
        if self.original_micrograph is not None:
            img_height, img_width = self.original_micrograph.shape[:2]
            img_aspect = img_width / img_height
            
            # Get the current figure size to calculate display aspect ratio
            fig_width, fig_height = self.fig.get_size_inches()
            ax_bbox = self.ax.get_position()
            ax_width = fig_width * ax_bbox.width
            ax_height = fig_height * ax_bbox.height
            display_aspect = ax_width / ax_height
            
            # Calculate the selected box dimensions
            box_width = x_max - x_min
            box_height = y_max - y_min
            
            # Adjust zoom limits to maintain image aspect ratio
            # We want to preserve the 1:1 pixel aspect ratio of the image
            if box_height > 0:
                box_aspect = box_width / box_height
                
                # Adjust limits to match display aspect ratio while centering the selection
                if box_aspect > display_aspect:
                    # Selected box is wider than display - expand height to match
                    center_y = (y_min + y_max) / 2
                    new_height = box_width / display_aspect
                    y_min = center_y - new_height / 2
                    y_max = center_y + new_height / 2
                else:
                    # Selected box is taller than display - expand width to match
                    center_x = (x_min + x_max) / 2
                    new_width = box_height * display_aspect
                    x_min = center_x - new_width / 2
                    x_max = center_x + new_width / 2
        
        # Apply zoom with adjusted limits to maintain aspect ratio
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        # Ensure aspect ratio is maintained (equal means 1:1 pixel ratio)
        self.ax.set_aspect('equal', adjustable='box')
        
        # Remove zoom box
        if self.zoom_box_rect is not None:
            self.zoom_box_rect.remove()
            self.zoom_box_rect = None
        
        # Disable zoom mode
        self.zoom_mode = False
        self.zoom_button.config(relief=tk.RAISED)
        self.reset_zoom_button.config(state=tk.NORMAL)
        
        self.canvas.draw()
    
    def on_key_press(self, event):
        """Handle key press events."""
        # ESC key to exit zoom box drawing mode
        if event.key == 'escape' and self.zoom_mode:
            self.zoom_mode = False
            self.zoom_box_start = None
            if self.zoom_box_rect is not None:
                self.zoom_box_rect.remove()
                self.zoom_box_rect = None
            self.zoom_button.config(relief=tk.RAISED)
            self.status_var.set("Ready")
            self.canvas.draw()
    
    def toggle_zoom_mode(self):
        """Toggle zoom box selection mode."""
        self.zoom_mode = not self.zoom_mode
        if self.zoom_mode:
            self.zoom_button.config(relief=tk.SUNKEN)
            self.status_var.set("Zoom mode: Click and drag to select area")
        else:
            self.zoom_button.config(relief=tk.RAISED)
            self.status_var.set("Ready")
    
    def reset_zoom(self):
        """Reset zoom to show full image."""
        if self.zoom_xlim is not None and self.zoom_ylim is not None:
            # Restore original limits
            self.ax.set_xlim(self.zoom_xlim)
            self.ax.set_ylim(self.zoom_ylim)
            self.zoom_xlim = None
            self.zoom_ylim = None
        else:
            # Reset to auto limits (full image)
            self.ax.relim()
            self.ax.autoscale()
        
        self.reset_zoom_button.config(state=tk.DISABLED)
        self.canvas.draw()
    
    def increase_projection_count(self):
        """Increase the number of projections to show."""
        self._reset_idle_timer()  # Reset idle timer on user interaction
        if self.current_particles is None:
            return
        
        max_particles = len(self.current_particles['poses'])
        if self.projection_generation_limit < max_particles:
            self.projection_generation_limit += 1
            self.projection_count_var.set(self.projection_generation_limit)
            self.projection_count_label.config(text=str(self.projection_generation_limit))
            
            # Generate additional projections if needed
            if self.current_micrograph_idx is not None:
                cache_key = (self.current_micrograph_idx, self.projection_generation_limit - 1)
                with self.cache_lock:
                    cache_missing = cache_key not in self.projection_cache
                if cache_missing:
                    # Need to generate this projection
                    if self.pdb_data is not None:
                        mg_file = self.micrograph_files[self.current_micrograph_idx]
                        base_name = mg_file.stem
                        self._generate_all_projections(self.current_micrograph_idx, 
                                                      base_filename=base_name, 
                                                      limit=self.projection_generation_limit)
                self.update_display()
    
    def decrease_projection_count(self):
        """Decrease the number of projections to show."""
        self._reset_idle_timer()  # Reset idle timer on user interaction
        if self.projection_generation_limit > 1:
            self.projection_generation_limit -= 1
            self.projection_count_var.set(self.projection_generation_limit)
            self.projection_count_label.config(text=str(self.projection_generation_limit))
            self.update_display()
    
    def show_particle_context_menu(self, event, particle_idx):
        """Show context menu for right-clicked particle."""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Add Projection Image", 
                        command=lambda: self.add_projection_for_particle(particle_idx))
        menu.add_command(label="Compare: Micrograph vs Projection", 
                        command=lambda: self.compare_particle_projection(particle_idx))
        menu.add_command(label="Open in ChimeraX", 
                        command=lambda: self.open_chimerax(particle_idx=particle_idx))
        
        # Convert matplotlib event coordinates to tkinter coordinates
        # Get the canvas widget
        canvas_widget = self.canvas.get_tk_widget()
        # Get the figure bbox in display coordinates
        bbox = self.canvas.figure.bbox
        # Convert matplotlib event coordinates to tkinter coordinates
        x_tk = canvas_widget.winfo_rootx() + int(event.x)
        y_tk = canvas_widget.winfo_rooty() + int(event.y)
        
        menu.tk_popup(x_tk, y_tk)
    
    def compare_particle_projection(self, particle_idx):
        """
        Create a side-by-side comparison image showing:
        - Left: Actual micrograph particle region (extracted like a particle extractor)
        - Right: Simulated PyMOL projection
        
        This allows verification that the projection matching is correct.
        """
        if self.current_micrograph_idx is None or self.current_particles is None:
            messagebox.showwarning("No Data", "No micrograph or particle data loaded.")
            return
        
        if particle_idx >= len(self.current_particles['poses']):
            messagebox.showwarning("Invalid Particle", f"Particle index {particle_idx} is out of range.")
            return
        
        if self.current_micrograph is None:
            messagebox.showwarning("No Micrograph", "No micrograph loaded.")
            return
        
        if self.pdb_data is None:
            messagebox.showwarning("No Structure", "No structure file loaded. Cannot generate projection.")
            return
        
        # Run in background thread to avoid freezing GUI
        def generate_comparison():
            try:
                # Update status in main thread
                self.root.after(0, lambda: self.status_var.set(f"Generating comparison for particle {particle_idx+1}..."))
                
                # Get particle data
                pose = self.current_particles['poses'][particle_idx]
                shift = self.current_particles['shifts'][particle_idx]
                center_x_frac = self.current_particles['center_x_frac'][particle_idx]
                center_y_frac = self.current_particles['center_y_frac'][particle_idx]
                
                # Get pixel size - try from current_particles first, then GUI entry
                if 'pixel_size' in self.current_particles and len(self.current_particles['pixel_size']) > particle_idx:
                    pixel_size = self.current_particles['pixel_size'][particle_idx]
                else:
                    # Fall back to GUI entry or default
                    try:
                        pixel_size = float(self.pixel_size_entry.get().strip())
                    except (ValueError, AttributeError):
                        pixel_size = 1.1  # Default
            
            # Get micrograph shape - use the stored shape (singular, not array)
            if 'micrograph_shape' in self.current_particles and self.current_particles['micrograph_shape'] is not None:
                micrograph_shape = self.current_particles['micrograph_shape']
            else:
                # Fall back to actual micrograph dimensions
                micrograph_shape = self.current_micrograph.shape
            
            # Calculate particle center in pixel coordinates
            mg_height, mg_width = micrograph_shape[0], micrograph_shape[1]
            x_pixel = center_x_frac * mg_width
            y_pixel = center_y_frac * mg_height
            
            # Apply shifts
            x_pixel += shift[0] / pixel_size
            y_pixel += shift[1] / pixel_size
            
            # Apply fine-tuning offsets
            x_pixel += self.projection_offset_x
            y_pixel += self.projection_offset_y
            
            # Get box size from GUI entry (default 256)
            try:
                box_size = int(self.particle_box_size_entry.get().strip())
                if box_size < 32:
                    box_size = 256  # Minimum reasonable size
                if box_size > 1024:
                    box_size = 1024  # Maximum reasonable size
            except (ValueError, AttributeError):
                box_size = 256  # Default
            
            half_size = box_size / 2.0
            
            # Calculate extraction bounds (in micrograph coordinates)
            # Note: micrograph is stored as [height, width]
            # x_pixel and y_pixel are in micrograph pixel coordinates
            x_min = int(max(0, x_pixel - half_size))
            x_max = int(min(mg_width, x_pixel + half_size))
            y_min = int(max(0, y_pixel - half_size))
            y_max = int(min(mg_height, y_pixel + half_size))
            
            # Extract region from micrograph
            # Micrograph array indexing: [row, col] = [y, x]
            # y increases downward in array, but upward in micrograph coordinates
            # So we need: mg[mg_height - y_max : mg_height - y_min, x_min : x_max]
            mg_extracted = self.current_micrograph[mg_height - y_max:mg_height - y_min, x_min:x_max]
            
            # Create output array and pad if needed (if near edges)
            mg_output = np.zeros((box_size, box_size), dtype=mg_extracted.dtype)
            extracted_h, extracted_w = mg_extracted.shape
            
            # Calculate padding offsets
            pad_y = (box_size - extracted_h) // 2
            pad_x = (box_size - extracted_w) // 2
            
            # Place extracted region in center of output
            mg_output[pad_y:pad_y + extracted_h, pad_x:pad_x + extracted_w] = mg_extracted
            
            # Normalize micrograph region to [0, 1] for display
            if mg_output.max() > mg_output.min():
                mg_extracted_norm = (mg_output - mg_output.min()) / (mg_output.max() - mg_output.min())
            else:
                mg_extracted_norm = mg_output.copy().astype(np.float32)
            
            # Generate simulated EM projection (right side)
            # This simulates what the EM image would look like from this view
            em_projection = simulate_em_projection_from_pdb(
                self.pdb_data,
                pose,
                output_size=(box_size, box_size),
                pixel_size=pixel_size,
                atom_radius=2.0  # 2 Angstrom atom radius
            )
            
            # Normalize EM projection to [0, 1] for display
            if em_projection.max() > em_projection.min():
                em_proj_norm = (em_projection - em_projection.min()) / (em_projection.max() - em_projection.min())
            else:
                em_proj_norm = em_projection.copy().astype(np.float32)
            
            # Create side-by-side image
            comparison = np.zeros((box_size, box_size * 2, 3), dtype=np.float32)
            # Left side: actual micrograph (grayscale -> RGB)
            comparison[:, :box_size, 0] = mg_extracted_norm
            comparison[:, :box_size, 1] = mg_extracted_norm
            comparison[:, :box_size, 2] = mg_extracted_norm
            # Right side: simulated EM projection (grayscale -> RGB)
            comparison[:, box_size:, 0] = em_proj_norm
            comparison[:, box_size:, 1] = em_proj_norm
            comparison[:, box_size:, 2] = em_proj_norm
            
                # Create window in main thread
                def create_window():
                    comparison_window = tk.Toplevel(self.root)
                    comparison_window.title(f"Particle {particle_idx+1} Comparison: Micrograph vs Projection")
                    comparison_window.geometry(f"{box_size * 2 + 100}x{box_size + 100}")
                    
                    # Create matplotlib figure
                    fig = Figure(figsize=(box_size * 2 / 100, box_size / 100), dpi=100)
                    ax = fig.add_subplot(111)
                    ax.imshow(comparison, origin='lower', aspect='auto')
                    ax.axvline(x=box_size, color='red', linewidth=2, linestyle='--', label='Divider')
                    ax.set_xlabel('Left: Actual Micrograph | Right: Simulated Projection')
                    ax.set_title(f'Particle {particle_idx+1} Comparison\n(Should match if projection mapping is correct)')
                    ax.axis('off')
                    
                    # Add labels
                    ax.text(box_size / 2, box_size - 20, 'Actual Micrograph', 
                           ha='center', va='top', color='white', fontsize=12, 
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    ax.text(box_size * 1.5, box_size - 20, 'Simulated Projection', 
                           ha='center', va='top', color='white', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    
                    # Embed in tkinter
                    canvas = FigureCanvasTkAgg(fig, comparison_window)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    
                    # Add save button
                    def save_comparison():
                        from tkinter import filedialog
                        filename = filedialog.asksaveasfilename(
                            defaultextension='.png',
                            filetypes=[('PNG files', '*.png'), ('All files', '*.*')],
                            initialfile=f'particle_{particle_idx+1}_comparison.png'
                        )
                        if filename:
                            fig.savefig(filename, dpi=150, bbox_inches='tight')
                            messagebox.showinfo("Saved", f"Comparison saved to {filename}")
                    
                    save_button = tk.Button(comparison_window, text="Save Comparison", command=save_comparison)
                    save_button.pack(pady=5)
                    
                    self.status_var.set(f"Comparison generated for particle {particle_idx+1}")
                
                # Create window in main thread
                self.root.after(0, create_window)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate comparison: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Error generating comparison")
    
    def add_projection_for_particle(self, particle_idx):
        """Generate and add projection for a specific particle."""
        if self.current_micrograph_idx is None or self.current_particles is None:
            return
        
        if particle_idx >= len(self.current_particles['poses']):
            return
        
        # Check if already cached (thread-safe)
        cache_key = (self.current_micrograph_idx, particle_idx)
        with self.cache_lock:
            already_cached = cache_key in self.projection_cache
        
        # If already cached, just update the display limit to show it
        if already_cached:
            if particle_idx >= self.projection_generation_limit:
                self.projection_generation_limit = particle_idx + 1
                if hasattr(self, 'projection_count_var'):
                    self.projection_count_var.set(self.projection_generation_limit)
                if hasattr(self, 'projection_count_label'):
                    self.projection_count_label.config(text=str(self.projection_generation_limit))
                self.update_display()
                self.status_var.set(f"Showing projection for particle {particle_idx+1}")
            else:
                messagebox.showinfo("Info", f"Projection for particle {particle_idx+1} already exists and is being displayed.")
            return
        
        # Generate projection for this particle
        if self.pdb_data is not None:
            mg_file = self.micrograph_files[self.current_micrograph_idx]
            base_name = mg_file.stem
            
            pose = self.current_particles['poses'][particle_idx]
            
            try:
                self.status_var.set(f"Generating projection for particle {particle_idx+1}...")
                self.root.update()
                
                # Generate projection using PDB structure
                projection = self._generate_pdb_projection_for_particle(
                    particle_idx, pose, output_size=(self.projection_size, self.projection_size)
                )
                
                # Resize if needed
                if projection.shape[0] != self.projection_size or projection.shape[1] != self.projection_size:
                    from PIL import Image
                    rgba_uint8 = (projection * 255.0).astype(np.uint8)
                    pil_img = Image.fromarray(rgba_uint8)
                    pil_img_resized = pil_img.resize((self.projection_size, self.projection_size), 
                                                     Image.Resampling.LANCZOS)
                    projection = np.array(pil_img_resized).astype(np.float32) / 255.0
                    projection[:,:,3] = np.clip(projection[:,:,3], 0, 1)
                
                # Cache the projection (thread-safe)
                with self.cache_lock:
                    self.projection_cache[cache_key] = projection
                
                # Update projection count limit to include this particle if needed
                if particle_idx >= self.projection_generation_limit:
                    self.projection_generation_limit = particle_idx + 1
                    if hasattr(self, 'projection_count_var'):
                        self.projection_count_var.set(self.projection_generation_limit)
                    if hasattr(self, 'projection_count_label'):
                        self.projection_count_label.config(text=str(self.projection_generation_limit))
                
                self.status_var.set(f"Added projection for particle {particle_idx+1}")
                self.update_display()
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate projection: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _get_chimerax_executable(self, chimerax_path):
        """Get the actual ChimeraX executable path from various input formats."""
        chimerax_path_obj = Path(chimerax_path)
        
        # If it's already an executable file, return it
        if chimerax_path_obj.is_file() and chimerax_path_obj.exists():
            return chimerax_path_obj
        
        # If it's an app bundle, find the executable inside
        if chimerax_path.endswith('.app') or (chimerax_path_obj.exists() and chimerax_path_obj.is_dir()):
            # Try common locations inside app bundle
            for subpath in ['Contents/MacOS/ChimeraX', 'Contents/bin/ChimeraX']:
                exec_path = chimerax_path_obj / subpath
                if exec_path.exists():
                    return exec_path
        
        # If it's a directory that might contain the app, look for it
        if chimerax_path_obj.is_dir():
            for subpath in ['Contents/MacOS/ChimeraX', 'Contents/bin/ChimeraX']:
                exec_path = chimerax_path_obj / subpath
                if exec_path.exists():
                    return exec_path
        
        return chimerax_path_obj
    

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='POSEMAP - Pose-Oriented Single-particle EM Micrograph Annotation & Projection')
    parser.add_argument('--refinement-cs', type=str, help='Path to refinement .cs file')
    parser.add_argument('--passthrough-cs', type=str, help='Path to passthrough .cs file')
    parser.add_argument('--pdb', type=str, help='Path to PDB structure file')
    parser.add_argument('--micrograph-dir', type=str, help='Directory containing micrograph files')
    
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ParticleMapperGUI(root, 
                           refinement_cs_path=args.refinement_cs,
                           passthrough_cs_path=args.passthrough_cs,
                           pdb_path=args.pdb,
                           micrograph_dir=args.micrograph_dir)
    root.mainloop()


if __name__ == '__main__':
    main()

