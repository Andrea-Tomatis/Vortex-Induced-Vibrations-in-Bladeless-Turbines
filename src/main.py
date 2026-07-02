"""
/src/main.py

MAIN FILE: This file contains the main() function to run a simulation given         
           some parameters. This parameters can be set in 2 ways:
           - from command line, using the notation python3 main.py --keyword value
           - from json file, using the notation python3 main.py --load_conf filename.json
             (an example of such json file is in ./src/input_test.json)

Usage: it is possible to execute the simulations as described above BUT,
       in case you want to run all a bunch of simulations with different 
       parameters it is suggested to use the file test.py that contains
       a framework to facilitate this operations.

The implementation in this file is complete and doesn't require any updates.
"""


import argparse
import json
from geometry import *
from solver import *
from sim_config import SimConfig


def run_batch_from_json(filepath):
    """
    Loads a JSON file containing multiple simulation configurations
    and runs them sequentially.
    """
    print(f"Loading batch configuration from: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)

    simulations = data.get("simulations", [])
    if not simulations:
        print("No simulations found in the JSON file.")
        return

    for i, sim_data in enumerate(simulations):
        name = sim_data.get("name", f"Simulation_{i+1}")
        print(f"\n{'='*50}\nStarting Batch Run: {name}\n{'='*50}")

        # 1. Build the Configuration
        cfg = SimConfig(
            nx=sim_data.get('nx', 600),
            ny=sim_data.get('ny', 200),
            Re=sim_data.get('re', 250.0),
            max_iter=sim_data.get('steps', 5000),
            top_bottom_walls=sim_data.get('walls', False),
            flow_profile=sim_data.get('flow', 'uniform'), # <--- ADD THIS LINE
            save_video=sim_data.get('video', True),
            video_filename=sim_data.get('video_out', f"{name}.mp4"),
            save_csv=sim_data.get('csv', False),
            csv_filename=sim_data.get('csv_out', f"{name}.csv")
        )

        # 2. Build the Scene
        main_scene = Scene()
        objects = sim_data.get('objects', [])
        
        if not objects:
            print(f"Warning: No objects defined for {name}. Running empty tunnel.")

        for idx, obj_data in enumerate(objects):
            shape = obj_data.get('shape', 'cylinder')
            mode = obj_data.get('mode', 'stationary')
            
            # Allow JSON to define exact coordinates, or fallback to defaults
            cx = obj_data.get('cx', cfg.nx // 5)
            cy = obj_data.get('cy', cfg.ny // 2)
            size = obj_data.get('size', cfg.ny // 10)

            amplitude = size * 0.8
            f_shedding = 0.2 * cfg.uLB / (2 * size)

            # Assign Characteristic Length based on the primary (first) object
            if idx == 0:
                cfg.L_char = (size * 3.0) if shape == 'airfoil' else (size * 2.0)

            # Generate the Geometry
            geom = None
            if shape == 'cylinder':
                geom = OscillatingCylinder(cx, cy, size, amplitude, f_shedding) if mode == 'oscillating' else StationaryCylinder(cx, cy, size)
            elif shape == 'rectangle':
                geom = OscillatingRectangle(cx, cy, size, size*2, amplitude, f_shedding) if mode == 'oscillating' else StationaryRectangle(cx, cy, size, size*2)
            elif shape == 'airfoil':
                geom = OscillatingAirfoil(cx, cy, size*3, 0.15, amplitude, f_shedding) if mode == 'oscillating' else StationaryAirfoil(cx, cy, size*3, 0.15)
            elif shape == 'flexible_pole':
                geom = FlexibleCantilever(
                    cx=cx, 
                    cy_base=obj_data.get('cy', 1),             # Defaults to bottom wall
                    width=obj_data.get('width', size // 2),    # Decidable width
                    height=obj_data.get('height', size * 5),   # Decidable height
                    stiffness=obj_data.get('stiffness', 0.005),# Decidable stiffness
                    damping=obj_data.get('damping', 0.001),    # Decidable damping friction
                    mass=obj_data.get('mass', 2.0)             # Decidable mass/weight
                )
            if geom:
                main_scene.add_object(geom)

        # 3. Execute the Simulation
        run_simulation(cfg, main_scene)


def main():
    parser = argparse.ArgumentParser(description="Absolute Modular LBM CFD Framework")
    
    # NEW: JSON Batch Configuration
    parser.add_argument('--load_conf', type=str, help="Path to a JSON file containing batch simulation setups.")
    
    # Physics & Environment Parameters
    parser.add_argument('--re', type=float, default=250.0, help="Reynolds number.")
    parser.add_argument('--steps', type=int, default=5000, help="Total iterations.")
    parser.add_argument('--nx', type=int, default=600, help="Grid width.")
    parser.add_argument('--ny', type=int, default=200, help="Grid height.")
    parser.add_argument('--walls', action='store_true', help="Enable solid top/bottom walls.")
    parser.add_argument('--flow', type=str, choices=['uniform', 'shear'], default='uniform', help="Choose the inflow velocity profile.") 

    # Geometry Parameters
    parser.add_argument('--shape', type=str, choices=['cylinder', 'rectangle', 'airfoil', 'flexible_pole'], default='cylinder')
    parser.add_argument('--mode', type=str, choices=['stationary', 'oscillating'], default='stationary')
    parser.add_argument('--multi', action='store_true', help="Adds a secondary oscillating cylinder.")
    
    # Output Controls
    parser.add_argument('--video', action='store_true', help="Enable MP4 video rendering.")
    parser.add_argument('--video_out', type=str, default='modular_out.mp4')
    parser.add_argument('--csv', action='store_true', help="Enable CSV data logging.")
    parser.add_argument('--csv_out', type=str, default='modular_data.csv')
    
    args = parser.parse_args()

    # ==========================================
    # EXECUTION ROUTING
    # ==========================================
    if args.load_conf:
        # Route 1: Run massive batch jobs from JSON
        run_batch_from_json(args.load_conf)
    else:
        # Route 2: Run a single quick job from the command line flags
        cfg = SimConfig(
            nx=args.nx, ny=args.ny, Re=args.re, max_iter=args.steps,
            top_bottom_walls=args.walls,
            flow_profile=args.flow,
            save_video=args.video, video_filename=args.video_out,
            save_csv=args.csv, csv_filename=args.csv_out
        )

        cx, cy = cfg.nx // 5, cfg.ny // 2
        size = cfg.ny // 10 
        amplitude = size * 0.8
        f_shedding = 0.2 * cfg.uLB / (2 * size)

        main_scene = Scene()

        if args.shape == 'cylinder':
            cfg.L_char = size * 2.0
            primary_geom = OscillatingCylinder(cx, cy, size, amplitude, f_shedding) if args.mode == 'oscillating' else StationaryCylinder(cx, cy, size)
        elif args.shape == 'rectangle':
            cfg.L_char = size * 2.0
            primary_geom = OscillatingRectangle(cx, cy, size, size*2, amplitude, f_shedding) if args.mode == 'oscillating' else StationaryRectangle(cx, cy, size, size*2)
        elif args.shape == 'airfoil':
            cfg.L_char = size * 3.0
            primary_geom = OscillatingAirfoil(cx, cy, size*3, 0.15, amplitude, f_shedding) if args.mode == 'oscillating' else StationaryAirfoil(cx, cy, size*3, 0.15)
        elif args.shape == 'flexible_pole':
            cfg.L_char = size * 2.0
            # A vertical rectangle anchored at the bottom of the grid
            primary_geom = FlexibleCantilever(
                cx=cx, 
                cy_base=1,               # Anchored to bottom wall
                width=size // 2,         # Thin profile
                height=cfg.ny // 2,      # Extends halfway up the tunnel
                stiffness=0.005,         # Low enough to bend
                damping=0.001,           # Prevents infinite oscillation
                mass=2.0                 # Inertia
            )
        main_scene.add_object(primary_geom)

        if args.multi:
            rear_cx = cx + cfg.nx // 3
            main_scene.add_object(OscillatingCylinder(rear_cx, cy, size, amplitude, f_shedding))

        run_simulation(cfg, main_scene)



if __name__ == "__main__":
    main()