import numpy as np
import cv2 # type: ignore
import time
import argparse
import csv
from dataclasses import dataclass
import json

# ==========================================
# 1. Configuration Data Structure
# ==========================================
@dataclass
class SimConfig:
    nx: int = 600
    ny: int = 200
    Re: float = 250.0
    uLB: float = 0.04
    max_iter: int = 5000
    L_char: float = 20.0  
    
    # Environment boundaries
    top_bottom_walls: bool = False  # True = Closed Wind Tunnel, False = Open Space

    flow_profile: str = 'uniform'  # Can be 'uniform' or 'shear'
    
    save_video: bool = True
    video_filename: str = 'modular_out.mp4'
    save_csv: bool = False
    csv_filename: str = 'modular_data.csv'
    output_freq: int = 20 

    @property
    def nulb(self):
        return self.uLB * self.L_char / self.Re
    
    @property
    def omega(self):
        return 1.0 / (3.0 * self.nulb + 0.5)

# ==========================================
# 2. Geometry Module
# ==========================================

class Geometry:
    def get_mask(self, X, Y, step, rho=None, u=None):
        raise NotImplementedError()
        
    def get_tracking_data(self):
        """Returns: (Absolute X, Absolute Y, Deflection dX)"""
        return (0.0, 0.0, 0.0)

class Scene:
    def __init__(self):
        self.geometries = []

    def add_object(self, geometry: Geometry):
        self.geometries.append(geometry)

    # ADD rho and u here as well
    def get_mask(self, X, Y, step, rho=None, u=None):
        master_mask = np.zeros(X.shape, dtype=bool)
        for geom in self.geometries:
            master_mask = master_mask | geom.get_mask(X, Y, step, rho, u)
        return master_mask
    

class StationaryCylinder(Geometry):
    def __init__(self, cx, cy, r):
        self.cx = cx
        self.cy = cy
        self.r = r

    def get_mask(self, X, Y, step):
        return (X - self.cx)**2 + (Y - self.cy)**2 < self.r**2

    def get_tracking_point(self):
        return (self.cx, self.cy)

class OscillatingCylinder(Geometry):
    def __init__(self, cx, cy_base, r, amplitude, f_eigen):
        self.cx = cx
        self.cy_base = cy_base
        self.r = r
        self.amplitude = amplitude
        self.f_eigen = f_eigen
        self.current_cy = cy_base

    def get_mask(self, X, Y, step):
        self.current_cy = self.cy_base + self.amplitude * np.sin(2 * np.pi * self.f_eigen * step)
        return (X - self.cx)**2 + (Y - self.current_cy)**2 < self.r**2

    def get_tracking_point(self):
        return (self.cx, self.current_cy)
    
# ==========================================
# RECTANGLE GEOMETRIES
# ==========================================
class StationaryRectangle(Geometry):
    def __init__(self, cx, cy, width, height):
        self.cx = cx
        self.cy = cy
        self.w = width
        self.h = height

    def get_mask(self, X, Y, step):
        # A point is inside the rectangle if its X and Y distances are within half the width/height
        return (np.abs(X - self.cx) <= self.w / 2) & (np.abs(Y - self.cy) <= self.h / 2)

    def get_tracking_point(self):
        return (self.cx, self.cy)

class OscillatingRectangle(Geometry):
    def __init__(self, cx, cy_base, width, height, amplitude, f_eigen):
        self.cx = cx
        self.cy_base = cy_base
        self.w = width
        self.h = height
        self.amplitude = amplitude
        self.f_eigen = f_eigen
        self.current_cy = cy_base

    def get_mask(self, X, Y, step):
        self.current_cy = self.cy_base + self.amplitude * np.sin(2 * np.pi * self.f_eigen * step)
        return (np.abs(X - self.cx) <= self.w / 2) & (np.abs(Y - self.current_cy) <= self.h / 2)

    def get_tracking_point(self):
        return (self.cx, self.current_cy)

class FlexibleCantilever(Geometry):
    def __init__(self, cx, cy_base, width, height, stiffness, damping, mass):
        self.cx = cx
        self.cy_base = cy_base
        self.w = width
        self.h = height
        
        # Structural Physics Parameters
        self.k = stiffness
        self.c = damping
        self.m = mass
        
        # State Variables
        self.delta = 0.0  # Tip deflection distance
        self.vel = 0.0    # Tip velocity
        self.last_mask = None

    def get_mask(self, X, Y, step, rho=None, u=None):
        # 1. Calculate Fluid Force (If we have fluid data)
        if rho is not None and self.last_mask is not None:
            # Sample density (pressure) directly upstream and downstream of the base
            up_x = max(0, int(self.cx - self.w))
            #down_x = min(rho.shape[0] - 1, int(self.cx + self.w + self.delta))
            down_x = min(rho.shape[0] - 1, int(self.cx + self.w + 10))
            y_range = slice(int(self.cy_base), int(self.cy_base + self.h))

            # In LBM, pressure p = rho / 3. Force is the difference across the shape.
            p_up = np.sum(rho[up_x, y_range]) / 3.0
            p_down = np.sum(rho[down_x, y_range]) / 3.0
            F_fluid = (p_up - p_down) * 0.1 # Scale factor to prevent vacuum explosion

            # 2. Structural Solver (Euler Integration)
            acceleration = (F_fluid - self.k * self.delta - self.c * self.vel) / self.m
            self.vel += acceleration
            self.delta += self.vel

            self.vel += acceleration
            self.delta += self.vel
            
            # If it hits the limit, stop it from winding up
            if self.delta > self.h/2:
                self.delta = self.h/2
                self.vel = 0.0
            elif self.delta < -self.h/2:
                self.delta = -self.h/2
                self.vel = 0.0
            
            # Clip deflection to prevent it from tearing the grid apart
            self.delta = np.clip(self.delta, -self.h/2, self.h/2)

        # 3. Generate the Bent Mask
        # Deflection follows a parabolic curve: dX = delta * (y/H)^2
        Y_norm = np.clip((Y - self.cy_base) / self.h, 0, 1)
        dX = self.delta * (Y_norm ** 2)

        y_bounds = (Y >= self.cy_base) & (Y <= self.cy_base + self.h)
        x_bounds = np.abs(X - (self.cx + dX)) <= self.w / 2

        self.last_mask = y_bounds & x_bounds
        return self.last_mask

    def get_tracking_data(self):
        # Track the absolute tip coordinates AND the specific deflection delta
        return (self.cx + self.delta, self.cy_base + self.h, self.delta)
# ==========================================
# AIRFOIL GEOMETRIES (NACA 4-Digit Symmetric)
# ==========================================
class StationaryAirfoil(Geometry):
    def __init__(self, cx, cy, chord, thickness=0.12):
        self.cx = cx
        self.cy = cy
        self.c = chord        # Length of the airfoil from tip to tail
        self.t = thickness    # Maximum thickness as a fraction of the chord (0.12 = NACA 0012)
        self.le_x = cx - chord / 2  # Leading edge X coordinate

    def get_mask(self, X, Y, step):
        # 1. Normalize X coordinates along the chord from 0.0 to 1.0
        x_c = (X - self.le_x) / self.c
        
        # 2. Prevent invalid square roots by clipping negative values (they will be masked out anyway)
        x_safe = np.clip(x_c, 0, 1)
        
        # 3. The NACA symmetric airfoil thickness equation
        y_t = 5 * self.t * self.c * (
            0.2969 * np.sqrt(x_safe) - 
            0.1260 * x_safe - 
            0.3516 * x_safe**2 + 
            0.2843 * x_safe**3 - 
            0.1015 * x_safe**4
        )
        
        # 4. A point is inside if it is within the chord length bounds AND below the thickness curve
        in_chord = (x_c >= 0.0) & (x_c <= 1.0)
        return in_chord & (np.abs(Y - self.cy) <= y_t)

    def get_tracking_point(self):
        return (self.cx, self.cy)

class OscillatingAirfoil(Geometry):
    def __init__(self, cx, cy_base, chord, thickness, amplitude, f_eigen):
        self.cx = cx
        self.cy_base = cy_base
        self.c = chord
        self.t = thickness
        self.le_x = cx - chord / 2
        self.amplitude = amplitude
        self.f_eigen = f_eigen
        self.current_cy = cy_base

    def get_mask(self, X, Y, step):
        # Update Y position dynamically
        self.current_cy = self.cy_base + self.amplitude * np.sin(2 * np.pi * self.f_eigen * step)
        
        x_c = (X - self.le_x) / self.c
        x_safe = np.clip(x_c, 0, 1)
        
        y_t = 5 * self.t * self.c * (
            0.2969 * np.sqrt(x_safe) - 0.1260 * x_safe - 0.3516 * x_safe**2 + 
            0.2843 * x_safe**3 - 0.1015 * x_safe**4
        )
        
        in_chord = (x_c >= 0.0) & (x_c <= 1.0)
        return in_chord & (np.abs(Y - self.current_cy) <= y_t)

    def get_tracking_point(self):
        return (self.cx, self.current_cy)


# ==========================================
# 3. The Physics Solver
# ==========================================
class LatticeBoltzmannSolver:
    def __init__(self, config: SimConfig, scene: Scene):
        self.cfg = config
        self.scene = scene
        
        # D2Q9 Lattice parameters
        self.v = np.array([ [ 1,  1], [ 1,  0], [ 1, -1], [ 0,  1], [ 0,  0],
                            [ 0, -1], [-1,  1], [-1,  0], [-1, -1] ])
        self.t = np.array([ 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])

        self.col1, self.col2, self.col3 = np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])

        # Coordinate grids
        self.Y, self.X = np.meshgrid(np.arange(self.cfg.ny), np.arange(self.cfg.nx))
        
        self._init_fields()

    def _init_fields(self):
        # Original: Uniform flow with a tiny perturbation
        def inivel_uniform(d, x, y):
            return (1-d) * self.cfg.uLB * (1 + 1e-4 * np.sin(y/(self.cfg.ny-1) * 2 * np.pi))
            
        # New: Shear flow (Atmospheric Boundary Layer)
        def inivel_shear(d, x, y):
            # d=0 is X-velocity, d=1 is Y-velocity
            # Creates a gradient: 20% of uLB at the bottom, 180% of uLB at the top
            gradient = 0.2 + 1.6 * (y / (self.cfg.ny - 1))
            return (1-d) * self.cfg.uLB * gradient

        # Apply the chosen profile
        if self.cfg.flow_profile == 'shear':
            self.vel = np.fromfunction(inivel_shear, (2, self.cfg.nx, self.cfg.ny))
        else:
            self.vel = np.fromfunction(inivel_uniform, (2, self.cfg.nx, self.cfg.ny))
            
        self.fin = self._equilibrium(1.0, self.vel)

    def _macroscopic(self):
        rho = np.sum(self.fin, axis=0)
        u = np.zeros((2, self.cfg.nx, self.cfg.ny))
        for i in range(9):
            u[0,:,:] += self.v[i,0] * self.fin[i,:,:]
            u[1,:,:] += self.v[i,1] * self.fin[i,:,:]
        u /= rho
        return rho, u

    def _equilibrium(self, rho, u):
        usq = 3/2 * (u[0]**2 + u[1]**2)
        feq = np.zeros((9, self.cfg.nx, self.cfg.ny))
        for i in range(9):
            cu = 3 * (self.v[i,0]*u[0,:,:] + self.v[i,1]*u[1,:,:])
            feq[i,:,:] = rho * self.t[i] * (1 + cu + 0.5*cu**2 - usq)
        return feq

    def step(self, iter_step):
        # 1. Outflow condition
        self.fin[self.col3, -1, :] = self.fin[self.col3, -2, :]

        # 2. Macroscopic
        rho, u = self._macroscopic()

        # 3. Inflow condition
        u[:, 0, :] = self.vel[:, 0, :]
        rho[0, :] = 1 / (1 - u[0, 0, :]) * (np.sum(self.fin[self.col2, 0, :], axis=0) +
                                            2 * np.sum(self.fin[self.col3, 0, :], axis=0))
        feq = self._equilibrium(rho, u)
        self.fin[self.col1, 0, :] = feq[self.col1, 0, :] + self.fin[self.col3, 0, :] - feq[self.col3, 0, :]

        # 4. Collision
        fout = self.fin - self.cfg.omega * (self.fin - feq)

        # 5. ENVIRONMENT BOUNDARIES: Top and Bottom Walls
        if self.cfg.top_bottom_walls:
            for i in range(9):
                fout[i, :, 0] = self.fin[8-i, :, 0]         # Bottom Wall
                fout[i, :, -1] = self.fin[8-i, :, -1]       # Top Wall

        # 6. SCENE BOUNDARY: Combine all masks and apply Bounce-back
        obstacle = self.scene.get_mask(self.X, self.Y, iter_step, rho, u)
        for i in range(9):
            fout[i, obstacle] = self.fin[8-i, obstacle]

        # 7. Streaming
        for i in range(9):
            self.fin[i,:,:] = np.roll(np.roll(fout[i,:,:], self.v[i,0], axis=0), self.v[i,1], axis=1)
            
        return rho, u, obstacle
# ==========================================
# 4. Execution & Output Handler
# ==========================================
def run_simulation(config: SimConfig, scene: Scene):
    solver = LatticeBoltzmannSolver(config, scene)
    
    video_out = None
    csv_file = None
    csv_writer = None

    if config.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(config.video_filename, fourcc, 60.0, (config.nx, config.ny))
        
    if config.save_csv:
        csv_file = open(config.csv_filename, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        # NEW HEADERS: Perfectly set up for data science / Excel plotting
        csv_writer.writerow(['Step', 'Obj_Name', 'Obj_X', 'Obj_Y', 'Deflection_dX', 'Max_Flow_Vel'])

    print(f"Starting Simulation. Max steps: {config.max_iter}")
    start_time = time.time()

    for step in range(config.max_iter):
        rho, u, obstacle = solver.step(step)

        if step % config.output_freq == 0:
            # Handle CSV Logging
            if config.save_csv:
                max_u = np.max(np.sqrt(u[0]**2 + u[1]**2))
                
                # Loop through every object in the scene and log its specific data
                for idx, geom in enumerate(scene.geometries):
                    obj_name = f"{geom.__class__.__name__}_{idx}"
                    
                    # Fallback for older shapes that haven't been updated to get_tracking_data yet
                    if hasattr(geom, 'get_tracking_data'):
                        track_x, track_y, defl_x = geom.get_tracking_data()
                    else:
                        track_x, track_y = geom.get_tracking_point()
                        defl_x = 0.0
                        
                    csv_writer.writerow([step, obj_name, track_x, track_y, defl_x, max_u])

            # Handle Video Rendering
            if config.save_video:
                vorticity = (np.roll(u[1], -1, axis=0) - np.roll(u[1], 1, axis=0)) - \
                            (np.roll(u[0], -1, axis=1) - np.roll(u[0], 1, axis=1))
                
                vorticity[obstacle] = 0.0 
                vorticity = np.clip(vorticity, -0.02, 0.02)
                norm_vort = ((vorticity + 0.02) / 0.04 * 255).astype(np.uint8)
                
                frame = norm_vort.T
                colored_frame = cv2.applyColorMap(frame, cv2.COLORMAP_OCEAN)
                colored_frame[obstacle.T] = [0, 0, 0]
                
                if config.top_bottom_walls:
                    colored_frame[0, :] = [0, 0, 0]
                    colored_frame[-1, :] = [0, 0, 0]
                
                colored_frame = cv2.flip(colored_frame, 0)
                video_out.write(colored_frame)

        if step % 1000 == 0 and step > 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{config.max_iter} completed in {elapsed:.1f}s...")

    if config.save_video:
        video_out.release()
    if config.save_csv:
        csv_file.close()
        
    print("Simulation finished successfully!")


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