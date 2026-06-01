"""
/src/solver.py

This file implement the Lattice Boltzmann method to simulate the dynamics.
It's the core of the entire code.

The implementation in this file is complete and doesn't require any updates.
"""


import numpy as np
import cv2 # type: ignore
import time
import csv
from main import SimConfig
from geometry import *


# Physics solver
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
        # Uniform flow with a tiny perturbation
        def inivel_uniform(d, x, y):
            return (1-d) * self.cfg.uLB * (1 + 1e-4 * np.sin(y/(self.cfg.ny-1) * 2 * np.pi))
            
        # Shear flow (Atmospheric Boundary Layer)
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
        # Outflow condition
        self.fin[self.col3, -1, :] = self.fin[self.col3, -2, :]

        # Macroscopic
        rho, u = self._macroscopic()

        # Inflow condition
        u[:, 0, :] = self.vel[:, 0, :]
        rho[0, :] = 1 / (1 - u[0, 0, :]) * (np.sum(self.fin[self.col2, 0, :], axis=0) +
                                            2 * np.sum(self.fin[self.col3, 0, :], axis=0))
        feq = self._equilibrium(rho, u)
        self.fin[self.col1, 0, :] = feq[self.col1, 0, :] + self.fin[self.col3, 0, :] - feq[self.col3, 0, :]

        # Collision
        fout = self.fin - self.cfg.omega * (self.fin - feq)

        # environmental boundaries: Top and Bottom Walls
        if self.cfg.top_bottom_walls:
            for i in range(9):
                fout[i, :, 0] = self.fin[8-i, :, 0]         # Bottom Wall
                fout[i, :, -1] = self.fin[8-i, :, -1]       # Top Wall

        # scene boundary: Combine all masks and apply Bounce-back
        obstacle = self.scene.get_mask(self.X, self.Y, iter_step, rho, u)
        for i in range(9):
            fout[i, obstacle] = self.fin[8-i, obstacle]

        # 7. Streaming
        for i in range(9):
            self.fin[i,:,:] = np.roll(np.roll(fout[i,:,:], self.v[i,0], axis=0), self.v[i,1], axis=1)
            
        return rho, u, obstacle
    

# Execution & Output Handler
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
