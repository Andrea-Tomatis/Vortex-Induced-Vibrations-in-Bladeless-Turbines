"""
./src/geometry.py

This file contains the class Geometry and all of its implementation.
There is currently 1 geometry implemented:
1 - Flexible Cylinder (top-down view of the bladeless turbine)

The flexible cylinder models the turbine mast as a single transverse degree of
freedom: the cross-section is rigid and translates along the lift axis under the
fluid force, restrained by a lumped mass-spring-damper. This is the base model
for a bladeless turbine.

Implementation method: In order to implement any new shape it is required
                       to define a new class that inherits from the class Geometry.
"""




from dataclasses import dataclass
import numpy as np



# ==========================================
# GEOMETRY MODULES
# ==========================================

class Geometry:
    def get_mask(self, X, Y, step, rho=None, u=None):
        raise NotImplementedError()

    def get_wall_velocity(self):
        """Rigid-body surface velocity (ux, uy) in lattice units.

        Used by the moving-wall bounce-back so the fluid feels the body move.
        Defaults to a stationary wall.
        """
        return (0.0, 0.0)

    def get_tracking_data(self):
        """Returns: (Absolute X, Absolute Y, Deflection dX)"""
        return (0.0, 0.0, 0.0)

class Scene:
    def __init__(self):
        self.geometries = []

    def add_object(self, geometry: Geometry):
        self.geometries.append(geometry)

    def get_mask(self, X, Y, step, rho=None, u=None):
        master_mask, _ = self.get_mask_and_wall_velocity(X, Y, step, rho, u)
        return master_mask

    def get_mask_and_wall_velocity(self, X, Y, step, rho=None, u=None):
        """Return (mask, u_wall) where u_wall[:, i, j] is the surface velocity
        of whichever body occupies cell (i, j), and zero elsewhere.

        Each body's own velocity is written only into its own cells, so an array
        of several independently moving turbines is handled correctly.
        """
        master_mask = np.zeros(X.shape, dtype=bool)
        u_wall = np.zeros((2,) + X.shape)
        for geom in self.geometries:
            mask = geom.get_mask(X, Y, step, rho, u)
            wx, wy = geom.get_wall_velocity()
            u_wall[0][mask] = wx
            u_wall[1][mask] = wy
            master_mask = master_mask | mask
        return master_mask, u_wall


#=======================================================================

# Flexible Cylinder (Turbine Model - Top-Down View)
class FlexibleCylinder(Geometry):
    def __init__(self, cx, cy_base, r, stiffness, damping, mass):
        self.cx = cx
        self.cy_base = cy_base
        self.r = r

        # Structural Physics Parameters
        self.k = stiffness
        self.c = damping
        self.m = mass

        # State Variables
        self.dy = 0.0  # Transverse deflection (lift axis)
        self.vy = 0.0  # Velocity
        self.last_mask = None

    def get_mask(self, X, Y, step, rho=None, u=None):
        # Calculate Fluid Force (If we have fluid data)
        if rho is not None and self.last_mask is not None:
            # Measure fluid pressure across the top and bottom of the cylinder
            top_y = min(rho.shape[1] - 1, int(self.cy_base + self.dy + self.r + 1))
            bottom_y = max(0, int(self.cy_base + self.dy - self.r - 1))
            x_range = slice(int(self.cx - self.r), int(self.cx + self.r))

            # In LBM, pressure p = rho / 3. Force is the difference across the shape.
            p_top = np.sum(rho[x_range, top_y]) / 3.0
            p_bottom = np.sum(rho[x_range, bottom_y]) / 3.0

            # Net Lift Force (pushing along the Y axis)
            F_lift = (p_bottom - p_top) * 0.1 # Scale factor

            # Structural Solver (Euler Integration for Mass-Spring System)
            acceleration = (F_lift - self.k * self.dy - self.c * self.vy) / self.m
            self.vy += acceleration
            self.dy += self.vy

            # Safety Guardrails against LBM numerical explosions (NaNs/Infs)
            max_disp = self.r * 1.5

            # The wall velocity is fed back into the fluid by the moving-wall
            # bounce-back, so an unbounded vy does not stay contained in the
            # structure: it injects momentum into the lattice and blows the
            # simulation up. vy must therefore be bounded as strictly as dy.
            # The cap is a fraction of the lattice sound speed (c_s = 1/sqrt(3));
            # a boundary moving faster than that is unphysical for LBM anyway.
            max_vel = 0.1 / np.sqrt(3.0)

            if np.isnan(self.dy) or np.isnan(self.vy):
                self.dy, self.vy = 0.0, 0.0
            else:
                self.dy = float(np.clip(self.dy, -max_disp, max_disp))
                self.vy = float(np.clip(self.vy, -max_vel, max_vel))

                # Hitting the displacement limit must also kill the velocity.
                # Clamping position while letting vy keep accumulating breaks the
                # energy balance and pumps the oscillator instead of arresting it.
                if abs(self.dy) >= max_disp:
                    self.vy = 0.0

        # Generate the moving circular mask
        current_cy = self.cy_base + self.dy
        self.last_mask = (X - self.cx)**2 + (Y - current_cy)**2 <= self.r**2
        return self.last_mask

    def get_wall_velocity(self):
        # The cylinder is rigid and translates only along the transverse axis,
        # so every surface cell shares the single structural velocity vy.
        return (0.0, self.vy)

    def get_tracking_data(self):
        # Track the absolute center coordinates AND the specific deflection dy
        return (self.cx, self.cy_base + self.dy, self.dy)
