"""
./src/geometry.py

This file contains the class Geometry and all of its implementation.
There are currently 3 different type of objects implemented:
1 - Cylinder
2 - Rectangle
3 - Airfoil

Each of this geometries comes with a stationary and an oscillating implementations.
The rectangle geometry includes aswell a flexible class that allows the object 
to bend under external forces (i.e. wind). This shape will be the base model
for a bladeless turbine.

Implementation method: In order to implement any new shape it is required
                       to define a new class that inherits from the class Geometry.

TODO: (optional) implement new geometries for the blades (i.e. cone, double cone).
"""




from dataclasses import dataclass
import numpy as np



# ==========================================
# GEOMETRY MODULES
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
   

#=======================================================================


# Cylinder geometries
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
    

#=======================================================================


#Rectangle Geometries
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


#=======================================================================


#Flexible Rectangle (Turbine Model)
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
        # Calculate Fluid Force (If we have fluid data)
        if rho is not None and self.last_mask is not None:
            # Sample density (pressure) directly upstream and downstream of the base
            up_x = max(0, int(self.cx - self.w))
            down_x = min(rho.shape[0] - 1, int(self.cx + self.w + self.delta))
            y_range = slice(int(self.cy_base), int(self.cy_base + self.h))

            # In LBM, pressure p = rho / 3. Force is the difference across the shape.
            p_up = np.sum(rho[up_x, y_range]) / 3.0
            p_down = np.sum(rho[down_x, y_range]) / 3.0
            F_fluid = (p_up - p_down) * 0.1 # Scale factor to prevent vacuum explosion

            # Structural Solver (Euler Integration)
            acceleration = (F_fluid - self.k * self.delta - self.c * self.vel) / self.m
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

        # Generate the Bent Mask
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


#=======================================================================


# Airfoil Geometries (NACA 4-Digit Symmetric)
class StationaryAirfoil(Geometry):
    def __init__(self, cx, cy, chord, thickness=0.12):
        self.cx = cx
        self.cy = cy
        self.c = chord        # Length of the airfoil from tip to tail
        self.t = thickness    # Maximum thickness as a fraction of the chord (0.12 = NACA 0012)
        self.le_x = cx - chord / 2  # Leading edge X coordinate

    def get_mask(self, X, Y, step):
        # Normalize X coordinates along the chord from 0.0 to 1.0
        x_c = (X - self.le_x) / self.c
        
        # Prevent invalid square roots by clipping negative values (they will be masked out anyway)
        x_safe = np.clip(x_c, 0, 1)
        
        # The NACA symmetric airfoil thickness equation
        y_t = 5 * self.t * self.c * (
            0.2969 * np.sqrt(x_safe) - 
            0.1260 * x_safe - 
            0.3516 * x_safe**2 + 
            0.2843 * x_safe**3 - 
            0.1015 * x_safe**4
        )
        
        # A point is inside if it is within the chord length bounds AND below the thickness curve
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
