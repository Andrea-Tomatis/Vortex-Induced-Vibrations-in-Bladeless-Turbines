"""
/src/test.py

This is the automatic testing facility. Here all the test infrastructure is contatined.
You can run all of the test (if you have a lot of time) or run each one individually by calling the
method test_suite.run_phase_x_...() in the main function.

All phases drive the top-down FlexibleCylinder model. The mast is represented by its
horizontal cross-section (radius `size`) free to translate along the transverse (lift)
axis, so the cantilever's `width`/`height` pair is replaced by a single radius and the
object is centred in the channel instead of being anchored to the bottom wall.

NOTE: report_dynamics() prints the frequency ratio for every configuration BEFORE the
      batch runs. If the reduced velocity U* falls outside the lock-in band (roughly
      4-8) the structure is too stiff or too heavy to resonate and the sweep will
      produce a flat amplitude curve. Check that output before committing to a long run.

TODO: (ALREADY SOLVED) THe scaling for the real material stiffness and mass is not tuned perfectly. This phenomenon
      results in the turbine maximum displacement to be cut by threshold of 50. This can be seen
      in /analysis_plots/Phase2_Material_Comparison where the plots look like square functions
      eventhouh they should look like sine(ish) functions.
      Hint: probably the best way to solve this is to tune the scale_k and scale_m parameters in
      the run_phase_2_material_optimization(). The report explains what those parameters do.

"""


import os
import json
import math
from main import run_batch_from_json  # Ensure your previous script is named main.py

# Lattice-unit constants shared with SimConfig
U_LB = 0.02      # Inflow velocity in lattice units
RHO_LB = 1.0     # Reference lattice density


def strouhal(Re):
    """Strouhal number for a circular cylinder (Roshko empirical fit).

    St is NOT constant: it climbs from ~0.13 near the shedding onset to an
    asymptote around 0.21. Because uLB is fixed in this framework, this
    Re-dependence is the ONLY thing that moves the shedding frequency during a
    Reynolds sweep, so it must not be approximated as a constant 0.2.
    """
    if Re <= 47.0:
        return 0.0      # below the onset of vortex shedding: steady wake
    return 0.212 * (1 - 21.2 / Re) if Re < 150 else 0.212 * (1 - 12.7 / Re)


def solve_mass_for_reduced_velocity(size, stiffness, u_star):
    """Return the mass that places the turbine at the requested U*.

    U* = U / (f_n * D) with f_n = sqrt(k/m) / 2*pi, so
    m = k / (2*pi*f_n)^2 where f_n = U / (U* * D).
    """
    D = 2.0 * size
    f_n = U_LB / (u_star * D)
    return stiffness / ((2.0 * math.pi * f_n) ** 2)


def report_dynamics(label, size, stiffness, mass, Re=None):
    """Print the structural vs. shedding frequency ratio for one turbine.

    f_n  natural frequency of the mass-spring system, cycles per lattice step
    f_s  vortex shedding frequency from the Strouhal relation, St(Re) * U / D
    U*   reduced velocity, U / (f_n * D); lock-in occurs around U* = 4-8
    m*   mass ratio, structural mass over displaced fluid mass
    """
    D = 2.0 * size
    f_n = math.sqrt(stiffness / mass) / (2.0 * math.pi)
    u_star = U_LB / (f_n * D)
    m_star = mass / (RHO_LB * math.pi * size ** 2)

    flag = "" if 4.0 <= u_star <= 8.0 else "   <-- outside lock-in band"
    print(f"  [{label}] f_n={f_n:.3e}  U*={u_star:6.3f}  m*={m_star:6.2f}{flag}")

    # Resonance is governed by f_s/f_n = St(Re) * U*, so it can only be
    # reported per Reynolds number, not once for the whole sweep.
    if Re is not None:
        ratios = "  ".join(f"Re{r}:{strouhal(r) * u_star:.2f}" for r in Re)
        print(f"      f_s/f_n across sweep -> {ratios}")


class BladelessTurbineTestSuite:
    def __init__(self, output_dir="../results"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def write_and_run(self, filename, data):
        """Saves the generated JSON dictionary to a file and runs it."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n[Test Suite] Generated {filename}. Launching batch runner...")
        run_batch_from_json(filepath)

    # def run_phase_1_lock_in_sweep(self):
    #     """Phase 1: Sweep Reynolds numbers covering Steady State to High-Re Unsteady flow."""
    #     print("Generating Phase 1: Lock-in & Transition Sweep...")
    #     simulations = []

    #     # Structural parameters
    #     stiffness = 0.0114
    #     size = 15          # Cylinder radius -> D = 30 lattice units
    #     mass = solve_mass_for_reduced_velocity(size, stiffness, u_star=5.89)

    #     # Map each Reynolds number to appropriate step counts, domain sizes, and video intervals
    #     re_configs = {
    #         35:  {"steps": 8000,  "nx": 600,  "ny": 200},   # Steady state (rapid convergence)
    #         150: {"steps": 8000,  "nx": 600,  "ny": 200},   # Laminar Von Karman Street
    #         250: {"steps": 12000, "nx": 700,  "ny": 250},   # Peak Structural Lock-In
    #         500: {"steps": 15000, "nx": 900,  "ny": 300},   # Shear-layer roll-up transition
    #         800: {"steps": 18000, "nx": 1000, "ny": 300}    # Highly unsteady 2D turbulent wake
    #     }

    #     re_points = list(re_configs.keys())
    #     report_dynamics("P1 baseline", size, stiffness, mass, Re=re_points)

    #     for re, cfg in re_configs.items():
    #         name = f"P1_Re_{re}"
    #         steps = cfg["steps"]
    #         nx = cfg["nx"]
    #         ny = cfg["ny"]
            
    #         # Keep video frames to ~400-600 rendered frames (~15-20s playback at 30fps)
    #         video_interval = max(5, steps // 1200)

    #         simulations.append({
    #             "name": name,
    #             "nx": nx, 
    #             "ny": ny, 
    #             "re": float(re), 
    #             "steps": steps,
    #             "walls": False,             # Open boundaries to prevent unnatural wall stabilization
    #             "flow": "uniform",
    #             "video": True, 
    #             "video_out": f"{self.output_dir}/{name}.mp4",
    #             "video_interval": video_interval,
    #             "csv": True, 
    #             "csv_out": f"{self.output_dir}/{name}.csv",
    #             "objects": [
    #                 {
    #                     "shape": "flexible_cylinder",
    #                     "cx": int(nx * 0.25),  # Dynamically position cylinder at 25% domain width
    #                     "cy": int(ny * 0.50),  # Centred in the channel
    #                     "size": size,
    #                     "stiffness": stiffness,
    #                     "mass": mass,
    #                     "damping": 0.002
    #                 }
    #             ]
    #         })

    #     self.write_and_run("phase1_sweep.json", {"simulations": simulations})
    
    # New version:
    def run_phase_1_lock_in_sweep(self):
        """Phase 1: Sweep Reynolds numbers covering Steady State to High-Re Unsteady flow."""
        print("Generating Phase 1: Lock-in & Transition Sweep (Baseline PVC)...")
        simulations = []

        size = 15          
        stiffness = 0.0000355 
        mass = 50.0        

        re_configs = {
            35:  {"steps": 20000, "nx": 600},   
            150: {"steps": 20000, "nx": 600},   
            250: {"steps": 20000, "nx": 800},   
            500: {"steps": 20000, "nx": 1000},  
            800: {"steps": 20000, "nx": 1200}   
        }

        re_points = list(re_configs.keys())
        report_dynamics("P1 baseline PVC", size, stiffness, mass, Re=re_points)

        for re, cfg in re_configs.items():
            name = f"P1_Re_{re}"
            steps = cfg["steps"]
            nx = cfg["nx"]
            ny = 200 
            
            video_interval = max(5, steps // 800) 

            simulations.append({
                "name": name,
                "nx": nx, 
                "ny": ny, 
                "re": float(re), 
                "steps": steps,
                "walls": True,              
                "flow": "uniform",
                "video": True, 
                "video_out": f"{self.output_dir}/{name}.mp4",
                "video_interval": video_interval,
                "csv": True, 
                "csv_out": f"{self.output_dir}/{name}.csv",
                "objects": [
                    {
                        "shape": "flexible_cylinder",
                        "cx": 200,             
                        "cy": int(ny * 0.50),  
                        "size": size,
                        "stiffness": stiffness,
                        "mass": mass,
                        "damping": 0.002
                    }
                ]
            })

        self.write_and_run("phase1_sweep.json", {"simulations": simulations})


    def run_phase_2_material_optimization(self, optimal_re=250.0):
        """Phase 2: Test real-world materials using proportional non-dimensional scaling."""

        # Real-world values for tested materials 
        # E = Young's Modulus (Pa), rho = Density (kg/m^3)
        materials = {
            "PVC_Plastic": {"E": 3.0e9, "rho": 1380},
            # "Fiberglass": {"E": 15.0e9, "rho": 1900},
            # "Aluminum": {"E": 69.0e9, "rho": 2700},
            # "Carbon_Fiber": {"E": 150.0e9, "rho": 1600}
        }

        # Real-world dimensions of the hollow cylindrical mast
        height_m = 8.0       
        d_out_m = 0.5        # outer diameter
        thickness_m = 0.05   # wall thickness
        d_in_m = d_out_m - (2 * thickness_m) # inner diameter

        size_lbm = 15        # Cylinder radius in LBM pixels

        # Calculate exact geometric properties for a hollow cylinder
        area = (math.pi / 4.0) * (d_out_m**2 - d_in_m**2)
        I = (math.pi / 64.0) * (d_out_m**4 - d_in_m**4)

        # To keep LBM numerically stable, we pick a baseline material (PVC) 
        # and define our safe LBM limits for that specific material.
        # All other materials will be scaled relative to this baseline.
        base_mat = "PVC_Plastic"
        base_real_k = (3 * materials[base_mat]["E"] * I) / (height_m ** 3)
        base_real_mass = materials[base_mat]["rho"] * area * height_m

        # Define stable LBM values for the baseline material 
        lbm_target_k = 0.0005   
        lbm_target_mass = 50.0 

        # Establish global scaling factors that apply to ALL materials uniformly
        scale_k = lbm_target_k / base_real_k
        scale_m = lbm_target_mass / base_real_mass

        simulations = []

        for mat_name, props in materials.items():
            # 1. Real-World Physics (Cantilever beam mechanics)
            real_k = (3 * props["E"] * I) / (height_m ** 3)
            real_mass = props["rho"] * area * height_m
            
            # 2. Apply Proportional LBM Scaling
            lbm_stiffness = real_k * scale_k
            lbm_mass = real_mass * scale_m

            # Calculate natural frequency to verify scaling (fn = 1/(2*pi) * sqrt(k/m))
            real_fn = (1 / (2 * math.pi)) * math.sqrt(real_k / real_mass)
            lbm_fn = (1 / (2 * math.pi)) * math.sqrt(lbm_stiffness / lbm_mass)

            print(f"[{mat_name}] Real k: {real_k/1000:.1f} kN/m, Mass: {real_mass:.1f} kg")
            print(f"      -> LBM k: {lbm_stiffness:.5f}, LBM Mass: {lbm_mass:.5f}")
            print(f"      -> Freq mapping: Real fn={real_fn:.2f} Hz -> LBM fn={lbm_fn:.4f}")
            
            # Assuming report_dynamics is an external function you built
            # report_dynamics(mat_name, size_lbm, lbm_stiffness, lbm_mass)

            name = f"P2_{mat_name}"
            simulations.append({
                "name": name,
                "nx": 600, "ny": 200, "re": optimal_re, "steps": 8000,
                "walls": True, "flow": "uniform",
                "video": True, "video_out": f"{self.output_dir}/{name}.mp4",
                "csv": True, "csv_out": f"{self.output_dir}/{name}.csv",
                "objects": [
                    {
                        "shape": "flexible_cylinder",
                        "cx": 200, "cy": 100,
                        "size": size_lbm,
                        "stiffness": lbm_stiffness,
                        "mass": lbm_mass 
                    }
                ]
            })

        self.write_and_run("phase2_real_materials.json", {"simulations": simulations})

    def run_phase_4_wake_interference(self, optimal_re=250.0):
        """Phase 4: Simulate a twin-turbine wind farm layout using scaled PVC material properties."""
        
        # Material from phase 2 
        materials = {
            "PVC_Plastic": {"E": 3.0e9, "rho": 1380}
        }

        # Real-world measurements of turbine 
        height_m = 8.0        
        d_out_m = 0.5         
        thickness_m = 0.05    
        d_in_m = d_out_m - (2 * thickness_m)

        size_lbm = 15         # Cilinder radius in LBM pixels

        # Geometric calculations 
        area = (math.pi / 4.0) * (d_out_m**2 - d_in_m**2)
        I = (math.pi / 64.0) * (d_out_m**4 - d_in_m**4)

        base_mat = "PVC_Plastic"
        base_real_k = (3 * materials[base_mat]["E"] * I) / (height_m ** 3)
        base_real_mass = materials[base_mat]["rho"] * area * height_m

        # LBM target values from test phase 2
        lbm_target_k = 0.0005   
        lbm_target_mass = 50.0 

        scale_k = lbm_target_k / base_real_k
        scale_m = lbm_target_mass / base_real_mass

        props = materials[base_mat]
        real_k = (3 * props["E"] * I) / (height_m ** 3)
        real_mass = props["rho"] * area * height_m
        
        lbm_stiffness = real_k * scale_k
        lbm_mass = real_mass * scale_m

        real_fn = (1 / (2 * math.pi)) * math.sqrt(real_k / real_mass)
        lbm_fn = (1 / (2 * math.pi)) * math.sqrt(lbm_stiffness / lbm_mass)

        print(f"[P4 PVC array] Real k: {real_k/1000:.1f} kN/m, Mass: {real_mass:.1f} kg")
        print(f"    -> LBM k: {lbm_stiffness:.5f}, LBM Mass: {lbm_mass:.5f}")
        print(f"    -> Freq mapping: Real fn={real_fn:.2f} Hz -> LBM fn={lbm_fn:.4f}")

        # 4 PVC turbines in ine 
        simulations = [{
            "name": "P4_Twin_Turbines_InLine",
            "nx": 1200, 
            "ny": 200, 
            "re": optimal_re, 
            "steps": 12000,
            "walls": True, 
            "flow": "uniform",
            "video": True, 
            "video_out": f"{self.output_dir}/P4_Twin_Turbines.mp4",
            "csv": True, 
            "csv_out": f"{self.output_dir}/P4_Twin_Turbines_InLine.csv",
            
            "objects": [
                {
                    "shape": "flexible_cylinder", 
                    "cx": cx, 
                    "cy": 150, 
                    "size": size_lbm,
                    "stiffness": lbm_stiffness, 
                    "mass": lbm_mass
                }
                for cx in [200, 450, 700, 950] # 4 positions 
            ]
        }]

        self.write_and_run("phase4_wake_farm.json", {"simulations": simulations})

    def execute_full_thesis_roadmap(self):
        print("==================================================")
        print("STARTING FULL BLADELESS TURBINE TEST SUITE")
        print("==================================================")

        self.run_phase_1_lock_in_sweep()
        self.run_phase_2_material_optimization(optimal_re=250.0)
        self.run_phase_4_wake_interference()

        print("\nAll experiments completed! Check the 'experiment_results' folder.")

if __name__ == "__main__":
    test_suite = BladelessTurbineTestSuite()

    # You can run individual phases for quick testing:
    # test_suite.run_phase_2_material_optimization()
    test_suite.run_phase_1_lock_in_sweep()
    # test_suite.run_phase_4_wake_interference()

    # Or let it run overnight for the complete dataset:
    #test_suite.execute_full_thesis_roadmap()
