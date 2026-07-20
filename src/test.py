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
U_LB = 0.04      # Inflow velocity in lattice units
STROUHAL = 0.2   # Strouhal number for a circular cylinder in the shedding regime
RHO_LB = 1.0     # Reference lattice density


def report_dynamics(label, size, stiffness, mass):
    """Print the structural vs. shedding frequency ratio for one turbine.

    f_n  natural frequency of the mass-spring system, cycles per lattice step
    f_s  vortex shedding frequency from the Strouhal relation, St * U / D
    U*   reduced velocity, U / (f_n * D); lock-in occurs around U* = 4-8
    m*   mass ratio, structural mass over displaced fluid mass
    """
    D = 2.0 * size
    f_n = math.sqrt(stiffness / mass) / (2.0 * math.pi)
    f_s = STROUHAL * U_LB / D
    u_star = U_LB / (f_n * D)
    m_star = mass / (RHO_LB * math.pi * size ** 2)

    flag = "" if 4.0 <= u_star <= 8.0 else "   <-- outside lock-in band"
    print(f"  [{label}] f_n={f_n:.3e}  f_s={f_s:.3e}  f_n/f_s={f_n / f_s:7.2f}  "
          f"U*={u_star:6.3f}  m*={m_star:.2e}{flag}")


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

    def run_phase_1_lock_in_sweep(self):
        """Phase 1: Sweep Wind Speeds using stable LBM parameters."""
        print("Generating Phase 1: Lock-in Sweep...")
        simulations = []

        # STABLE MATHEMATICAL PARAMETERS
        # With F_fluid scaled down to 0.01, this stiffness will
        # bend smoothly into the 15-30 range without exploding.
        stiffness = 0.0114
        mass = 0.095
        size = 15          # Cylinder radius -> D = 30 lattice units

        # The structural response is identical across the sweep; only Re varies.
        report_dynamics("P1 baseline", size, stiffness, mass)

        # Test Reynolds numbers from 150 to 350
        for re in [40, 150, 200, 250, 300, 500]:
            name = f"P1_Re_{re}"
            simulations.append({
                "name": name,
                "nx": 600, "ny": 200, "re": float(re), "steps": 6000,
                "walls": True, "flow": "uniform",
                "video": True, "video_out": f"{self.output_dir}/{name}.mp4",
                "csv": True, "csv_out": f"{self.output_dir}/{name}.csv",
                "objects": [
                    {
                        "shape": "flexible_cylinder",
                        "cx": 200, "cy": 100,   # Centred in the channel
                        "size": size,
                        "stiffness": stiffness,
                        "mass": mass,
                        "damping": 0.002
                    }
                ]
            })

        self.write_and_run("phase1_sweep.json", {"simulations": simulations})

    def run_phase_2_material_optimization(self, optimal_re=250.0):
        """Phase 2: Test real-world materials using a scaling factor."""

        # Define the real-world properties of your materials
        # E = Young's Modulus (Pa), rho = Density (kg/m^3)
        materials = {
            "PVC_Plastic": {"E": 3.0e9, "rho": 1380},
            "Fiberglass": {"E": 15.0e9, "rho": 1900},
            "Aluminum": {"E": 69.0e9, "rho": 2700},
            "Carbon_Fiber": {"E": 150.0e9, "rho": 1600}
        }

        # Define the real-world dimensions of your test turbine
        height_m = 8.0       # 8 meters tall
        width_m = 0.5        # 0.5 meters wide
        thickness_m = 0.05   # 5 cm wall thickness

        # Choose your LBM Scaling Factors (You may need to tweak these
        # slightly so the softest material doesn't bend into infinity)
        scale_k = 25e-6
        scale_m = 25e-5

        size = 15            # Cylinder radius, identical across all materials

        simulations = []

        for mat_name, props in materials.items():
            # Calculate Real-World Physics
            I = (width_m * (thickness_m ** 3)) / 12.0
            real_k = (3 * props["E"] * I) / (height_m ** 3)
            real_mass = props["rho"] * height_m * width_m * thickness_m

            # Apply Scaling Factor for the LBM Engine
            lbm_stiffness = real_k * scale_k
            lbm_mass = real_mass * scale_m

            print(f"[{mat_name}] Real k: {real_k/1000:.1f} kN/m -> LBM k: {lbm_stiffness:.5f}")
            report_dynamics(mat_name, size, lbm_stiffness, lbm_mass)

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
                        "size": size,
                        "stiffness": lbm_stiffness,  # The scaled real-world value!
                        "mass": lbm_mass             # The scaled real-world value!
                    }
                ]
            })

        self.write_and_run("phase2_real_materials.json", {"simulations": simulations})

    def run_phase_3_shear_flow_reality(self, optimal_re=250.0):
        """Phase 3: Compare ideal Uniform flow against realistic Shear flow."""
        report_dynamics("P3 baseline", 15, 0.005, 2.0)

        simulations = []
        for flow_type in ['uniform', 'shear']:
            name = f"P3_Flow_{flow_type.capitalize()}"
            simulations.append({
                "name": name,
                "nx": 600, "ny": 200, "re": optimal_re, "steps": 8000,
                "walls": True, "flow": flow_type,
                "video": True, "video_out": f"{self.output_dir}/{name}.mp4",
                "csv": True, "csv_out": f"{self.output_dir}/{name}.csv",
                "objects": [
                    {"shape": "flexible_cylinder", "cx": 200, "cy": 100, "size": 15,
                     "stiffness": 0.005, "mass": 2.0}
                ]
            })

        self.write_and_run("phase3_shear_comparison.json", {"simulations": simulations})

    def run_phase_4_wake_interference(self):
        """Phase 4: Simulate a twin-turbine wind farm layout."""
        report_dynamics("P4 array member", 15, 0.006, 2.5)

        simulations = [{
            "name": "P4_Twin_Turbines_InLine",
            "nx": 1100, "ny": 300, "re": 300.0, "steps": 12000,
            "walls": True, "flow": "shear", # Testing in realistic shear flow
            "video": True, "video_out": f"{self.output_dir}/P4_Twin_Turbines.mp4",
            "csv": True, "csv_out": f"{self.output_dir}/P4_Twin_Turbines.csv",
            # Identical cylinders spaced along the channel, each sitting in the
            # wake of the one before it. Keeping them identical means any drop in
            # downstream amplitude is attributable to the wake, not to geometry.
            "objects": [
                {"shape": "flexible_cylinder", "cx": cx, "cy": 150, "size": 15,
                 "stiffness": 0.006, "mass": 2.5}
                for cx in [200, 450, 700, 950]
            ]
        }]

        self.write_and_run("phase4_wake_farm.json", {"simulations": simulations})

    def execute_full_thesis_roadmap(self):
        print("==================================================")
        print("STARTING FULL BLADELESS TURBINE TEST SUITE")
        print("==================================================")

        self.run_phase_1_lock_in_sweep()
        self.run_phase_2_material_optimization(optimal_re=250.0)
        self.run_phase_3_shear_flow_reality(optimal_re=250.0)
        self.run_phase_4_wake_interference()

        print("\nAll experiments completed! Check the 'experiment_results' folder.")

if __name__ == "__main__":
    test_suite = BladelessTurbineTestSuite()

    # You can run individual phases for quick testing:
    #test_suite.run_phase_2_material_optimization()
    test_suite.run_phase_1_lock_in_sweep()

    # Or let it run overnight for the complete dataset:
    #test_suite.execute_full_thesis_roadmap()
