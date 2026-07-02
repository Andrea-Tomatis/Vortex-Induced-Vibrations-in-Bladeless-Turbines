"""
/src/test.py

This is the automatic testing facility. Here all the test infrastructure is contatined.
You can run all of the test (if you have a lot of time) or run each one individually by calling the
method test_suite.run_phase_x_...() in the main function.

TODO: (ALREADY SOLVED) THe scaling for the real material stiffness and mass is not tuned perfectly. This phenomenon
      results in the turbine maximum displacement to be cut by threshold of 50. This can be seen
      in /analysis_plots/Phase2_Material_Comparison where the plots look like square functions
      eventhouh they should look like sine(ish) functions.
      Hint: probably the best way to solve this is to tune the scale_k and scale_m parameters in
      the run_phase_2_material_optimization(). The report explains what those parameters do.

"""


import os
import json
from main import run_batch_from_json  # Ensure your previous script is named main.py

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
        
        # Test Reynolds numbers from 150 to 350
        for re in [150, 200, 250, 300, 350]:
            name = f"P1_Re_{re}"
            simulations.append({
                "name": name,
                "nx": 600, "ny": 200, "re": float(re), "steps": 6000, 
                "walls": True, "flow": "uniform",
                "video": True, "video_out": f"{self.output_dir}/{name}.mp4",
                "csv": True, "csv_out": f"{self.output_dir}/{name}.csv",
                "objects": [
                    {
                        "shape": "flexible_pole", "cx": 200, "cy": 1, 
                        "width": 10, "height": 100, 
                        
                        # STABLE MATHEMATICAL PARAMETERS
                        # With F_fluid scaled down to 0.01, this stiffness will 
                        # bend smoothly into the 15-30 range without exploding.
                        "stiffness": 0.0114,  
                        "mass": 0.095 ,
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
        
        simulations = []
        
        for mat_name, props in materials.items():
            # Calculate Real-World Physics
            I = (width_m * (thickness_m ** 3)) / 12.0
            real_k = (3 * props["E"] * I) / (height_m ** 3)
            real_mass = props["rho"] * height_m * width_m * thickness_m
            
            # Apply Scaling Factor for the LBM Engine
            lbm_stiffness = real_k * scale_k
            lbm_mass = real_mass * scale_m

            print(lbm_mass, lbm_stiffness)
            
            print(f"[{mat_name}] Real k: {real_k/1000:.1f} kN/m -> LBM k: {lbm_stiffness:.5f}")
            
            name = f"P2_{mat_name}"
            simulations.append({
                "name": name,
                "nx": 600, "ny": 200, "re": optimal_re, "steps": 8000, 
                "walls": True, "flow": "uniform",
                "video": True, "video_out": f"{self.output_dir}/{name}.mp4",
                "csv": True, "csv_out": f"{self.output_dir}/{name}.csv",
                "objects": [
                    {
                        "shape": "flexible_pole", "cx": 200, "cy": 1, 
                        "width": 10, "height": 100, 
                        "stiffness": lbm_stiffness,  # The scaled real-world value!
                        "mass": lbm_mass             # The scaled real-world value!
                    }
                ]
            })
            
        self.write_and_run("phase2_real_materials.json", {"simulations": simulations})

    def run_phase_3_shear_flow_reality(self, optimal_re=250.0):
        """Phase 3: Compare ideal Uniform flow against realistic Shear flow."""
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
                    {"shape": "flexible_pole", "cx": 200, "cy": 1, "width": 10, "height": 100, 
                     "stiffness": 0.005, "mass": 2.0}
                ]
            })
            
        self.write_and_run("phase3_shear_comparison.json", {"simulations": simulations})

    def run_phase_4_wake_interference(self):
        """Phase 4: Simulate a twin-turbine wind farm layout."""
        simulations = [{
            "name": "P4_Twin_Turbines_InLine",
            "nx": 1100, "ny": 300, "re": 300.0, "steps": 12000, 
            "walls": True, "flow": "shear", # Testing in realistic shear flow
            "video": True, "video_out": f"{self.output_dir}/P4_Twin_Turbines.mp4",
            "csv": True, "csv_out": f"{self.output_dir}/P4_Twin_Turbines.csv",
            "objects": [
                {"shape": "flexible_pole", "cx": 200, "cy": 1, "width": 10, "height": 100, 
                 "stiffness": 0.006, "mass": 2.5},
                # Second turbine placed in the wake of the first
                {"shape": "flexible_pole", "cx": 450, "cy": 1, "width": 15, "height": 150, 
                 "stiffness": 0.006, "mass": 2.5},
                 {"shape": "flexible_pole", "cx": 700, "cy": 1, "width": 15, "height": 150, 
                 "stiffness": 0.006, "mass": 2.5},
                 {"shape": "flexible_pole", "cx": 950, "cy": 1, "width": 15, "height": 150, 
                 "stiffness": 0.006, "mass": 2.5}
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