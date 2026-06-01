"""
/src/analysis.py

This file contains the data processing regarding the simulation results. 
It generate multiple plots and files.

TODO: It would be nice to write better plotting solutions for phase 2 and 3. The implemented ones
      are not very clear to read.
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

class TurbineDataAnalyzer:
    def __init__(self, data_dir="../results", output_dir="../analysis_plots"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create the output directory for graphs if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Use a professional plotting style
        plt.style.use('seaborn-v0_8-darkgrid')

    def analyze_phase_1_resonance(self):
        """Phase 1: Plots the Lock-in Resonance Curve (Amplitude vs. Wind Speed)"""
        print("Analyzing Phase 1: Lock-in Sweep...")
        
        # Find all Phase 1 CSV files
        files = glob.glob(os.path.join(self.data_dir, "P1_Re_*.csv"))
        if not files:
            print("No Phase 1 files found. Skipping.")
            return

        re_values = []
        max_amplitudes = []

        for file in files:
            # Extract the Reynolds number from the filename (e.g., P1_Re_150.csv -> 150)
            re = float(file.split('P1_Re_')[-1].replace('.csv', ''))
            df = pd.read_csv(file)
            
            # We ignore the first 1000 steps to let the fluid "spin up" and settle
            steady_state = df[df['Step'] > 1000]
            
            # Find the maximum absolute deflection
            max_amp = steady_state['Deflection_dX'].abs().max()
            
            re_values.append(re)
            max_amplitudes.append(max_amp)

        # Sort the data mathematically
        sorted_indices = np.argsort(re_values)
        re_values = np.array(re_values)[sorted_indices]
        max_amplitudes = np.array(max_amplitudes)[sorted_indices]

        # Plot the Resonance Bell Curve
        plt.figure(figsize=(10, 6))
        plt.plot(re_values, max_amplitudes, marker='o', linestyle='-', linewidth=2, color='b')
        plt.title('Phase 1: Lock-in Resonance Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Wind Speed (Reynolds Number)', fontsize=12)
        plt.ylabel('Maximum Tip Deflection ($\Delta X$)', fontsize=12)
        plt.fill_between(re_values, max_amplitudes, alpha=0.2, color='b')
        plt.ylim(max(max_amplitudes)*0.8, max(max_amplitudes)*1.2)
        
        # Save and close
        output_path = os.path.join(self.output_dir, "Phase1_Resonance_Curve.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

    def analyze_phase_2_materials(self):
        """Phase 2: Compares material deflection and calculates Mechanical Power Proxy."""
        print("Analyzing Phase 2: Material Power Efficiency...")
        
        files = glob.glob(os.path.join(self.data_dir, "P2_*.csv"))
        if not files:
            print("No Phase 2 files found. Skipping.")
            return

        plt.figure(figsize=(12, 6))
        
        # Store results for a performance leaderboard
        performance_data = []
        
        for file in files:
            mat_name = os.path.basename(file).replace('P2_', '').replace('.csv', '')
            df = pd.read_csv(file)
            
            # Analyze steady-state oscillation (ignore startup)
            plot_data = df[(df['Step'] > 2000) & (df['Step'] < 8000)].copy()
            
            if plot_data.empty:
                continue

            # 1. Calculate Amplitude (A)
            # We take the root mean square (RMS) or simple mean of the absolute peaks
            amplitude = plot_data['Deflection_dX'].abs().mean()
            
            # 2. Calculate Frequency (f)
            # We count how many times the pole crosses the zero-axis (center point)
            zero_crossings = np.where(np.diff(np.sign(plot_data['Deflection_dX'])))[0]
            # Two zero crossings equal one full wave cycle
            num_cycles = len(zero_crossings) / 2.0
            
            # Frequency = Cycles per step (or time)
            time_steps = plot_data['Step'].max() - plot_data['Step'].min()
            frequency = num_cycles / time_steps if time_steps > 0 else 0
            
            # 3. Calculate the Power Proxy (A^2 * f^2)
            power_proxy = (amplitude ** 2) * (frequency ** 2)
            
            performance_data.append({
                'Material': mat_name,
                'Amplitude': amplitude,
                'Frequency': frequency * 1000, # Scaled up for readability
                'Power_Proxy': power_proxy
            })
            
            plt.plot(plot_data['Step'], plot_data['Deflection_dX'], label=mat_name, linewidth=1.5)

        # Plot Formatting
        plt.title('Phase 2: Material Deflection Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Simulation Step', fontsize=12)
        plt.ylabel('Tip Deflection ($\Delta X$)', fontsize=12)
        plt.legend(loc='upper right')
        
        output_path = os.path.join(self.output_dir, "Phase2_Material_Comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print the Efficiency Leaderboard
        print("\n--- PHASE 2: MATERIAL EFFICIENCY LEADERBOARD ---")
        # Sort by best power output
        performance_data.sort(key=lambda x: x['Power_Proxy'], reverse=True)
        
        for rank, data in enumerate(performance_data):
            print(f"{rank+1}. {data['Material']}")
            print(f"    Amplitude: {data['Amplitude']:.2f}")
            print(f"    Frequency (Scaled): {data['Frequency']:.2f}")
            print(f"    Relative Power Score: {data['Power_Proxy']:.4f}\n")
            
        print(f"Saved {output_path}")
        leaderboard_df = pd.DataFrame(performance_data)
        
        # Reorder columns so they look nice in Excel
        leaderboard_df = leaderboard_df[['Material', 'Power_Proxy', 'Amplitude', 'Frequency']]
        
        csv_output_path = os.path.join(self.output_dir, "Phase2_Efficiency_Leaderboard.csv")
        leaderboard_df.to_csv(csv_output_path, index=False)
        
        print(f"Saved numerical leaderboard to: {csv_output_path}")

    def analyze_phase_3_shear_flow(self):
        """Phase 3: Overlays Uniform vs. Shear flow and calculates performance drop."""
        print("Analyzing Phase 3: Flow Profile Reality Check...")
        
        files = glob.glob(os.path.join(self.data_dir, "P3_Flow_*.csv"))
        if len(files) < 2:
            print("Could not find both Uniform and Shear files. Skipping.")
            return

        plt.figure(figsize=(12, 6))
        colors = {'Uniform': '#2ca02c', 'Shear': '#d62728'}
        
        # Dictionary to hold the math for both flows
        performance_metrics = {}

        for file in files:
            flow_type = os.path.basename(file).replace('P3_Flow_', '').replace('.csv', '')
            df = pd.read_csv(file)
            
            # Look at a specific steady-state window
            plot_data = df[(df['Step'] > 4000) & (df['Step'] < 8000)].copy()
            
            if plot_data.empty:
                continue

            # 1. Calculate the Math (Same as Phase 2)
            amplitude = plot_data['Deflection_dX'].abs().mean()
            
            zero_crossings = np.where(np.diff(np.sign(plot_data['Deflection_dX'])))[0]
            num_cycles = len(zero_crossings) / 2.0
            time_steps = plot_data['Step'].max() - plot_data['Step'].min()
            frequency = (num_cycles / time_steps * 1000) if time_steps > 0 else 0
            
            power_proxy = (amplitude ** 2) * (frequency ** 2)
            
            # Store it for comparison
            performance_metrics[flow_type] = {
                'Amplitude': amplitude,
                'Frequency': frequency,
                'Power_Proxy': power_proxy
            }
            
            # Plot the line
            plt.plot(plot_data['Step'], plot_data['Deflection_dX'], 
                     label=f"{flow_type} Flow", color=colors.get(flow_type, 'blue'), linewidth=2)

        # Format the Graph
        plt.title('Phase 3: Atmospheric Boundary Layer Impact (Shear vs Uniform)', fontsize=14, fontweight='bold')
        plt.xlabel('Simulation Step', fontsize=12)
        plt.ylabel('Tip Deflection ($\Delta X$)', fontsize=12)
        plt.legend()
        
        output_path = os.path.join(self.output_dir, "Phase3_Shear_vs_Uniform.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # ==========================================
        # PRINT THE QUANTITATIVE INSIGHTS
        # ==========================================
        print("\n--- PHASE 3: ATMOSPHERIC BOUNDARY LAYER IMPACT ---")
        
        if 'Uniform' in performance_metrics and 'Shear' in performance_metrics:
            u_data = performance_metrics['Uniform']
            s_data = performance_metrics['Shear']
            
            # Calculate Percentage Drops
            amp_drop = (1.0 - (s_data['Amplitude'] / u_data['Amplitude'])) * 100
            pow_drop = (1.0 - (s_data['Power_Proxy'] / u_data['Power_Proxy'])) * 100
            
            print(f"[IDEAL] Uniform Flow:")
            print(f"    Amplitude:   {u_data['Amplitude']:.2f}")
            print(f"    Frequency:   {u_data['Frequency']:.2f}")
            print(f"    Power Score: {u_data['Power_Proxy']:.4f}\n")
            
            print(f"[REALITY] Shear Flow (Boundary Layer):")
            print(f"    Amplitude:   {s_data['Amplitude']:.2f}")
            print(f"    Frequency:   {s_data['Frequency']:.2f}")
            print(f"    Power Score: {s_data['Power_Proxy']:.4f}\n")
            
            print(f"-> IMPACT CONCLUSION:")
            print(f"   Moving from a wind tunnel to real-world shear flow resulted in a")
            print(f"   {amp_drop:.1f}% reduction in amplitude and a {pow_drop:.1f}% drop in total power.")
            
        print(f"\nSaved Phase 3 plot to {output_path}")

    def analyze_phase_4_wake_interference(self):
        """Phase 4: Plots the synchronization of ANY number of twin/array turbines."""
        print("Analyzing Phase 4: Wake Interference (Wind Farm)...")
        
        file = os.path.join(self.data_dir, "P4_Twin_Turbines_InLine.csv")
        if not os.path.exists(file):
            file = os.path.join(self.data_dir, "phase4_wake_farm.csv") # Alternate generic name
            if not os.path.exists(file):
                # Try to find any Phase 4 file
                files = glob.glob(os.path.join(self.data_dir, "P4_*.csv"))
                if not files:
                    print("No Phase 4 wind farm file found. Skipping.")
                    return
                file = files[0] # Just grab the first one we find

        df = pd.read_csv(file)
        
        # 1. Dynamically find ALL flexible poles in the CSV
        turbines = [name for name in df['Obj_Name'].unique() if 'FlexibleCantilever' in name]
        
        if not turbines:
            print("No flexible poles found in the data.")
            return
            
        # Sort them numerically so _0 is first, _1 is second, etc.
        turbines.sort(key=lambda x: int(x.split('_')[-1]))
        
        num_turbines = len(turbines)
        print(f"Detected {num_turbines} turbines in the simulation array.")

        # Focus on the end of the simulation where flutter is fully developed
        window_start = df['Step'].max() - 4000
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Generate a dynamic color palette so every turbine gets a distinct color
        colors = plt.cm.viridis(np.linspace(0, 0.9, num_turbines))
        
        front_mean_amp = None
        
        print(f"--- Phase 4 Array Insights ---")
        
        for idx, t_name in enumerate(turbines):
            # Extract data for this specific turbine
            t_data = df[df['Obj_Name'] == t_name]
            t_plot = t_data[t_data['Step'] > window_start]
            
            # Labeling
            label = "Turbine 0 (Upstream)" if idx == 0 else f"Turbine {idx} (Wake)"
            
            # Top Plot: Overlaid waveforms
            line_style = '-' if idx == 0 else '--' # Dash downstream turbines for clarity
            ax1.plot(t_plot['Step'], t_plot['Deflection_dX'], label=label, color=colors[idx], linestyle=line_style)
            
            # Bottom Plot: Moving Average of Absolute Amplitude (Power Proxy)
            power_proxy = t_plot['Deflection_dX'].abs().rolling(window=20).mean()
            ax2.plot(t_plot['Step'], power_proxy, label=label + ' Energy', color=colors[idx])
            
            # Calculate and Print Efficiency metrics
            mean_amp = t_plot['Deflection_dX'].abs().mean()
            if idx == 0:
                front_mean_amp = mean_amp
                print(f"[{label}] Mean Amplitude: {mean_amp:.3f}")
            else:
                efficiency = mean_amp / front_mean_amp if front_mean_amp > 0 else 0
                print(f"[{label}] Mean Amplitude: {mean_amp:.3f} | Efficiency Ratio: {efficiency:.2f}x")

        # Formatting
        ax1.set_title(f'Phase 4: Wake-Induced Flutter Synchronization ({num_turbines}-Turbine Array)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Deflection ($\Delta X$)')
        ax1.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

        ax2.set_xlabel('Simulation Step')
        ax2.set_ylabel('Mean Absolute Deflection')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "Phase4_Wind_Farm_Array.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

    def execute_all(self):
        print("==================================================")
        print("STARTING DATA ANALYSIS PIPELINE")
        print("==================================================")
        self.analyze_phase_1_resonance()
        self.analyze_phase_2_materials()
        self.analyze_phase_3_shear_flow()
        self.analyze_phase_4_wake_interference()
        print("\nAll analysis complete! Check the 'analysis_plots' folder.")

if __name__ == "__main__":
    analyzer = TurbineDataAnalyzer()
    analyzer.execute_all()