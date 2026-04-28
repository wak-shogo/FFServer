#!/usr/bin/env python
# coding: utf-8

import os
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.ndimage import uniform_filter1d

# ASE
from ase.io import read, write
from ase.neighborlist import neighbor_list

# MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis import rdf

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

PROJECTS_DIR = "FFServer/simulation_projects"
TRAJ_FILENAME = "trajectory.xyz"
SMOOTHED_FILENAME = "trajectory_smoothed.xyz"

# Smoothing
WINDOW_SIZE = 50  # Frames for moving average

# Analysis Params
FRAME_STEP = 5    # Stride for dynamics analysis
CUTOFF_B_O = 2.5  # Bond cutoff (Angstrom)
CUTOFF_A_O = 3.2
RDF_RANGE = (0.0, 10.0)
RDF_TEMP_BIN_SIZE = 100 # K

# Element Definitions (Extend as needed)
B_SITES = ["Fe", "Ru", "Sn", "Ti", "Zr", "Hf", "Pb", "Ir", "Mn", "Al", "Sc", "Ga", "Co", "Ni", "V", "Cr", "Zn", "Nb", "Ta", "Mo", "W"]
A_SITES = ["Bi", "Ca", "Sr", "Ba", "La", "Pb", "Na", "K", "Mg", "Li", "Y", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
OXYGEN  = ["O"]

# =============================================================================
# 1. Trajectory Smoothing
# =============================================================================

def smooth_trajectory(project_path, traj_path):
    """Applies moving average smoothing to the trajectory."""
    output_path = os.path.join(project_path, SMOOTHED_FILENAME)
    
    # Check file size to avoid OOM for very large trajectories
    file_size_gb = os.path.getsize(traj_path) / (1024**3)
    if file_size_gb > 0.5:
        print(f"    [Skip] Trajectory is too large ({file_size_gb:.2f} GB) for in-memory smoothing. Using original.")
        return traj_path

    print(f"    [Smoothing] Reading {os.path.basename(traj_path)}...")
    try:
        traj = read(traj_path, index=':')
        n_frames = len(traj)
        
        if n_frames < WINDOW_SIZE:
            print(f"    [Skip] Too few frames ({n_frames}) for smoothing window ({WINDOW_SIZE}). Using original.")
            return traj_path

        positions = np.array([atoms.get_positions() for atoms in traj])
        cells = np.array([atoms.get_cell()[:] for atoms in traj])

        smooth_positions = uniform_filter1d(positions, size=WINDOW_SIZE, axis=0, mode='nearest')
        smooth_cells = uniform_filter1d(cells, size=WINDOW_SIZE, axis=0, mode='nearest')

        new_traj = []
        for i in range(n_frames):
            atoms = traj[i].copy()
            atoms.set_positions(smooth_positions[i])
            atoms.set_cell(smooth_cells[i])
            new_traj.append(atoms)

        write(output_path, new_traj)
        print(f"    [Smoothing] Saved: {SMOOTHED_FILENAME}")
        return output_path

    except Exception as e:
        print(f"    [Error] Smoothing failed: {e}")
        return traj_path

# =============================================================================
# 2. Dynamics Analysis (Bond Lengths & Angles vs Temp)
# =============================================================================

def load_temps(project_path, step=1):
    """Loads temperature data from summary csv."""
    csvs = glob.glob(os.path.join(project_path, "*summary_full*.csv"))
    if not csvs:
        # Fallback: try to find any csv with 'set_temps'
        csvs = glob.glob(os.path.join(project_path, "*.csv"))
    
    for c in csvs:
        try:
            df = pd.read_csv(c)
            if 'set_temps' in df.columns:
                return df['set_temps'].values[::step]
        except:
            continue
    return None

def analyze_dynamics_frame(atoms, present_A, present_B):
    res = {'A-Site_Bond': {el: [] for el in present_A}, 
           'B-Site_Bond': {el: [] for el in present_B}, 
           'Tilt_Angle': {}}
    symbols = np.array(atoms.get_chemical_symbols())

    # A-O Bonds
    if present_A:
        i_idx, j_idx, dists = neighbor_list('ijd', atoms, CUTOFF_A_O)
        for k in range(len(i_idx)):
            s1, s2 = symbols[i_idx[k]], symbols[j_idx[k]]
            if s2 == 'O' and s1 in present_A: res['A-Site_Bond'][s1].append(dists[k])
    
    # B-O Bonds & Angles
    if present_B:
        i_idx, j_idx, dists = neighbor_list('ijd', atoms, CUTOFF_B_O)
        oxy_neighbors = {i: [] for i, s in enumerate(symbols) if s == 'O'}
        
        for k in range(len(i_idx)):
            idx_i, idx_j = i_idx[k], j_idx[k]
            s1, s2 = symbols[idx_i], symbols[idx_j]
            if s2 == 'O' and s1 in present_B:
                res['B-Site_Bond'][s1].append(dists[k])
                if idx_j in oxy_neighbors: oxy_neighbors[idx_j].append(idx_i)
        
        # Tilt Angles (M-O-M)
        for o_idx, neighbors in oxy_neighbors.items():
            mn = [n for n in neighbors if symbols[n] in present_B]
            if len(mn) >= 2:
                # Calculate angle between first two metal neighbors
                m1, m2 = mn[0], mn[1]
                pair_name = "-".join(sorted([symbols[m1], symbols[m2]]))
                angle = atoms.get_angle(m1, o_idx, m2, mic=True)
                # Normalize angle to be around 180 (for tilt) or just raw
                # Usually tilt is deviation from 180. Here we store the actual angle.
                if pair_name not in res['Tilt_Angle']: res['Tilt_Angle'][pair_name] = []
                res['Tilt_Angle'][pair_name].append(angle)
    return res

def run_dynamics_analysis(project_path, traj_path):
    print(f"    [Dynamics] Analyzing bonds & angles...")
    try:
        # Check file size for adaptive stride
        file_size_gb = os.path.getsize(traj_path) / (1024**3)
        actual_stride = FRAME_STEP
        if file_size_gb > 0.5:
            actual_stride = FRAME_STEP * 5
            print(f"    [Dynamics] Large file detected ({file_size_gb:.2f} GB). Using stride={actual_stride}.")

        traj = read(traj_path, index=f'::{actual_stride}')
        temps = load_temps(project_path, actual_stride)
        
        if temps is None:
            print("    [Skip] Temperature data not found.")
            return

        n = min(len(traj), len(temps))
        traj, temps = traj[:n], temps[:n]
        
        if len(traj) == 0: return

        syms = set(traj[0].get_chemical_symbols())
        p_A = [e for e in A_SITES if e in syms]
        p_B = [e for e in B_SITES if e in syms]
        
        temp_data = {}
        for i, atoms in enumerate(traj):
            T = temps[i]
            if T not in temp_data: temp_data[T] = {}
            fr = analyze_dynamics_frame(atoms, p_A, p_B)
            
            # Aggregate per frame
            for c, sd in fr.items():
                if c not in temp_data[T]: temp_data[T][c] = {}
                for s, vals in sd.items():
                    if vals:
                        if s not in temp_data[T][c]: temp_data[T][c][s] = []
                        temp_data[T][c][s].extend(vals) # Collect all values for this frame/temp

        # Calculate Statistics per Temp
        stats = []
        for T in sorted(temp_data.keys()):
            for c, sd in temp_data[T].items():
                for s, vals in sd.items():
                    if vals:
                        stats.append({
                            "Temperature": T, "Category": c, "Species": s,
                            "Mean": np.mean(vals), "Std": np.std(vals)
                        })
        
        if not stats: return

        df = pd.DataFrame(stats)
        df.to_csv(os.path.join(project_path, "dynamics_analysis.csv"), index=False)
        
        # Plotting
        categories = df['Category'].unique()
        for cat in categories:
            df_sub = df[df['Category'] == cat]
            species = df_sub['Species'].unique()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            for sp in species:
                dat = df_sub[df_sub['Species'] == sp].sort_values('Temperature')
                ax.plot(dat['Temperature'], dat['Mean'], marker='o', label=sp)
                ax.fill_between(dat['Temperature'], dat['Mean']-dat['Std'], dat['Mean']+dat['Std'], alpha=0.2)
            
            ax.set_xlabel("Temperature (K)")
            ylab = "Angle (deg)" if "Angle" in cat else "Distance (\AA)"
            ax.set_ylabel(ylab)
            ax.set_title(f"{cat}")
            if len(species) > 1: ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(project_path, f"Dynamics_{cat}.png"), dpi=150)
            plt.close()
            
        print("    [Dynamics] Saved csv and plots.")

    except Exception as e:
        print(f"    [Error] Dynamics failed: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# 3. RDF Analysis
# =============================================================================

def load_universe(xyz_path):
    try:
        # Check file size
        file_size_gb = os.path.getsize(xyz_path) / (1024**3)
        stride = 1
        if file_size_gb > 0.5:
            stride = 10 # Increase stride for large files
            print(f"    [RDF] Large file detected ({file_size_gb:.2f} GB). Using stride={stride} for loading.")

        ase_traj = read(xyz_path, index=f'::{stride}')
        if not ase_traj: return None
        
        n_atoms = len(ase_traj[0])
        
        # Check volume
        if ase_traj[0].get_volume() < 1.0:
            return None

        # Dimensions
        boxes = np.array([atoms.cell.cellpar() for atoms in ase_traj])
        coords = np.array([atoms.get_positions() for atoms in ase_traj])
        symbols = ase_traj[0].get_chemical_symbols()

        u = mda.Universe.empty(n_atoms, trajectory=True)
        u.add_TopologyAttr('name', symbols)
        u.load_new(coords, format=mda.coordinates.memory.MemoryReader)
        
        for i, ts in enumerate(u.trajectory):
            ts.dimensions = boxes[i]
        return u
    except Exception as e:
        print(f"    [Error] Universe loading failed: {e}")
        return None

def run_rdf_analysis(project_path, traj_path):
    print(f"    [RDF] Calculating RDFs...")
    try:
        temps_full = load_temps(project_path, step=1)
        u = load_universe(traj_path)
        if u is None: return

        if temps_full is None:
            # If no temps, just calculate total RDF
            n_frames = len(u.trajectory)
            temps_full = np.zeros(n_frames) # Dummy
        
        n_frames = min(len(u.trajectory), len(temps_full))
        temps_full = temps_full[:n_frames]
        
        atoms = u.atoms
        elements = set(atoms.names)
        targets = []
        
        # A-O and B-O
        for el in A_SITES + B_SITES:
            if el in elements:
                targets.append(el)
        
        O_sel = u.select_atoms("name O")
        if len(O_sel) == 0: return

        # 3.1 Total RDF
        fig, ax = plt.subplots(figsize=(8, 6))
        for el in targets:
            sel = u.select_atoms(f"name {el}")
            if len(sel) == 0: continue
            
            rdf_calc = rdf.InterRDF(sel, O_sel, range=RDF_RANGE, nbins=200, verbose=False)
            rdf_calc.run(start=0, stop=n_frames)
            
            ax.plot(rdf_calc.results.bins, rdf_calc.results.rdf, label=f"{el}-O", linewidth=2)
        
        ax.set_xlabel("r (\AA)"); ax.set_ylabel("g(r)")
        ax.set_title("Total RDF")
        if len(targets) > 1: ax.legend()
        ax.grid(True, alpha=0.3); ax.set_xlim(0, 6.0)
        plt.savefig(os.path.join(project_path, "RDF_Total.png"), dpi=150)
        plt.close()

        # 3.2 Temp-split RDF (Only if valid temps exist)
        if np.max(temps_full) > 0:
            min_T, max_T = np.floor(min(temps_full)/100)*100, np.ceil(max(temps_full)/100)*100
            bins = np.arange(min_T, max_T + 101, RDF_TEMP_BIN_SIZE)
            
            cmap = plt.get_cmap('coolwarm')
            
            for el in targets:
                sel = u.select_atoms(f"name {el}")
                if len(sel) == 0: continue
                
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_count = 0
                
                # Iterate bins
                for i in range(len(bins)-1):
                    low, high = bins[i], bins[i+1]
                    indices = np.where((temps_full >= low) & (temps_full < high))[0]
                    if len(indices) < 10: continue # Min frames
                    
                    start, stop = indices[0], indices[-1] + 1
                    
                    rdf_calc = rdf.InterRDF(sel, O_sel, range=RDF_RANGE, nbins=200, verbose=False)
                    rdf_calc.run(start=start, stop=stop)
                    
                    color = cmap(i / max(1, len(bins)-1))
                    ax.plot(rdf_calc.results.bins, rdf_calc.results.rdf, 
                            label=f"{int(low)}-{int(high)}K", color=color)
                    plot_count += 1
                
                if plot_count > 0:
                    ax.set_xlabel("r (\AA)"); ax.set_ylabel("g(r)")
                    ax.set_title(f"RDF Evolution: {el}-O")
                    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 6.0)
                    plt.savefig(os.path.join(project_path, f"RDF_TempOverlay_{el}.png"), dpi=150)
                plt.close()
        
        print("    [RDF] Saved plots.")

    except Exception as e:
        print(f"    [Error] RDF analysis failed: {e}")

# =============================================================================
# Main
# =============================================================================

def main():
    projects = sorted(glob.glob(os.path.join(PROJECTS_DIR, "*")))
    projects = [p for p in projects if os.path.isdir(p)]
    
    if not projects:
        print(f"No projects found in {PROJECTS_DIR}")
        return

    print(f"Found {len(projects)} projects. Starting batch analysis...")

    for i, proj in enumerate(projects):
        project_name = os.path.basename(proj)
        print(f"\n[{i+1}/{len(projects)}] Processing: {project_name}")
        
        # 1. Locate Trajectory
        traj_path = os.path.join(proj, TRAJ_FILENAME)
        if not os.path.exists(traj_path):
            print("    [Skip] No trajectory.xyz found.")
            continue
            
        # 2. Smooth Trajectory
        # Returns path to smoothed file (or original if failed/skipped)
        working_traj_path = smooth_trajectory(proj, traj_path)
        
        # 3. Dynamics Analysis
        run_dynamics_analysis(proj, working_traj_path)
        
        # 4. RDF Analysis
        run_rdf_analysis(proj, working_traj_path)

    print("\nAll batch processing complete.")

if __name__ == "__main__":
    main()
