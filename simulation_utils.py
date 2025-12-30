# simulation_utils.py

import numpy as np
import pandas as pd
import torch
import gc
import os
from joblib import Parallel, delayed
from ase.io import read, write
from ase.filters import ExpCellFilter
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.npt import NPT
from ase import units, Atoms
from chgnet.model.dynamics import CHGNetCalculator
from mattersim.forcefield import MatterSimCalculator
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from nequip.ase import NequIPCalculator

def get_calculator(model_name, use_device='cuda'):
    if model_name == "CHGNet": return CHGNetCalculator(use_device=use_device)
    elif model_name == "MatterSim": return MatterSimCalculator(device=use_device)
    elif model_name == "Orb":
        orbff = pretrained.orb_v3_conservative_inf_omat(device=use_device, precision="float32-high")
        return ORBCalculator(orbff, device=use_device)
    elif model_name == "NequipOLM":
        model_path = "/workspace/NequipOLM_model/nequip-oam-l.nequip.pt2"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"NequipOLM model not found at '{model_path}'. "
                f"Please compile the model and place it in the 'NequipOLM_model' directory."
            )
        return NequIPCalculator.from_compiled_model(model_path, device=use_device)
    else: raise ValueError(f"Unknown model specified: {model_name}")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def optimize_structure(atoms_obj, model_name, fmax=0.01):
    energies, lattice_constants = [], []
    atoms_obj.calc = get_calculator(model_name)
    atoms_filter = ExpCellFilter(atoms_obj)
    opt = BFGS(atoms_filter)
    def save_step_data(a=atoms_filter):
        energies.append(a.atoms.get_potential_energy())
        lattice_constants.append(np.mean(a.atoms.get_cell().lengths()))
    opt.attach(save_step_data); opt.run(fmax=fmax)
    return atoms_obj, energies, lattice_constants

def _run_single_temp_npt(params):
    (model_name, sim_mode, temp, initial_structure_dict, magmom_specie, time_step,
     eq_steps, pressure, ttime, pfactor, use_device) = params
    atoms, calc, dyn = None, None, None
    try:
        atoms = Atoms(**initial_structure_dict)
        if sim_mode == "Legacy (Orthorhombic)":
            cell = atoms.get_cell(); a, b, c = np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])
            atoms.set_cell(np.diag([a, b, c]), scale_atoms=True)
            if not NPT._isuppertriangular(atoms.get_cell()): atoms.set_cell(atoms.cell.cellpar(), scale_atoms=True)
            npt_mask = (1, 1, 1)
        else:
            cell = atoms.get_cell(); q, r = np.linalg.qr(cell)
            for i in range(3):
                if r[i, i] < 0: r[i, :] *= -1
            atoms.set_cell(r, scale_atoms=True); npt_mask = None
        
        atoms.calc = get_calculator(model_name, use_device=use_device)
        
        results_data = {
            "energies": [], "instant_temps": [], "volumes": [], "a_lengths": [], "b_lengths": [], "c_lengths": [],
            "alpha": [], "beta": [], "gamma": [], "positions": [], "cells": []
        }

        # ✅ --- 変更点 Start ---
        # magmomを記録する原子のインデックスと、それに対応する列名を準備
        magmom_indices = []
        magmom_column_keys = []
        if magmom_specie and isinstance(atoms.calc, CHGNetCalculator):
            symbols = atoms.get_chemical_symbols()
            count = 1
            for i, s in enumerate(symbols):
                if s == magmom_specie:
                    magmom_indices.append(i)
                    key = f"{magmom_specie}_{count}"
                    magmom_column_keys.append(key)
                    results_data[key] = [] # 各原子用の空リストを初期化
                    count += 1
        # ✅ --- 変更点 End ---

        def log_step_data():
            a, b, c, alpha, beta, gamma = atoms.get_cell().cellpar()
            results_data["energies"].append(atoms.get_potential_energy()); results_data["instant_temps"].append(atoms.get_temperature())
            results_data["volumes"].append(atoms.get_volume()); results_data["a_lengths"].append(a); results_data["b_lengths"].append(b)
            results_data["c_lengths"].append(c); results_data["alpha"].append(alpha); results_data["beta"].append(beta)
            results_data["gamma"].append(gamma); results_data["positions"].append(atoms.get_positions()); results_data["cells"].append(atoms.get_cell())
            
            # ✅ --- 変更点 Start ---
            # 平均値ではなく、各原子のmagmomを対応するリストに記録
            if magmom_indices:
                all_magmoms = atoms.get_magnetic_moments()
                for i, atom_idx in enumerate(magmom_indices):
                    key = magmom_column_keys[i]
                    results_data[key].append(all_magmoms[atom_idx])
            # ✅ --- 変更点 End ---

        MaxwellBoltzmannDistribution(atoms, temperature_K=temp, force_temp=True)
        dyn = NPT(atoms, timestep=time_step * units.fs, temperature_K=temp, externalstress=pressure * units.bar, ttime=ttime, pfactor=pfactor, mask=npt_mask)
        dyn.attach(log_step_data, interval=10); dyn.run(eq_steps)
        
        final_structure_dict = {'numbers': atoms.get_atomic_numbers(), 'positions': atoms.get_positions(), 'cell': atoms.get_cell(), 'pbc': atoms.get_pbc()}
        
        # ❌ --- 削除 --- (古いmagmom処理は不要)
        # if magmom_specie: results_data[f"{magmom_specie}_magmom"] = results_data.pop("magmoms")
        # else: results_data.pop("magmoms")
        
        results_data["set_temps"] = [temp] * len(results_data["energies"])
        return temp, final_structure_dict, results_data
    except Exception as e:
        import traceback; print(f"Error at {temp} K:"); traceback.print_exc()
        return None
    finally:
        del atoms, calc, dyn; clear_memory()

# run_npt_simulation_parallel関数は変更不要です
def run_npt_simulation_parallel(initial_atoms, model_name, sim_mode, magmom_specie, temp_range, time_step, eq_steps,
    pressure, n_gpu_jobs, use_device='cuda', progress_callback=None, traj_filepath=None, append_traj=False):
    ttime = 25 * units.fs
    pfactor = 2e6 * units.GPa * (units.fs**2)
    temperatures = np.arange(temp_range[0], temp_range[1] + temp_range[2], temp_range[2])
    all_results = []
    last_structure_dict = {'numbers': initial_atoms.get_atomic_numbers(), 'positions': initial_atoms.get_positions(), 'cell': initial_atoms.get_cell(), 'pbc': initial_atoms.get_pbc()}
    num_batches = int(np.ceil(len(temperatures) / n_gpu_jobs))
    
    for i in range(num_batches):
        if progress_callback: progress_callback(i, num_batches, f"Batch {i+1}/{num_batches} running...", None)
        batch_start_index, batch_end_index = i * n_gpu_jobs, min((i + 1) * n_gpu_jobs, len(temperatures))
        temp_batch = temperatures[batch_start_index:batch_end_index]
        if not len(temp_batch) > 0: continue
        
        tasks = [(model_name, sim_mode, t, last_structure_dict, magmom_specie, time_step, eq_steps, pressure, ttime, pfactor, use_device) for t in temp_batch]
        batch_results = Parallel(n_jobs=n_gpu_jobs, mmap_mode='r+')(delayed(_run_single_temp_npt)(task) for task in tasks)
        
        valid_results = [res for res in batch_results if res is not None]
        if not valid_results: break
        all_results.extend(valid_results)

        if temp_range[2] > 0:
            next_initial_result = max(valid_results, key=lambda x: x[0])
        else:
            next_initial_result = min(valid_results, key=lambda x: x[0])
        last_structure_dict = next_initial_result[1]
    
        if progress_callback:
            temp_df_list = [pd.DataFrame({k: v for k, v in res[2].items() if k not in ["positions", "cells"]}) for res in all_results]
            partial_df = pd.concat(temp_df_list, ignore_index=True)
            progress_callback(i + 1, num_batches, f"Batch {i+1}/{num_batches} finished.", partial_df)

    if progress_callback: progress_callback(num_batches, num_batches, "NPT simulation finished.", None)
    if not all_results: return pd.DataFrame(), last_structure_dict
    
    if traj_filepath:
        atomic_numbers = initial_atoms.get_atomic_numbers()
        new_frames = [Atoms(numbers=atomic_numbers, positions=p, cell=c, pbc=True) for _, _, res in all_results for p, c in zip(res.get("positions", []), res.get("cells", []))]
        if append_traj and os.path.exists(traj_filepath):
            existing_frames = read(traj_filepath, index=':')
        else:
            existing_frames = []
        all_frames = existing_frames + new_frames
        if all_frames: write(traj_filepath, all_frames, format='extxyz')

    df_list = []
    for temp, final_struct, result_dict in all_results:
        clean_dict = {k: v for k, v in result_dict.items() if k not in ["positions", "cells"]}
        df_list.append(pd.DataFrame(clean_dict))
    
    final_df = pd.concat(df_list, ignore_index=True)
    return final_df, last_structure_dict