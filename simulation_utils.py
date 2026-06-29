# simulation_utils.py

# --- Monkey-patch for importlib.metadata in Python 3.9 ---
import sys
if sys.version_info < (3, 10):
    import importlib_metadata
    import importlib
    importlib.metadata = importlib_metadata
# --- End of patch ---

import numpy as np
import pandas as pd
import torch
import gc
import os
from joblib import Parallel, delayed
from ase.io import read, write
from ase.filters import ExpCellFilter
from ase.optimize import BFGS, FIRE
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase import units, Atoms

_CALCULATOR_CACHE = {}

def get_calculator(model_name, use_device='cuda'):
    # Check if CUDA is actually available if requested
    if use_device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        use_device = 'cpu'

    cache_key = (model_name, use_device)
    if cache_key in _CALCULATOR_CACHE:
        return _CALCULATOR_CACHE[cache_key]

    if model_name == "CHGNet":
        from chgnet.model.dynamics import CHGNetCalculator
        try:
            calc = CHGNetCalculator(use_device=use_device)
        except IndexError:
            # Fallback for the "list index out of range" error in determine_device
            if use_device == 'cuda':
                print("Warning: CHGNet auto-device detection failed (IndexError). Attempting 'cuda:0'.")
                try:
                    calc = CHGNetCalculator(use_device='cuda:0')
                except Exception:
                    pass
            
            print("Warning: CHGNet device initialization failed. Falling back to CPU.")
            calc = CHGNetCalculator(use_device='cpu')
    elif model_name.startswith("matris_") or model_name == "MatRIS":
        from matris.applications.base import MatRISCalculator
        m_name = model_name if model_name != "MatRIS" else "matris_10m_oam"
        calc = MatRISCalculator(model=m_name, task="efsm", device=use_device)
    else:
        raise ValueError(f"Unknown or unsupported model specified: {model_name}. Supported models are 'CHGNet' and 'matris_10m_oam'/'matris_10m_mp'.")

    _CALCULATOR_CACHE[cache_key] = calc
    return calc

def clear_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def optimize_structure(atoms_obj, model_name, fmax=0.01, cancel_check_file=None):
    energies, lattice_constants = [], []
    atoms_obj.calc = get_calculator(model_name)
    
    # Revert to ExpCellFilter for memory efficiency and stability
    atoms_filter = ExpCellFilter(atoms_obj)
        
    opt = FIRE(atoms_filter)
    
    def save_step_data(a=atoms_filter):
        energies.append(a.atoms.get_potential_energy())
        lattice_constants.append(np.mean(a.atoms.get_cell().lengths()))
    
    opt.attach(save_step_data)
    
    def check_cancel():
        if cancel_check_file and os.path.exists(cancel_check_file):
            raise InterruptedError("Job cancelled by user during optimization.")
    opt.attach(check_cancel, interval=1)
    
    # Run optimization with a step limit to prevent freezing
    max_steps = 100
    max_attempts = 20 # Total 2000 steps
    converged = False
    
    for i in range(max_attempts):
        converged = opt.run(fmax=fmax, steps=max_steps)
        
        # 🟢 Force memory cleanup between steps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if converged:
            print(f"Optimization converged in {(i+1)*max_steps} steps or fewer.")
            break
        print(f"Optimization attempt {i+1}/{max_attempts} finished (not converged). Continuing...")
        
    if not converged:
        print("Warning: Optimization did not fully converge within the step limit. Returning current structure.")

    atoms_obj.wrap()
    return atoms_obj, energies, lattice_constants

def _run_single_temp_npt(params):
    # パラメータの受け取り (pfactorを含む)
    (model_name, sim_mode, temp, initial_structure_dict, magmom_specie, user_time_step,
     eq_steps, pressure, ttime, pfactor, use_device, cancel_check_file) = params
    atoms, calc, dyn = None, None, None
    try:
        # Check cancellation before starting
        if cancel_check_file and os.path.exists(cancel_check_file):
            return None

        # --- 1. 原子の復元 ---
        atoms = Atoms(**initial_structure_dict)
        atoms.wrap() # ドリフトによる数値精度低下を防ぐため、バッチ開始時にセル内に引き戻す
        
        # NPT(Nosé-Hoover)を使うため、maskを作成して直交性を維持します
        if sim_mode == "Legacy (Orthorhombic)":
            cell = atoms.get_cell(); a, b, c = np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])
            atoms.set_cell(np.diag([a, b, c]), scale_atoms=True)
            if not NPT._isuppertriangular(atoms.get_cell()): atoms.set_cell(atoms.cell.cellpar(), scale_atoms=True)
            # 角度(alpha, beta, gamma)を固定し、各軸の伸縮のみを許すマスク
            npt_mask = [1, 1, 1, 0, 0, 0] 
        else:
            cell = atoms.get_cell()
            # すでに上三角行列かつ適切な向きであれば、不必要な再配置(scale_atoms)を避ける
            if not NPT._isuppertriangular(cell):
                q, r = np.linalg.qr(cell)
                for i in range(3):
                    if r[i, i] < 0: r[i, :] *= -1
                atoms.set_cell(r, scale_atoms=True)
            npt_mask = None # フル緩和
        
        atoms.calc = get_calculator(model_name, use_device=use_device)
        
        # --- 2. データの準備 ---
        results_data = {
            "energies": [], "instant_temps": [], "volumes": [], "a_lengths": [], "b_lengths": [], "c_lengths": [],
            "alpha": [], "beta": [], "gamma": [], "positions": [], "cells": []
        }

        magmom_indices = []
        magmom_column_keys = []
        if magmom_specie:
            species_list = [s.strip() for s in magmom_specie.split(',') if s.strip()]
            from chgnet.model.dynamics import CHGNetCalculator
            from matris.applications.base import MatRISCalculator
            if isinstance(atoms.calc, (CHGNetCalculator, MatRISCalculator)):
                symbols = atoms.get_chemical_symbols()
                for target_s in species_list:
                    count = 1
                    for i, s in enumerate(symbols):
                        if s == target_s:
                            magmom_indices.append(i)
                            key = f"{target_s}_{count}"
                            magmom_column_keys.append(key)
                            results_data[key] = [] 
                            count += 1

        def log_step_data():
            nonlocal magmom_indices, magmom_column_keys
            # Check cancellation
            if cancel_check_file and os.path.exists(cancel_check_file):
                raise InterruptedError("Job cancelled by user.")

            a, b, c, alpha, beta, gamma = atoms.get_cell().cellpar()
            results_data["energies"].append(atoms.get_potential_energy())
            results_data["instant_temps"].append(atoms.get_temperature())
            results_data["volumes"].append(atoms.get_volume())
            results_data["a_lengths"].append(a); results_data["b_lengths"].append(b); results_data["c_lengths"].append(c)
            results_data["alpha"].append(alpha); results_data["beta"].append(beta); results_data["gamma"].append(gamma)
            results_data["positions"].append(atoms.get_positions())
            results_data["cells"].append(atoms.get_cell())
            
            if magmom_indices:
                try:
                    all_magmoms = atoms.get_magnetic_moments()
                    for i, atom_idx in enumerate(magmom_indices):
                        key = magmom_column_keys[i]
                        results_data[key].append(all_magmoms[atom_idx])
                except Exception as e:
                    print(f"Warning: Failed to retrieve magnetic moments: {e}. Disabling magnetic moment tracking for this run.")
                    # Clean up results_data to avoid length mismatch
                    for key in magmom_column_keys:
                        if key in results_data:
                            del results_data[key]
                    magmom_indices = []
                    magmom_column_keys = []

        # 🟢 Memory cleaner callback
        def clean_memory_step():
            # Clear CUDA cache periodically to prevent fragmentation/accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def check_cancel_npt():
            if cancel_check_file and os.path.exists(cancel_check_file):
                raise InterruptedError("Job cancelled by user.")

        # --- 3. MD初期化 ---
        init_temp = max(temp, 5.0)
        MaxwellBoltzmannDistribution(atoms, temperature_K=init_temp, force_temp=True)
        Stationary(atoms)
        ZeroRotation(atoms)
        
        # ==========================================
        # Phase 0: NVT (Langevin) 緩和
        # ==========================================
        # まず体積固定で温度をなじませる
        dyn_nvt = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=temp, friction=0.02)
        dyn_nvt.attach(check_cancel_npt, interval=5)
        dyn_nvt.run(100) 

        # ==========================================
        # Phase 1 & 2: NPT (Nosé-Hoover)
        # ==========================================
        
        # ✅ 【修正】 0Kでの除算エラー（爆発）を防ぐため、最低温度を1Kに設定する
        target_temp = max(temp, 1.0)

        dyn = NPT(
            atoms, 
            timestep=user_time_step * units.fs, 
            temperature_K=target_temp,   # temp -> target_temp に変更
            externalstress=pressure * units.bar,
            ttime=ttime,     # 呼び出し元から受け取った値 (25fs)
            pfactor=pfactor, # 呼び出し元から受け取った値 (軽い設定)
            mask=npt_mask    
        )
        
        # Attach memory cleaner (run every 50 steps)
        dyn.attach(clean_memory_step, interval=50)
        dyn.attach(check_cancel_npt, interval=5)
        
        # 初期緩和 (ログなし)
        dyn.run(200)

        # 本番 (ログあり)
        dyn.attach(log_step_data, interval=10)
        dyn.run(eq_steps)
        
        final_structure_dict = {'numbers': atoms.get_atomic_numbers(), 'positions': atoms.get_positions(), 'cell': atoms.get_cell(), 'pbc': atoms.get_pbc()}
        results_data["set_temps"] = [temp] * len(results_data["energies"])
        return temp, final_structure_dict, results_data

    except InterruptedError:
        print(f"Job at {temp} K interrupted by user flag.")
        raise
    except Exception as e:
        err_msg = str(e)
        import traceback; print(f"Error at {temp} K:"); traceback.print_exc()
        if "OutOfMemoryError" in str(e):
            err_msg = "GPU Out Of Memory. Try reducing --n-gpu-jobs."
            print(f"CRITICAL: {err_msg}")
        return None, err_msg, None
    finally:
        del atoms, calc, dyn; clear_memory()

def run_npt_simulation_parallel(initial_atoms, model_name, sim_mode, magmom_specie, temp_range, time_step, eq_steps,
    pressure, n_gpu_jobs, use_device='cuda', progress_callback=None, traj_filepath=None, append_traj=False, cancel_check_file=None):
    
    # ✅ パラメータを「成功していた古い設定」に戻す
    
    # 温度制御: 25 fs (強力な制御)
    ttime = 25 * units.fs 
    
    # 圧力制御: 固定値 (非常に軽い設定、体積依存性なし)
    # これが相転移の動きやすさを担保していた
    pfactor = 2e6 * units.GPa * (units.fs**2)
    
    initial_atoms.wrap()
    temperatures = np.arange(temp_range[0], temp_range[1] + temp_range[2], temp_range[2])
    all_results = []
    error_messages = []
    last_structure_dict = {'numbers': initial_atoms.get_atomic_numbers(), 'positions': initial_atoms.get_positions(), 'cell': initial_atoms.get_cell(), 'pbc': initial_atoms.get_pbc()}
    num_batches = int(np.ceil(len(temperatures) / n_gpu_jobs))
    
    for i in range(num_batches):
        if cancel_check_file and os.path.exists(cancel_check_file):
            print("Cancellation flag detected in parallel loop.")
            break

        if progress_callback: progress_callback(i, num_batches, f"Batch {i+1}/{num_batches} running...", None)
        batch_start_index, batch_end_index = i * n_gpu_jobs, min((i + 1) * n_gpu_jobs, len(temperatures))
        temp_batch = temperatures[batch_start_index:batch_end_index]
        if not len(temp_batch) > 0: continue
        
        # pfactor もタスク引数として渡す
        tasks = [(model_name, sim_mode, t, last_structure_dict, magmom_specie, time_step, eq_steps, pressure, ttime, pfactor, use_device, cancel_check_file) for t in temp_batch]
        batch_results = Parallel(n_jobs=n_gpu_jobs, mmap_mode='r+')(delayed(_run_single_temp_npt)(task) for task in tasks)
        
        valid_results = [res for res in batch_results if res[0] is not None]
        batch_errors = [res[1] for res in batch_results if res[0] is None and res[1] is not None]
        error_messages.extend(batch_errors)

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
    
    # Return captured errors along with results
    status_info = {"errors": list(set(error_messages))}
    
    if not all_results: return pd.DataFrame(), last_structure_dict, status_info
    
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
    return final_df, last_structure_dict, status_info