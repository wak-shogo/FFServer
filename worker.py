# worker.py

import time
import os
import json
import pandas as pd
import torch
from ase.io import read, write
from ase import Atoms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import simulation_utils as sim
import visualization as viz
import notifications as notify

# --- 定数定義 ---
PROJECTS_DIR = "simulation_projects"
QUEUE_FILE = os.path.join(PROJECTS_DIR, "queue.json")
CURRENT_JOB_FILE = os.path.join(PROJECTS_DIR, "current_job.json")
REALTIME_DATA_FILE = os.path.join(PROJECTS_DIR, "realtime_data.csv")

def run_job(job_details):
    # ... (この関数の中身は変更なし) ...
    project_name = job_details['project_name']
    original_filename = job_details['original_filename']
    model_name = job_details['model']
    
    project_path = os.path.join(PROJECTS_DIR, project_name)
    if not os.path.exists(project_path): os.makedirs(project_path)
    job_type = job_details.get("job_type", "full_simulation")
    
    try:
        start_time = time.time()
        atoms = read(os.path.join(PROJECTS_DIR, original_filename))
        
        notify.send_to_discord(f"⚙️ Worker started optimizing: `{project_name}`", color=3447003)
        opt_atoms, _, _ = sim.optimize_structure(atoms, model_name=model_name, fmax=0.001)
        
        if job_type == "optimize_only":
            opt_cif_path = os.path.join(project_path, "optimized_structure.cif")
            write(opt_cif_path, opt_atoms, format="cif")
            
            elapsed_time = time.time() - start_time
            with open(os.path.join(project_path, "execution_time.txt"), "w") as f:
                f.write(f"{elapsed_time:.2f}")
            notify.send_to_discord(f"✅ Optimization finished: `{project_name}`\nTime: {elapsed_time:.2f} sec.", color=3066993)
            return
        
        else: # job_type == "full_simulation"
            notify.send_to_discord(f"🚀 NPT simulation started for: `{project_name}`", color=3447003)
            sim_mode = job_details['sim_mode']
            params = job_details['params']
            enable_cooling = params.get("enable_cooling", False)
            temp_range = params['temp_range']
            temp_start, temp_end, temp_step = temp_range
            
            def realtime_callback(current, total, message, partial_df=None):
                print(f"Progress: {message}")
                if partial_df is not None and not partial_df.empty:
                    partial_df.to_csv(REALTIME_DATA_FILE)

            traj_filepath = os.path.join(project_path, "trajectory.xyz")
            
            notify.send_to_discord(f"🔥 Heating phase started for: `{project_name}`", color=3447003)
            npt_df_heating, heating_final_struct = sim.run_npt_simulation_parallel(
                initial_atoms=opt_atoms, model_name=model_name, sim_mode=sim_mode,
                magmom_specie=params['magmom_specie'], temp_range=temp_range,
                time_step=1.0, eq_steps=params['eq_steps'], pressure=1.0,
                n_gpu_jobs=params['n_gpu_jobs'], progress_callback=realtime_callback,
                traj_filepath=traj_filepath, append_traj=False
            )
            if npt_df_heating.empty:
                raise ValueError("Heating phase failed.")

            npt_df = npt_df_heating
            
            if enable_cooling:
                notify.send_to_discord(f"❄️ Cooling phase started for: `{project_name}` (from {temp_end}K to {temp_start}K)", color=3447003)
                cooling_temp_range = (temp_end, temp_start, -temp_step)
                cooling_initial_atoms = Atoms(**heating_final_struct)
                npt_df_cooling, _ = sim.run_npt_simulation_parallel(
                    initial_atoms=cooling_initial_atoms, model_name=model_name, sim_mode=sim_mode,
                    magmom_specie=params['magmom_specie'], temp_range=cooling_temp_range,
                    time_step=1.0, eq_steps=params['eq_steps'], pressure=1.0,
                    n_gpu_jobs=params['n_gpu_jobs'], progress_callback=realtime_callback,
                    traj_filepath=traj_filepath, append_traj=True
                )
                if not npt_df_cooling.empty:
                    npt_df = pd.concat([npt_df_heating, npt_df_cooling], ignore_index=True)
                    notify.send_to_discord(f"❄️ Cooling phase completed for: `{project_name}`", color=3066993)
                else:
                    notify.send_to_discord(f"⚠️ Cooling phase failed for: `{project_name}`. Using heating data only.", color=16776960)

            if not npt_df.empty:
                elapsed_time = time.time() - start_time
                try:
                    with open(os.path.join(project_path, "execution_time.txt"), "w") as f:
                        f.write(f"{elapsed_time:.2f}")
                except Exception as e:
                    print(f"Error saving execution_time.txt for {project_name}: {e}")

                try:
                    fig_temp = viz.plot_temperature_dependent_properties(npt_df, 100)
                    fig_temp.savefig(os.path.join(project_path, "npt_vs_temp.png"))
                    plt.close(fig_temp)
                except Exception as e:
                    print(f"Error saving npt_vs_temp.png for {project_name}: {e}")
                    notify.send_to_discord(f"⚠️ Warning: Failed to generate plot for `{project_name}`.", color=16776960)

                try:
                    npt_df.to_csv(os.path.join(project_path, "npt_summary_full.csv"), index=False)
                except Exception as e:
                    print(f"Error saving npt_summary_full.csv for {project_name}: {e}")

                try:
                    npt_df.groupby('set_temps').last().reset_index().to_csv(
                        os.path.join(project_path, "npt_last_steps.csv"), index=False)
                except Exception as e:
                    print(f"Error saving npt_last_steps.csv for {project_name}: {e}")

                try:
                    agg_cols = [col for col in npt_df.columns if col != 'set_temps']
                    stats_df = npt_df.groupby('set_temps')[agg_cols].agg(['mean', 'std'])
                    stats_df.columns = ['_'.join(map(str, col)).strip() for col in stats_df.columns.values]
                    rename_mapping = {f'{col}_mean': col for col in agg_cols}
                    stats_df = stats_df.rename(columns=rename_mapping)
                    stats_df = stats_df.reset_index()
                    output_path = os.path.join(project_path, "npt_summary_stats.csv")
                    stats_df.to_csv(output_path, index=False, float_format='%.6f')
                except Exception as e:
                    print(f"Error generating statistical summary for {project_name}: {e}")
                    notify.send_to_discord(f"⚠️ Warning: Failed to generate statistical summary for `{project_name}`.", color=16776960)
                
                try:
                    magmom_specie = params.get('magmom_specie')
                    if magmom_specie:
                        magmom_cols = [col for col in npt_df.columns if col.startswith(f"{magmom_specie}_")]
                        if magmom_cols:
                            magmom_df = npt_df[magmom_cols].copy()
                            magmom_df.insert(0, 'step', range(len(magmom_df)))
                            magmom_df.to_csv(os.path.join(project_path, "magmoms_per_atom.csv"), index=False)
                except Exception as e:
                    print(f"Error saving magmoms_per_atom.csv for {project_name}: {e}")
                    notify.send_to_discord(f"⚠️ Warning: Failed to generate magmom-per-atom CSV for `{project_name}`.", color=16776960)

                cooling_str = " with cooling" if enable_cooling else ""
                notify.send_to_discord(f"🎉 NPT simulation finished: `{project_name}`{cooling_str}\nTime: {elapsed_time:.2f} sec.", color=3066993)
            else:
                notify.send_to_discord(f"❌ NPT simulation failed: `{project_name}`.", color=15158332)
    except Exception as e:
        import traceback
        error_msg = f"Unhandled exception in worker for job `{project_name}`: {e}\n{traceback.format_exc()}"
        print(error_msg)
        notify.send_to_discord(error_msg, color=15158332)

def main_worker_loop():
    print("Worker started. Watching for jobs...")
    while True:
        try:
            if not os.path.exists(CURRENT_JOB_FILE):
                queue = []
                if os.path.exists(QUEUE_FILE):
                    try:
                        with open(QUEUE_FILE, 'r') as f: queue = json.load(f)
                    except json.JSONDecodeError:
                        queue = []
                
                if queue:
                    next_job = queue.pop(0)
                    try:
                        with open(CURRENT_JOB_FILE, 'w') as f: json.dump(next_job, f)
                        with open(QUEUE_FILE, 'w') as f: json.dump(queue, f)
                        run_job(next_job)
                    finally:
                        # This block runs whether run_job succeeds or fails
                        if os.path.exists(CURRENT_JOB_FILE): os.remove(CURRENT_JOB_FILE)
                        if os.path.exists(REALTIME_DATA_FILE): os.remove(REALTIME_DATA_FILE)
                        
                        # Release GPU memory
                        if torch.cuda.is_available():
                            print("Clearing CUDA cache...")
                            torch.cuda.empty_cache()
                            print("CUDA cache cleared.")

        except Exception as e:
            print(f"Error in worker main loop: {e}")
            if os.path.exists(CURRENT_JOB_FILE): os.remove(CURRENT_JOB_FILE)
            if os.path.exists(REALTIME_DATA_FILE): os.remove(REALTIME_DATA_FILE)
        
        time.sleep(5)

if __name__ == "__main__":
    main_worker_loop()