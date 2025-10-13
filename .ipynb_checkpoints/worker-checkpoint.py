# worker.py
import time
import os
import json
import pandas as pd
from ase.io import read, write # 🔄 `write` をインポート
from ase import Atoms  # ✅ --- 追加: Atomsのインポート ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # 📈 可視化のためにインポート
import simulation_utils as sim
import visualization as viz
import notifications as notify
# --- 定数定義 ---
PROJECTS_DIR = "simulation_projects"
QUEUE_FILE = os.path.join(PROJECTS_DIR, "queue.json")
CURRENT_JOB_FILE = os.path.join(PROJECTS_DIR, "current_job.json")
REALTIME_DATA_FILE = os.path.join(PROJECTS_DIR, "realtime_data.csv")
def run_job(job_details):
    """単一のジョブを実行するメイン関数。ジョブタイプによって処理を分岐する。"""
    project_name = job_details['project_name']
    original_filename = job_details['original_filename']
    model_name = job_details['model']
   
    project_path = os.path.join(PROJECTS_DIR, project_name)
    if not os.path.exists(project_path): os.makedirs(project_path)
    # 🔑 job_typeを取得。存在しない場合は'full_simulation'をデフォルトとする
    job_type = job_details.get("job_type", "full_simulation")
    try:
        start_time = time.time()
        atoms = read(os.path.join(PROJECTS_DIR, original_filename))
       
        # --- 構造最適化 (全てのジョブで共通) ---
        notify.send_to_discord(f"⚙️ Worker started optimizing: `{project_name}`", color=3447003)
        opt_atoms, _, _ = sim.optimize_structure(atoms, model_name=model_name, fmax=0.001)
        # ✅ --- ここから追加 --- (ジョブタイプによる分岐)
        if job_type == "optimize_only":
            # --- 最適化のみのジョブ ---
            opt_cif_path = os.path.join(project_path, "optimized_structure.cif")
            write(opt_cif_path, opt_atoms, format="cif")
           
            elapsed_time = time.time() - start_time
            with open(os.path.join(project_path, "execution_time.txt"), "w") as f:
                f.write(f"{elapsed_time:.2f}")
            notify.send_to_discord(f"✅ Optimization finished: `{project_name}`\nTime: {elapsed_time:.2f} sec.", color=3066993)
            return # NPTシミュレーションは行わずに終了
        # 🔄 --- ここから変更 --- (既存のNPT処理を `else` ブロックに移動)
        else: # job_type == "full_simulation" の場合
            # --- NPTシミュレーションジョブ ---
            notify.send_to_discord(f"🚀 NPT simulation started for: `{project_name}`", color=3447003)
            sim_mode = job_details['sim_mode']
            params = job_details['params']
            # ✅ --- ここに追加 --- (冷却オプションの取得)
            enable_cooling = params.get("enable_cooling", False)
            temp_range = params['temp_range']
            temp_start, temp_end, temp_step = temp_range
            def realtime_callback(current, total, message, partial_df=None):
                print(f"Progress: {message}")
                if partial_df is not None and not partial_df.empty:
                    partial_df.to_csv(REALTIME_DATA_FILE)
            traj_filepath = os.path.join(project_path, "trajectory.xyz")
            # --- 昇温シミュレーション ---
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
            # --- 冷却オプションの場合 ---
            npt_df = npt_df_heating
            if enable_cooling:
                notify.send_to_discord(f"❄️ Cooling phase started for: `{project_name}` (from {temp_end}K to {temp_start}K)", color=3447003)
                # 冷却用の温度範囲: end to start で同じstep
                cooling_temp_range = (temp_end, temp_start, -temp_step)  # 負のstepで降温
                # 冷却開始時の構造を昇温最終構造から取得
                cooling_initial_atoms = Atoms(**heating_final_struct)
                npt_df_cooling, _ = sim.run_npt_simulation_parallel(
                    initial_atoms=cooling_initial_atoms, model_name=model_name, sim_mode=sim_mode,
                    magmom_specie=params['magmom_specie'], temp_range=cooling_temp_range,
                    time_step=1.0, eq_steps=params['eq_steps'], pressure=1.0,
                    n_gpu_jobs=params['n_gpu_jobs'], progress_callback=realtime_callback,
                    traj_filepath=traj_filepath, append_traj=True  # 追記モード
                )
                if not npt_df_cooling.empty:
                    # DataFrameを結合（冷却データを追加）
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
                    try: # JSONファイルが空、または壊れている場合への対策
                        with open(QUEUE_FILE, 'r') as f: queue = json.load(f)
                    except json.JSONDecodeError:
                        queue = []
                if queue:
                    next_job = queue.pop(0)
                    with open(CURRENT_JOB_FILE, 'w') as f: json.dump(next_job, f)
                    with open(QUEUE_FILE, 'w') as f: json.dump(queue, f)
                    run_job(next_job)
                    if os.path.exists(CURRENT_JOB_FILE): os.remove(CURRENT_JOB_FILE)
                    if os.path.exists(REALTIME_DATA_FILE): os.remove(REALTIME_DATA_FILE)
        except Exception as e:
            print(f"Error in worker main loop: {e}")
            if os.path.exists(CURRENT_JOB_FILE): os.remove(CURRENT_JOB_FILE)
            if os.path.exists(REALTIME_DATA_FILE): os.remove(REALTIME_DATA_FILE)
        time.sleep(5)
if __name__ == "__main__":
    main_worker_loop()