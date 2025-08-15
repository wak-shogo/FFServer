# worker.py
import time
import os
import json
import pandas as pd
from ase.io import read

import simulation_utils as sim
import notifications as notify

# --- 定数定義 ---
PROJECTS_DIR = "simulation_projects"
QUEUE_FILE = os.path.join(PROJECTS_DIR, "queue.json")
CURRENT_JOB_FILE = os.path.join(PROJECTS_DIR, "current_job.json")
REALTIME_DATA_FILE = os.path.join(PROJECTS_DIR, "realtime_data.csv")

def run_job(job_details):
    """単一のジョブを実行するメイン関数"""
    project_name = job_details['project_name']
    original_filename = job_details['original_filename']
    model_name = job_details['model']
    sim_mode = job_details['sim_mode']
    params = job_details['params']
    
    project_path = os.path.join(PROJECTS_DIR, project_name)
    if not os.path.exists(project_path):
        os.makedirs(project_path)

    notify.send_to_discord(f"🚀 Worker started processing: `{project_name}`", color=3447003)

    try:
        start_time = time.time()
        atoms = read(os.path.join(PROJECTS_DIR, original_filename))
        
        # 1. 構造最適化
        opt_atoms, _, _ = sim.optimize_structure(atoms, model_name=model_name, fmax=0.01)

        # 2. NPTシミュレーション
        def realtime_callback(current, total, message, partial_df):
            print(f"Progress: {message}")
            if not partial_df.empty:
                partial_df.to_csv(REALTIME_DATA_FILE)

        traj_filepath = os.path.join(project_path, "trajectory.xyz")
        npt_df = sim.run_npt_simulation_parallel(
            initial_atoms=opt_atoms, model_name=model_name, sim_mode=sim_mode, 
            magmom_specie=params['magmom_specie'], temp_range=params['temp_range'],
            time_step=1.0, eq_steps=params['eq_steps'], pressure=1.0, 
            n_gpu_jobs=params['n_gpu_jobs'], progress_callback=realtime_callback, 
            traj_filepath=traj_filepath
        )
        
        # 3. 結果の保存
        if not npt_df.empty:
            elapsed_time = time.time() - start_time
            with open(os.path.join(project_path, "execution_time.txt"), "w") as f:
                f.write(f"{elapsed_time:.2f}")

            # visualizationはmatplotlibに依存するため、GUIなし環境用に設定
            import matplotlib
            matplotlib.use('Agg')
            import visualization as viz
            fig_temp = viz.plot_temperature_dependent_properties(npt_df, 100)
            fig_temp.savefig(os.path.join(project_path, "npt_vs_temp.png"))

            npt_df.to_csv(os.path.join(project_path, "npt_summary_full.csv"), index=False)
            npt_df.groupby('set_temps').last().reset_index().to_csv(
                os.path.join(project_path, "npt_last_steps.csv"), index=False)
            
            notify.send_to_discord(f"🎉 Simulation finished: `{project_name}`\nTime: {elapsed_time:.2f} sec.", color=3066993)
        else:
             notify.send_to_discord(f"❌ Simulation failed: `{project_name}`.", color=15158332)

    except Exception as e:
        import traceback
        error_msg = f"Unhandled exception in worker for job `{project_name}`: {e}\n{traceback.format_exc()}"
        print(error_msg)
        notify.send_to_discord(error_msg, color=15158332)

def main_worker_loop():
    """ワーカーのメインループ"""
    print("Worker started. Watching for jobs...")
    while True:
        try:
            if not os.path.exists(CURRENT_JOB_FILE):
                # キューファイルからジョブを取得
                queue = []
                if os.path.exists(QUEUE_FILE):
                    with open(QUEUE_FILE, 'r') as f:
                        queue = json.load(f)
                
                if queue:
                    next_job = queue.pop(0)
                    
                    # 現在のジョブとしてマーク
                    with open(CURRENT_JOB_FILE, 'w') as f:
                        json.dump(next_job, f)
                    
                    # キューを更新
                    with open(QUEUE_FILE, 'w') as f:
                        json.dump(queue, f)
                        
                    # ジョブを実行
                    run_job(next_job)
                    
                    # ジョブ完了後、ファイルを削除
                    if os.path.exists(CURRENT_JOB_FILE): os.remove(CURRENT_JOB_FILE)
                    if os.path.exists(REALTIME_DATA_FILE): os.remove(REALTIME_DATA_FILE)
        
        except Exception as e:
            print(f"Error in worker main loop: {e}")
            # エラー発生時もクリーンアップ
            if os.path.exists(CURRENT_JOB_FILE): os.remove(CURRENT_JOB_FILE)
            if os.path.exists(REALTIME_DATA_FILE): os.remove(REAL_TIME_DATA_FILE)

        time.sleep(5) # 5秒ごとにキューをチェック

if __name__ == "__main__":
    main_worker_loop()