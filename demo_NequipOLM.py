##############################################################################
# NequIP-OAM-L を用いた NPT MDシミュレーション - Google Colab 統合ノートブック
#
# 使い方：
# 1. このセルを実行します。
# 2. 途中でCIFファイルをアップロードするよう求められます。
# 3. シミュレーションが完了すると、"simulation_results.zip" がダウンロードされます。
#
# (バグ修正済みバージョン)
##############################################################################

# --- パート1：依存関係のインストール ---
print("--- パート1：依存関係のインストール ---")
# nequip-allegroをインストールすると、nequip本体もインストールされます
!pip install ase pandas matplotlib nequip-allegro

print("\n--- ライブラリのインポート ---")
import os
import zipfile
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

# ASE (Atomic Simulation Environment) のコンポーネント
from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.npt import NPT  # NPT (等温等圧) アンサンブルを使用
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# NequIP Calculator
from nequip.ase import NequIPCalculator


# --- パート2：パラメータ設定とファイルアップロード ---
print("\n--- パート2：パラメータ設定とファイルアップロード ---")

# --- シミュレーション設定 ---
TARGET_TEMP_K = 300.0       # ターゲット温度 (K) [15, 16]
TARGET_PRESS_BAR = 1.01325  # ターゲット圧力 (bar)
TIME_STEP_FS = 1.0          # タイムステップ (fs) [15, 16]

# --- 実行時間設定 ---
TOTAL_STEPS = 5000          # 総ステップ数
EQUILIBRATION_STEPS = 1000    # 平衡化ステップ数
LOG_INTERVAL_STEPS = 10       # ログ記録間隔 (ステップ) [19, 20, 21]

# --- ファイル名設定 ---
COMPILED_MODEL_NAME = 'nequip-oam-l.nequip.pt2' # コンパイル済みモデル名
OUTPUT_ZIP_NAME = 'simulation_results.zip'     # 出力ZIPファイル名 [22]

# --- NPTアンサンブル用定数 ---
TTEMP_FS = 25 * units.fs  # サーモスタット時定数
PFACTOR_FS2 = 75**2 * units.fs**2 # バロスタット時定数

# --- CIFファイルのアップロード ---
print("シミュレーションに使用するCIFファイルをアップロードしてください：")
uploaded_files = files.upload()

if not uploaded_files:
    raise Exception("ファイルがアップロードされませんでした。セルを再実行してください。")

# 【修正1】アップロードされた辞書のキーのリストから、最初のファイル名（文字列）を取得
cif_filename = list(uploaded_files.keys())[0]
print(f"ファイル '{cif_filename}' がアップロードされました。")


# --- パート3：モデルの取得とコンパイル ---
print("\n--- パート3：モデルの取得とコンパイル ---")
print("NequIP-OAM-L:0.1 モデルのコンパイルを開始します...")
print("（これには数分かかる場合があります）")

# nequip-compile を実行し、モデルをAOTコンパイルします
!nequip-compile \
  nequip.net:mir-group/NequIP-OAM-L:0.1 \
  {COMPILED_MODEL_NAME} \
  --mode aotinductor \
  --device cuda \
  --target ase

print("モデルのコンパイルが完了しました。")

# コンパイルが成功したかを確認
if not os.path.exists(COMPILED_MODEL_NAME):
    raise Exception(f"モデルコンパイルに失敗しました。'{COMPILED_MODEL_NAME}' が見つかりません。")
else:
    print(f"コンパイル済みモデル '{COMPILED_MODEL_NAME}' が正常に生成されました。")


# --- パート4：シミュレーションの実行 ---
print("\n--- パート4：シミュレーションの実行 ---")
print("シミュレーションのセットアップを開始します...")

# 1. 構造のロード (cif_filename は文字列なので、そのまま read に渡せる)
atoms = read(cif_filename)
print(f"初期構造 '{cif_filename}' をロードしました。原子数: {len(atoms)}")

# 2. NequIP計算機のインスタンス化
# from_compiled_model を使用して、最適化されたモデルをロードします
calculator = NequIPCalculator.from_compiled_model(
    COMPILED_MODEL_NAME,
    device="cuda"
)
print("コンパイル済み NequIP-OAM-L Calculator をGPUにロードしました。")

# 3. 計算機のアタッチ
atoms.calc = calculator

# 初期速度の設定
MaxwellBoltzmannDistribution(atoms, temperature_K=TARGET_TEMP_K)

# 4. MDアンサンブルの選択 (NPT)
dyn = NPT(
    atoms,
    timestep=TIME_STEP_FS * units.fs,
    temperature_K=TARGET_TEMP_K,
    external_pressure=TARGET_PRESS_BAR * units.bar,
    ttime=TTEMP_FS,
    pfactor=PFACTOR_FS2
)

# 5. データロガーのアタッチ（カスタムロガー）
# 【修正2】md_data を空のリストとして初期化
md_data = []
# 【修正3】data_columns をリストとして定義
data_columns = ['Time_ps', 'PotEnergy_eV', 'KinEnergy_eV', 'TotalEnergy_eV', 'Temperature_K', 'Volume_A3']


def custom_logger():
    time_ps = dyn.get_time() / 1000.0
    potential_energy = atoms.get_potential_energy()
    kinetic_energy = atoms.get_kinetic_energy()
    total_energy = potential_energy + kinetic_energy
    temperature = atoms.get_temperature() #
    volume = atoms.get_volume()           #

    md_data.append([
        time_ps,
        potential_energy,
        kinetic_energy,
        total_energy,
        temperature,
        volume
    ])

dyn.attach(custom_logger, interval=LOG_INTERVAL_STEPS)
print(f"カスタムロガーを {LOG_INTERVAL_STEPS} ステップごとにアタッチしました。")

# 6. 軌跡（Trajectory）ロガーのアタッチ
traj_filename = 'simulation.xyz'
traj_writer = Trajectory(traj_filename, 'w', atoms)
dyn.attach(traj_writer.write, interval=LOG_INTERVAL_STEPS)
print(f"XYZ軌跡 ({traj_filename}) を {LOG_INTERVAL_STEPS} ステップごとにアタッチしました。")

# 7. シミュレーションの実行
print(f"\n--- {TOTAL_STEPS} ステップのNPTシミュレーションを開始（ログ間隔：{LOG_INTERVAL_STEPS} ステップ）---")
# ヘッダーを print_status の出力と一致させる
print("ステップ | 時間 (ps) | 温度 (K) | 体積 (A^3) | E_pot (eV) ")

# 【修正4】print_status 関数を修正
def print_status(is_initial=False):
    step = dyn.get_number_of_steps()
    # ログ間隔の10倍ごと、または初期ステップでステータスを出力
    if is_initial or (step % (LOG_INTERVAL_STEPS * 10) == 0 and step > 0):
        if not md_data: # md_dataが空の場合は何もしない（初期呼び出し対策）
            return
        data = md_data[-1] # custom_logger がすでに追加した最新データを取得
        # Header: ステップ | 時間 (ps) | 温度 (K) | 体積 (A^3) | E_pot (eV)
        # Data:   step    | data[0]    | data[4]    | data[5]    | data[1]
        # 【修正5】data リストの正しいインデックス（0, 4, 5, 1）を参照
        print(f"{step:>6} | {data[0]:>9.2f} | {data[4]:>10.2f} | {data[5]:>10.2f} | {data[1]:>10.2f}")


dyn.attach(print_status, interval=LOG_INTERVAL_STEPS)

# 最初のステップ（ステップ0）を手動でログ
custom_logger()
# 【修正6】is_initial=True を指定して呼び出し、ステップ0の情報を表示
print_status(is_initial=True)

# シミュレーション実行
dyn.run(TOTAL_STEPS)

# 最終ステップの情報を表示（TOTAL_STEPSが100の倍数でない場合のため）
if TOTAL_STEPS % (LOG_INTERVAL_STEPS * 10)!= 0:
    data = md_data[-1]
    # 【修正7】data リストの正しいインデックス（0, 4, 5, 1）を参照
    print(f"{TOTAL_STEPS:>6} | {data[0]:>9.2f} | {data[4]:>10.2f} | {data[5]:>10.2f} | {data[1]:>10.2f}")

print("--- シミュレーション完了 ---")


# --- パート5：分析、視覚化、CSVの生成 ---
print("\n--- パート5：分析、視覚化、CSVの生成 ---")

# 5.1. データフレームの作成とCSV保存
# (data_columns が定義されたので、これは正常に動作します)
df = pd.DataFrame(md_data, columns=data_columns)
csv_log_filename = 'simulation_log.csv'
df.to_csv(csv_log_filename, index=False)
print(f"シミュレーションログを '{csv_log_filename}' に保存しました。")

# 5.2. 時系列プロットの生成
print("時系列プロットを生成中...")

equilibration_time_ps = (EQUILIBRATION_STEPS * TIME_STEP_FS) / 1000.0
# 【修正8】構文エラーだったコメント行を削除
# Time_ps はほぼ正確ですが、ステップベースの方が確実です
production_start_index = int(EQUILIBRATION_STEPS / LOG_INTERVAL_STEPS)
df_production = df.iloc[production_start_index:]


def plot_time_series(column_name, y_label, filename):
    plt.figure(figsize=(10, 6))
    # 【修正9】plot の第一引数（x軸）に df['Time_ps'] を指定
    plt.plot(df['Time_ps'], df[column_name], label='Simulation Data', alpha=0.8)
    plt.axvline(x=equilibration_time_ps, color='r', linestyle='--', label=f'Equilibration ({equilibration_time_ps} ps)')

    if not df_production.empty:
        mean_val = df_production[column_name].mean()
        std_val = df_production[column_name].std()

        plt.axhline(y=mean_val, color='k', linestyle='-', label=f'Mean: {mean_val:.2f}')
        # 【修正10】fill_between の第一引数（x軸）に df_production['Time_ps'] を指定
        plt.fill_between(
            df_production['Time_ps'],
            mean_val - std_val,
            mean_val + std_val,
            color='gray',
            alpha=0.3,
            label=f'±1 std (std={std_val:.2f})'
        ) #

    plt.xlabel('Time (ps)')
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs. Time')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(filename)
    plt.close()
    print(f"プロットを '{filename}' に保存しました。")

plot_filenames = {
    'temp_plot.png': ('Temperature_K', 'Temperature (K)'),
    'pot_energy_plot.png': ('PotEnergy_eV', 'Potential Energy (eV)'),
    'volume_plot.png': ('Volume_A3', 'Volume (Å³)')
}
for filename, (col, label) in plot_filenames.items():
    plot_time_series(col, label, filename)

# 5.3. 統計分析の実行
print("\n--- 生産フェーズ（Production Phase）の統計分析 ---")
if not df_production.empty:
    # [32, 33]
    stats = df_production[data_columns[1:]].agg(['mean', 'std']).transpose()
    # 【修正11】stats.columns をリストとして定義
    stats.columns = ['Mean', 'Std'] # [34, 35, 36, 37, 38]
    print(stats)

    summary_filename = 'simulation_summary.csv'
    stats.to_csv(summary_filename)
    print(f"\n統計サマリーを '{summary_filename}' に保存しました。")
else:
    print("平衡化ステップ後の生産フェーズのデータがありません。統計サマリーはスキップされました。")
    summary_filename = None # ZIPに追加しないように


# --- パート6：結果のパッケージ化とダウンロード ---
print("\n--- パート6：結果のパッケージ化とダウンロード ---")
print(f"すべての結果を '{OUTPUT_ZIP_NAME}' にまとめています...")

files_to_zip = [
    csv_log_filename,
    traj_filename,
]
if summary_filename:
    files_to_zip.append(summary_filename)

files_to_zip.extend(list(plot_filenames.keys()))

# ZIPファイルを作成
with zipfile.ZipFile(OUTPUT_ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in files_to_zip:
        if os.path.exists(file):
            zipf.write(file)
            print(f"  - {file} を追加しました。")
        else:
            print(f"  - 警告: {file} が見つかりません。")

print(f"'{OUTPUT_ZIP_NAME}' の作成が完了しました。")

# ユーザーにZIPファイルをダウンロード
print("ダウンロードダイアログを起動します...")
files.download(OUTPUT_ZIP_NAME) # [39, 40, 22]
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
from nequip.ase import NequIPCalculator
# REMOVED: from nequip.model.deployment import load_deployed_model # This import is no longer needed
import matplotlib.pyplot as plt
from datetime import datetime
from google.colab import files
import io
import time

# Define Constants
PROJECTS_DIR = "simulation_projects"
if not os.path.exists(PROJECTS_DIR):
    os.makedirs(PROJECTS_DIR)

# Utility Functions
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 修正: get_calculator 関数のエラー対応

def get_calculator(model_name, use_device='cuda'):
    if model_name == "NequIP-OAM-L":
        # 修正: コンパイルされたモデルの正しいファイル名を使用
        model_path = "nequip-oam-l.nequip.pt2"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Run nequip-compile first.")

        # NequIPCalculator.from_compiled_model を使用してモデルをロード
        return NequIPCalculator.from_compiled_model(model_path, device=use_device)
    else:
        raise ValueError(f"Unknown model specified: {model_name}")

def optimize_structure(atoms_obj, model_name, fmax=0.001):
    energies, lattice_constants = [], []
    atoms_obj.calc = get_calculator(model_name)
    atoms_filter = ExpCellFilter(atoms_obj)
    opt = BFGS(atoms_filter)
    def save_step_data(a=atoms_filter):
        energies.append(a.atoms.get_potential_energy())
        lattice_constants.append(np.mean(a.atoms.get_cell().lengths()))
    opt.attach(save_step_data)
    opt.run(fmax=fmax)
    return atoms_obj, energies, lattice_constants

def _run_single_temp_npt(params):
    (model_name, sim_mode, temp, initial_structure_dict, time_step,
     eq_steps, pressure, ttime, pfactor, use_device) = params
    atoms, calc, dyn = None, None, None
    try:
        atoms = Atoms(**initial_structure_dict)
        if sim_mode == "Legacy (Orthorombic)":
            cell = atoms.get_cell()
            a, b, c = np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])
            atoms.set_cell(np.diag([a, b, c]), scale_atoms=True)
            if not NPT._isuppertriangular(atoms.get_cell()):
                atoms.set_cell(atoms.cell.cellpar(), scale_atoms=True)
            npt_mask = (1, 1, 1)
        else:
            cell = atoms.get_cell()
            q, r = np.linalg.qr(cell)
            for i in range(3):
                if r[i, i] < 0:
                    r[i, :] *= -1
            atoms.set_cell(r, scale_atoms=True)
            npt_mask = None

        atoms.calc = get_calculator(model_name, use_device=use_device)

        results_data = {
            "energies": [], "instant_temps": [], "volumes": [], "a_lengths": [], "b_lengths": [], "c_lengths": [],
            "alpha": [], "beta": [], "gamma": [], "positions": [], "cells": []
        }

        def log_step_data():
            a, b, c, alpha, beta, gamma = atoms.get_cell().cellpar()
            results_data["energies"].append(atoms.get_potential_energy())
            results_data["instant_temps"].append(atoms.get_temperature())
            results_data["volumes"].append(atoms.get_volume())
            results_data["a_lengths"].append(a)
            results_data["b_lengths"].append(b)
            results_data["c_lengths"].append(c)
            results_data["alpha"].append(alpha)
            results_data["beta"].append(beta)
            results_data["gamma"].append(gamma)
            results_data["positions"].append(atoms.get_positions())
            results_data["cells"].append(atoms.get_cell())

        MaxwellBoltzmannDistribution(atoms, temperature_K=temp, force_temp=True)
        # Corrected: changed 'external_pressure' to 'externalstress'
        dyn = NPT(atoms, timestep=time_step * units.fs, temperature_K=temp, externalstress=pressure * units.bar, ttime=ttime, pfactor=pfactor, mask=npt_mask)
        dyn.attach(log_step_data, interval=10)
        dyn.run(eq_steps)

        final_structure_dict = {'numbers': atoms.get_atomic_numbers(), 'positions': atoms.get_positions(), 'cell': atoms.get_cell(), 'pbc': atoms.get_pbc()}

        results_data["set_temps"] = [temp] * len(results_data["energies"])
        return temp, final_structure_dict, results_data
    except Exception as e:
        import traceback
        print(f"Error at {temp} K:")
        traceback.print_exc()
        return None
    finally:
        del atoms, calc, dyn
        clear_memory()

def run_npt_simulation_parallel(initial_atoms, model_name, sim_mode, temp_range, time_step, eq_steps,
                                pressure, n_gpu_jobs, use_device='cuda', progress_callback=None, traj_filepath=None, append_traj=False):
    ttime = 25 * units.fs
    pfactor = 2e6 * units.GPa * (units.fs**2)
    temperatures = np.arange(temp_range[0], temp_range[1] + temp_range[2], temp_range[2])
    all_results = []
    last_structure_dict = {'numbers': initial_atoms.get_atomic_numbers(), 'positions': initial_atoms.get_positions(), 'cell': initial_atoms.get_cell(), 'pbc': initial_atoms.get_pbc()}
    num_batches = int(np.ceil(len(temperatures) / n_gpu_jobs))

    for i in range(num_batches):
        if progress_callback:
            progress_callback(i, num_batches, f"Batch {i+1}/{num_batches} running...", None)
        batch_start_index, batch_end_index = i * n_gpu_jobs, min((i + 1) * n_gpu_jobs, len(temperatures))
        temp_batch = temperatures[batch_start_index:batch_end_index]
        if not len(temp_batch) > 0:
            continue

        tasks = [(model_name, sim_mode, t, last_structure_dict, time_step, eq_steps, pressure, ttime, pfactor, use_device) for t in temp_batch]
        batch_results = Parallel(n_jobs=n_gpu_jobs, mmap_mode='r+')(delayed(_run_single_temp_npt)(task) for task in tasks)

        valid_results = [res for res in batch_results if res is not None]
        if not valid_results:
            break
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

    if progress_callback:
        progress_callback(num_batches, num_batches, "NPT simulation finished.", None)
    if not all_results:
        return pd.DataFrame(), last_structure_dict

    if traj_filepath:
        atomic_numbers = initial_atoms.get_atomic_numbers()
        new_frames = [Atoms(numbers=atomic_numbers, positions=p, cell=c, pbc=True) for _, _, res in all_results for p, c in zip(res.get("positions", []), res.get("cells", []))]
        if append_traj and os.path.exists(traj_filepath):
            existing_frames = read(traj_filepath, index=':')
        else:
            existing_frames = []
        all_frames = existing_frames + new_frames
        if all_frames:
            write(traj_filepath, all_frames, format='extxyz')

    df_list = []
    for temp, final_struct, result_dict in all_results:
        clean_dict = {k: v for k, v in result_dict.items() if k not in ["positions", "cells"]}
        df_list.append(pd.DataFrame(clean_dict))

    final_df = pd.concat(df_list, ignore_index=True)
    return final_df, last_structure_dict

def plot_temperature_dependent_properties(npt_df, window_size=100):
    # Simple rolling mean plot for volumes vs temperature as example
    fig, ax = plt.subplots()
    stats_df = npt_df.groupby('set_temps').agg({'volumes': ['mean', 'std']}).reset_index()
    stats_df.columns = ['set_temps', 'volumes_mean', 'volumes_std']
    ax.errorbar(stats_df['set_temps'], stats_df['volumes_mean'], yerr=stats_df['volumes_std'], fmt='o-')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Volume (Å³)')
    ax.set_title('Volume vs Temperature')
    return fig

# Step 4: User Input Section
from google.colab import widgets
import ipywidgets as widgets

# Upload CIF File
uploaded = files.upload()
if not uploaded:
    raise ValueError("Please upload a CIF file.")
cif_filename = list(uploaded.keys())[0]
with open(cif_filename, 'wb') as f:
    f.write(uploaded[cif_filename])

# Parameters
model_name = "NequIP-OAM-L"  # Fixed as per request
sim_mode = widgets.Dropdown(options=["Realistic (ISIF=3)", "Legacy (Orthorombic)"], description="Sim Mode:")
project_prefix = widgets.Text(value=datetime.now().strftime("%Y%m%d"), description="Project Prefix:")
temp_start = widgets.IntText(value=1, description="Start Temp (K):")
temp_end = widgets.IntText(value=1000, description="End Temp (K):")
temp_step = widgets.IntText(value=5, description="Temp Step (K):")
eq_steps = widgets.IntText(value=2000, description="Steps per Temp:")
n_gpu_jobs = widgets.IntText(value=1, description="Parallel Jobs:")  # Default 1 for Colab
enable_cooling = widgets.Checkbox(value=False, description="Enable Cooling:")
optimize_only = widgets.Checkbox(value=False, description="Optimize Only:")

# Display widgets
display(sim_mode, project_prefix, temp_start, temp_end, temp_step, eq_steps, n_gpu_jobs, enable_cooling, optimize_only)

# Run Button (simulate with code execution)
print("Run the next cell to start simulation after setting parameters.")

# Step 5: Run Simulation (Execute this cell after setting parameters)
project_name = f"{project_prefix.value}_{os.path.splitext(cif_filename)[0]}_{model_name}_NPT"
if optimize_only.value:
    project_name = project_name.replace("_NPT", "_OPT")

project_path = os.path.join(PROJECTS_DIR, project_name)
if not os.path.exists(project_path):
    os.makedirs(project_path)

start_time = time.time()
atoms = read(cif_filename)

print(f"Optimizing structure for {project_name}...")
opt_atoms, _, _ = optimize_structure(atoms, model_name)

opt_cif_path = os.path.join(project_path, "optimized_structure.cif")
write(opt_cif_path, opt_atoms, format="cif")

if optimize_only.value:
    elapsed_time = time.time() - start_time
    with open(os.path.join(project_path, "execution_time.txt"), "w") as f:
        f.write(f"{elapsed_time:.2f}")
    print(f"Optimization finished: {project_name}. Time: {elapsed_time:.2f} sec.")
else:
    print(f"Starting NPT simulation for {project_name}...")
    temp_range = (temp_start.value, temp_end.value, temp_step.value)
    traj_filepath = os.path.join(project_path, "trajectory.xyz")

    def progress_callback(current, total, message, partial_df=None):
        print(message)

    npt_df_heating, heating_final_struct = run_npt_simulation_parallel(
        initial_atoms=opt_atoms, model_name=model_name, sim_mode=sim_mode.value,
        temp_range=temp_range, time_step=1.0, eq_steps=eq_steps.value, pressure=1.0,
        n_gpu_jobs=n_gpu_jobs.value, progress_callback=progress_callback,
        traj_filepath=traj_filepath, append_traj=False
    )

    npt_df = npt_df_heating

    if enable_cooling.value:
        print(f"Starting cooling phase for {project_name}...")
        cooling_temp_range = (temp_end.value, temp_start.value, -temp_step.value)
        cooling_initial_atoms = Atoms(**heating_final_struct)
        npt_df_cooling, _ = run_npt_simulation_parallel(
            initial_atoms=cooling_initial_atoms, model_name=model_name, sim_mode=sim_mode.value,
            temp_range=cooling_temp_range, time_step=1.0, eq_steps=eq_steps.value, pressure=1.0,
            n_gpu_jobs=n_gpu_jobs.value, progress_callback=progress_callback,
            traj_filepath=traj_filepath, append_traj=True
        )
        if not npt_df_cooling.empty:
            npt_df = pd.concat([npt_df_heating, npt_df_cooling], ignore_index=True)

    if not npt_df.empty:
        elapsed_time = time.time() - start_time
        with open(os.path.join(project_path, "execution_time.txt"), "w") as f:
            f.write(f"{elapsed_time:.2f}")

        fig_temp = plot_temperature_dependent_properties(npt_df)
        fig_temp.savefig(os.path.join(project_path, "npt_vs_temp.png"))
        plt.close(fig_temp)
        plt.show()  # Display plot in Colab

        npt_df.to_csv(os.path.join(project_path, "npt_summary_full.csv"), index=False)

        npt_df.groupby('set_temps').last().reset_index().to_csv(
            os.path.join(project_path, "npt_last_steps.csv"), index=False)

        agg_cols = [col for col in npt_df.columns if col != 'set_temps']
        stats_df = npt_df.groupby('set_temps')[agg_cols].agg(['mean', 'std'])
        stats_df.columns = ['_'.join(map(str, col)).strip() for col in stats_df.columns.values]
        rename_mapping = {f'{col}_mean': col for col in agg_cols}
        stats_df = stats_df.rename(columns=rename_mapping).reset_index()
        stats_df.to_csv(os.path.join(project_path, "npt_summary_stats.csv"), index=False, float_format='%.6f')

        print(f"NPT simulation finished: {project_name}. Time: {elapsed_time:.2f} sec.")

# Step 6: Download Outputs
files_to_download = [
    "npt_summary_full.csv", "npt_summary_stats.csv", "npt_last_steps.csv",
    "trajectory.xyz", "optimized_structure.cif", "npt_vs_temp.png", "execution_time.txt"
]

for filename in files_to_download:
    filepath = os.path.join(project_path, filename)
    if os.path.exists(filepath):
        files.download(filepath)
