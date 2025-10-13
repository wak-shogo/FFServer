# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # Numpyをインポート
import nglview
from ase.io import Trajectory
from io import StringIO
import base64
def plot_optimization_history(energies, lattice_constants):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(energies, '-o')
    ax1.set(xlabel='Optimization step', ylabel='Potential Energy (eV)', title='Energy vs. Optimization Step')
    ax1.grid(True)
    ax2.plot(lattice_constants, '-o', color='r')
    ax2.set(xlabel='Optimization step', ylabel='Average Lattice Constant (Å)', title='Lattice Constant vs. Optimization Step')
    ax2.grid(True)
    plt.tight_layout()
    return fig
def plot_npt_results(df, magmom_specie):
    if df.empty: return plt.figure()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    cols = ['energies', 'instant_temps', 'volumes', 'a_lengths', f"{magmom_specie}_magmom"]
    titles = ['Energy', 'Temperature', 'Volume', 'Lattice Parameter', f'Avg {magmom_specie} Magmom']
    ylabels = ['Energy (eV)', 'Temperature (K)', 'Volume (Å³)', 'Lattice Parameters (Å)', f'Magmom (μB)']
   
    for i in range(5):
        axes[i].set(xlabel='Step', title=f'{titles[i]} Evolution', xlim=(0, len(df)), ylabel=ylabels[i])
        axes[i].grid(True)
        if i == 0: axes[i].plot(df.index, df[cols[i]], 'b-')
        if i == 1:
            axes[i].plot(df.index, df[cols[i]], 'r-', label='Instantaneous T')
            axes[i].plot(df.index, df['set_temps'], 'k--', label='Set T', alpha=0.7)
            axes[i].legend()
        if i == 2: axes[i].plot(df.index, df[cols[i]], 'g-')
        if i == 3:
            axes[i].plot(df.index, df['a_lengths'], label='a')
            axes[i].plot(df.index, df['b_lengths'], label='b')
            axes[i].plot(df.index, df['c_lengths'], label='c')
            axes[i].legend()
        if i == 4 and cols[i] in df.columns:
            axes[i].plot(df.index, df[cols[i]], 'o-', color='purple', markersize=2)
   
    axes[5].axis('off')
    plt.tight_layout()
    return fig
# ✅ 新規追加: 温度依存性プロット関数
def plot_temperature_dependent_properties(df, moving_avg_window=100):
    """移動平均を適用し、温度を横軸とする物理量の変化をプロットする。冷却フェーズがあれば色分け"""
    if df.empty or len(df) < moving_avg_window:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data for smoothing", ha="center")
        return fig
    
    # 🔧 --- 追加: フェーズの識別と分離 ---
    # set_tempsの変化を基にphase列を追加（昇温: heating, 降温: cooling）
    df = df.copy()  # コピーして列追加
    df['phase'] = 'heating'  # デフォルト
    temp_diff = df['set_temps'].diff()
    cooling_start_idx = (temp_diff < 0).idxmax() if (temp_diff < 0).any() else len(df)  # 最初の降温点
    if cooling_start_idx < len(df):
        df.loc[df.index >= cooling_start_idx, 'phase'] = 'cooling'
    
    # 昇温と冷却のデータを分離
    df_heating = df[df['phase'] == 'heating']
    df_cooling = df[df['phase'] == 'cooling']
    has_cooling = not df_cooling.empty
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
   
    # --- 移動平均の計算（全体で一括、ただしプロット時分離） ---
    b = np.ones(moving_avg_window) / moving_avg_window
    temp_mean_all = np.convolve(df["instant_temps"], b, mode='valid')
    # 移動平均のインデックスを調整（convolveのvalidモードのため、window分短くなる）
    adjusted_len = len(temp_mean_all)
    temp_mean_heating = temp_mean_all[:len(df_heating) - moving_avg_window + 1] if has_cooling else temp_mean_all
    temp_mean_cooling = temp_mean_all[len(df_heating) - moving_avg_window + 1 : adjusted_len] if has_cooling else None
   
    # 1. 格子定数 vs 温度
    a_mean_all = np.convolve(df["a_lengths"], b, mode='valid')
    b_mean_all = np.convolve(df["b_lengths"], b, mode='valid')
    c_mean_all = np.convolve(df["c_lengths"], b, mode='valid')
    a_mean_heating = a_mean_all[:len(df_heating) - moving_avg_window + 1] if has_cooling else a_mean_all
    b_mean_heating = b_mean_all[:len(df_heating) - moving_avg_window + 1] if has_cooling else b_mean_all
    c_mean_heating = c_mean_all[:len(df_heating) - moving_avg_window + 1] if has_cooling else c_mean_all
    if has_cooling:
        a_mean_cooling = a_mean_all[len(df_heating) - moving_avg_window + 1 : adjusted_len]
        b_mean_cooling = b_mean_all[len(df_heating) - moving_avg_window + 1 : adjusted_len]
        c_mean_cooling = c_mean_all[len(df_heating) - moving_avg_window + 1 : adjusted_len]
    else:
        a_mean_cooling = b_mean_cooling = c_mean_cooling = np.array([])
   
    ax1 = axes[0]
    # 昇温（青系）
    ax1.plot(temp_mean_heating, a_mean_heating, label="a (heating)", color='blue', linestyle='-')
    ax1.plot(temp_mean_heating, b_mean_heating, label="b (heating)", color='cyan', linestyle='-')
    # 冷却（赤系）
    if has_cooling:
        ax1.plot(temp_mean_cooling, a_mean_cooling, label="a (cooling)", color='red', linestyle='--')
        ax1.plot(temp_mean_cooling, b_mean_cooling, label="b (cooling)", color='orange', linestyle='--')
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("Lattice Parameter (Å)")
    ax1.legend(loc="upper left")
    ax1_twin = ax1.twinx()
    # cも色分け
    ax1_twin.plot(temp_mean_heating, c_mean_heating, label="c (heating)", color='green', linestyle='-')
    if has_cooling:
        ax1_twin.plot(temp_mean_cooling, c_mean_cooling, label="c (cooling)", color='darkred', linestyle='--')
    ax1_twin.set_ylabel("Lattice Parameter c (Å)")
    ax1_twin.legend(loc="upper right")
    ax1.set_title("Lattice Parameters vs. Temperature")
    ax1.grid(True)
   
    # 2. 体積 vs 温度
    vol_mean_all = np.convolve(df["volumes"], b, mode='valid')
    vol_mean_heating = vol_mean_all[:len(df_heating) - moving_avg_window + 1] if has_cooling else vol_mean_all
    if has_cooling:
        vol_mean_cooling = vol_mean_all[len(df_heating) - moving_avg_window + 1 : adjusted_len]
    else:
        vol_mean_cooling = np.array([])
    axes[1].plot(temp_mean_heating, vol_mean_heating, 'b-', label='Heating')
    if has_cooling:
        axes[1].plot(temp_mean_cooling, vol_mean_cooling, 'r--', label='Cooling')
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel("Volume (Å³)")
    axes[1].set_title("Volume vs. Temperature")
    axes[1].legend()
    axes[1].grid(True)
   
    # 3. 角度 vs 温度
    alpha_mean_all = np.convolve(df["alpha"], b, mode='valid')
    beta_mean_all = np.convolve(df["beta"], b, mode='valid')
    gamma_mean_all = np.convolve(df["gamma"], b, mode='valid')
    alpha_mean_heating = alpha_mean_all[:len(df_heating) - moving_avg_window + 1] if has_cooling else alpha_mean_all
    beta_mean_heating = beta_mean_all[:len(df_heating) - moving_avg_window + 1] if has_cooling else beta_mean_all
    gamma_mean_heating = gamma_mean_all[:len(df_heating) - moving_avg_window + 1] if has_cooling else gamma_mean_all
    if has_cooling:
        alpha_mean_cooling = alpha_mean_all[len(df_heating) - moving_avg_window + 1 : adjusted_len]
        beta_mean_cooling = beta_mean_all[len(df_heating) - moving_avg_window + 1 : adjusted_len]
        gamma_mean_cooling = gamma_mean_all[len(df_heating) - moving_avg_window + 1 : adjusted_len]
    else:
        alpha_mean_cooling = beta_mean_cooling = gamma_mean_cooling = np.array([])
    axes[2].plot(temp_mean_heating, alpha_mean_heating, label="α (heating)", color='blue')
    axes[2].plot(temp_mean_heating, beta_mean_heating, label="β (heating)", color='cyan')
    if has_cooling:
        axes[2].plot(temp_mean_cooling, alpha_mean_cooling, label="α (cooling)", color='red', linestyle='--')
        axes[2].plot(temp_mean_cooling, beta_mean_cooling, label="β (cooling)", color='orange', linestyle='--')
    axes[2].plot(temp_mean_heating, gamma_mean_heating, label="γ (heating)", color='green')
    if has_cooling:
        axes[2].plot(temp_mean_cooling, gamma_mean_cooling, label="γ (cooling)", color='darkred', linestyle='--')
    axes[2].set_xlabel("Temperature (K)")
    axes[2].set_ylabel("Lattice Angle (°)")
    axes[2].set_title("Lattice Angles vs. Temperature")
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    return fig
def get_df_download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'