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
    """移動平均を適用し、温度を横軸とする物理量の変化をプロットする"""
    if df.empty or len(df) < moving_avg_window:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data for smoothing", ha="center")
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # --- 移動平均の計算 ---
    b = np.ones(moving_avg_window) / moving_avg_window
    temp_mean = np.convolve(df["instant_temps"], b, mode='valid')
    
    # 1. 格子定数 vs 温度
    a_mean = np.convolve(df["a_lengths"], b, mode='valid')
    b_mean = np.convolve(df["b_lengths"], b, mode='valid')
    c_mean = np.convolve(df["c_lengths"], b, mode='valid')
    
    ax1 = axes[0]
    ax1.plot(temp_mean, a_mean, label="a")
    ax1.plot(temp_mean, b_mean, label="b")
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("Lattice Parameter (Å)")
    ax1.legend(loc="upper left")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(temp_mean, c_mean, label="c", color="g")
    ax1_twin.set_ylabel("Lattice Parameter c (Å)")
    ax1_twin.legend(loc="upper right")
    ax1.set_title("Lattice Parameters vs. Temperature")
    ax1.grid(True)
    
    # 2. 体積 vs 温度
    vol_mean = np.convolve(df["volumes"], b, mode='valid')
    axes[1].plot(temp_mean, vol_mean, 'm-')
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel("Volume (Å³)")
    axes[1].set_title("Volume vs. Temperature")
    axes[1].grid(True)
    
    # 3. 角度 vs 温度
    alpha_mean = np.convolve(df["alpha"], b, mode='valid')
    beta_mean = np.convolve(df["beta"], b, mode='valid')
    gamma_mean = np.convolve(df["gamma"], b, mode='valid')
    axes[2].plot(temp_mean, alpha_mean, label="α")
    axes[2].plot(temp_mean, beta_mean, label="β")
    axes[2].plot(temp_mean, gamma_mean, label="γ")
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