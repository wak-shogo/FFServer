# app.py
import streamlit as st
import pandas as pd
from ase.io import read, write
from io import StringIO, BytesIO
import os
import shutil
import time
import matplotlib.pyplot as plt # ✅ Matplotlibをインポート

import simulation_utils as sim
import visualization as viz
import notifications as notify

PROJECTS_DIR = "simulation_projects"
if not os.path.exists(PROJECTS_DIR):
    os.makedirs(PROJECTS_DIR)

# --- Session Stateの初期化 ---
if 'job_queue' not in st.session_state: st.session_state.job_queue = []
if 'current_job' not in st.session_state: st.session_state.current_job = None
if 'is_running' not in st.session_state: st.session_state.is_running = False

st.set_page_config(page_title="Universal MD Simulator", layout="wide")
st.title("🧪 Universal MD Simulator")

# --- UI定義 ---
tab1, tab2 = st.tabs(["🚀 Simulation Dashboard", "📂 Project Browser"])

with tab1:
    st.header("Job Queue Status")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("▶️ Now Running")
        st.info(st.session_state.current_job or "No job is currently running.")
    with col2:
        st.subheader("📋 Waiting Queue")
        if st.session_state.job_queue:
            queue_display_data = [item['project_name'] for item in st.session_state.job_queue]
            st.dataframe(pd.DataFrame(queue_display_data, columns=["Queued Project Name"]))
        else:
            st.info("Queue is empty.")

    st.subheader("📈 Real-time Monitoring")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("##### Lattice Parameters (a, b, c)")
        chart_placeholder_abc = st.empty()
    with col_chart2:
        st.markdown("##### Volume")
        chart_placeholder_v = st.empty()

with st.sidebar:
    st.header("1. Model Selection")
    selected_model = st.selectbox("Select ML Force Field", ["CHGNet", "MatterSim"])
    sim_mode = st.selectbox("Simulation Mode", ["Realistic (ISIF=3)", "Legacy (Orthorhombic)"])
    
    st.header("2. Structure Input")
    uploaded_files = st.file_uploader("Upload CIF Files", type=["cif"], accept_multiple_files=True)
    
    st.header("3. Project Settings")
    allow_overwrite = st.checkbox("Overwrite existing project?", value=True)
    project_prefix = ""
    if not allow_overwrite:
        project_prefix = st.text_input("Project Name Prefix", value="run2", help="Prefix for new project if name conflicts.")

    st.header("4. Simulation Parameters")
    with st.expander("NPT Simulation", expanded=True):
        magmom_specie = st.text_input("Species for Magmom Tracking", "Co") if selected_model == "CHGNet" else None
        # ✅ 修正点: パラメータのデフォルト値を変更
        temp_start = st.number_input("Start Temp (K)", value=1)
        temp_end = st.number_input("End Temp (K)", value=800)
        temp_step = st.number_input("Temp Step (K)", value=1)
        eq_steps = st.number_input("Steps per Temp", value=100)
        n_gpu_jobs = st.slider("Parallel Jobs", 1, 8, 3)
    
    if st.button("➕ Add to Queue", type="primary", use_container_width=True):
        for uploaded_file in uploaded_files:
            base_filename = os.path.splitext(uploaded_file.name)[0]
            project_name_candidate = f"{base_filename}_{selected_model}"
            final_project_name = project_name_candidate
            
            project_path_candidate = os.path.join(PROJECTS_DIR, project_name_candidate)
            if not allow_overwrite and os.path.exists(project_path_candidate):
                if not project_prefix:
                    st.sidebar.error("Prefix is required to avoid overwrite.")
                    continue
                final_project_name = f"{project_prefix}_{project_name_candidate}"

            job_info = {"original_filename": uploaded_file.name, "project_name": final_project_name}
            st.session_state.job_queue.append(job_info)
            
            with open(os.path.join(PROJECTS_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            notify.send_to_discord(f"✅ Job added to queue: `{final_project_name}`")
    
    st.header("5. Progress")
    progress_text = st.empty()
    progress_bar = st.progress(0)

# --- メイン処理ループ ---
if not st.session_state.is_running and st.session_state.job_queue:
    st.session_state.is_running = True
    job_info = st.session_state.job_queue.pop(0)
    st.session_state.current_job = job_info['project_name']
    st.rerun()

if st.session_state.is_running:
    project_name = st.session_state.current_job
    job_parts = project_name.split('_')
    original_filename = ""
    if job_parts[-1] in ["CHGNet", "MatterSim"]:
        original_basename = "_".join(job_parts[:-1])
        if not allow_overwrite and project_prefix and project_name.startswith(project_prefix):
             original_basename = original_basename[len(project_prefix)+1:]
        original_filename = f"{original_basename}.cif"

    project_path = os.path.join(PROJECTS_DIR, project_name)
    if not os.path.exists(project_path): os.makedirs(project_path)
    
    try:
        start_time = time.time()
        atoms = read(os.path.join(PROJECTS_DIR, original_filename))
        
        st.subheader("Geometry Optimization")
        with st.spinner(f"Optimizing {project_name}..."):
            progress_text.text("Optimizing initial structure...")
            initial_vol, initial_params = atoms.get_volume(), atoms.cell.cellpar()
            opt_atoms, _, _ = sim.optimize_structure(atoms, model_name=selected_model, fmax=0.01)
            final_vol, final_params = opt_atoms.get_volume(), opt_atoms.cell.cellpar()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Initial Structure**"); st.metric("Volume (Å³)", f"{initial_vol:.2f}")
            st.text(f"a={initial_params[0]:.3f}, b={initial_params[1]:.3f}, c={initial_params[2]:.3f}\nα={initial_params[3]:.2f}, β={initial_params[4]:.2f}, γ={initial_params[5]:.2f}")
        with col2:
            st.write("**Optimized Structure**"); st.metric("Volume (Å³)", f"{final_vol:.2f}", delta=f"{final_vol - initial_vol:.2f}")
            st.text(f"a={final_params[0]:.3f}, b={final_params[1]:.3f}, c={final_params[2]:.3f}\nα={final_params[3]:.2f}, β={final_params[4]:.2f}, γ={final_params[5]:.2f}")
        
        # ✅ 修正点: リアルタイムグラフをMatplotlibに変更
        def update_progress(current, total, message, partial_df):
            progress_bar.progress(current / total if total > 0 else 0)
            progress_text.text(f"NPT Status: {message}")
            if not partial_df.empty:
                # グラフ1: 格子定数
                fig_abc, ax_abc = plt.subplots()
                ax_abc.plot(partial_df.index, partial_df['a_lengths'], label='a')
                ax_abc.plot(partial_df.index, partial_df['b_lengths'], label='b')
                ax_abc.plot(partial_df.index, partial_df['c_lengths'], label='c')
                ax_abc.set_xlabel("Total Steps"); ax_abc.set_ylabel("Lattice Param (Å)")
                ax_abc.grid(True); ax_abc.legend()
                chart_placeholder_abc.pyplot(fig_abc, clear_figure=True)
                
                # グラフ2: 体積
                fig_v, ax_v = plt.subplots()
                ax_v.plot(partial_df.index, partial_df['volumes'], color='g')
                ax_v.set_xlabel("Total Steps"); ax_v.set_ylabel("Volume (Å³)")
                ax_v.grid(True)
                chart_placeholder_v.pyplot(fig_v, clear_figure=True)
                
                # pltオブジェクトを閉じてメモリを解放
                plt.close(fig_abc)
                plt.close(fig_v)

        traj_filepath = os.path.join(project_path, "trajectory.xyz")
        npt_df = sim.run_npt_simulation_parallel(
            initial_atoms=opt_atoms, model_name=selected_model, sim_mode=sim_mode, magmom_specie=magmom_specie,
            temp_range=(temp_start, temp_end, temp_step), time_step=1.0, eq_steps=eq_steps,
            pressure=1.0, n_gpu_jobs=n_gpu_jobs, progress_callback=update_progress, traj_filepath=traj_filepath
        )
        
        if not npt_df.empty:
            elapsed_time = time.time() - start_time
            with open(os.path.join(project_path, "execution_time.txt"), "w") as f: f.write(f"{elapsed_time:.2f}")
            fig_temp = viz.plot_temperature_dependent_properties(npt_df, 100)
            fig_temp.savefig(os.path.join(project_path, "npt_vs_temp.png"))
            npt_df.to_csv(os.path.join(project_path, "npt_summary_full.csv"), index=False)
            npt_df.groupby('set_temps').last().reset_index().to_csv(os.path.join(project_path, "npt_last_steps.csv"), index=False)
            notify.send_to_discord(f"🎉 Simulation finished: `{project_name}`\nTime: {elapsed_time:.2f} sec.", color=3066993)
        else:
             notify.send_to_discord(f"❌ Simulation failed: `{project_name}`.", color=15158332)
    finally:
        st.session_state.current_job = None; st.session_state.is_running = False
        st.rerun()

with tab2:
    st.header("📂 Saved Project Browser")
    projects = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
    if not projects:
        st.info("No projects found.")
    else:
        selected_project = st.selectbox("Select a project", sorted(projects, reverse=True))
        if selected_project:
            project_path = os.path.join(PROJECTS_DIR, selected_project)
            st.subheader(f"Results for: `{selected_project}`")
            time_file = os.path.join(project_path, "execution_time.txt")
            if os.path.exists(time_file):
                with open(time_file, "r") as f: st.metric("Total Calculation Time", f"{f.read()} seconds")
            
            npt_vs_temp_png = os.path.join(project_path, "npt_vs_temp.png")
            if os.path.exists(npt_vs_temp_png):
                with st.expander("Temperature-Dependent Properties", expanded=True): st.image(npt_vs_temp_png)

            st.subheader("Download Artifacts")
            files_to_download = { "Full Data (CSV)": "npt_summary_full.csv", "Last Step Data (CSV)": "npt_last_steps.csv", "Trajectory (XYZ)": "trajectory.xyz" }
            cols = st.columns(len(files_to_download))
            for i, (label, filename) in enumerate(files_to_download.items()):
                filepath = os.path.join(project_path, filename)
                if os.path.exists(filepath):
                    with cols[i], open(filepath, "rb") as f:
                        st.download_button(label, f.read(), file_name=filename)
            if st.button(f"🗑️ Delete Project '{selected_project}'"):
                shutil.rmtree(project_path); st.success(f"Project '{selected_project}' deleted."); st.rerun()