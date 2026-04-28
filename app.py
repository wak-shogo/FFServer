# app.py
import streamlit as st
import pandas as pd
from ase.io import read, write
from io import StringIO, BytesIO
import os
import shutil
import time
import matplotlib.pyplot as plt
import json
from datetime import datetime
import simulation_utils as sim
import visualization as viz
import notifications as notify
import cif_editor as cif_edit
from streamlit_autorefresh import st_autorefresh
import subprocess
# --- 定数と状態管理ファイルのパス ---
PROJECTS_DIR = "simulation_projects"
QUEUE_FILE = os.path.join(PROJECTS_DIR, "queue.json")
CURRENT_JOB_FILE = os.path.join(PROJECTS_DIR, "current_job.json")
REALTIME_DATA_FILE = os.path.join(PROJECTS_DIR, "realtime_data.csv")
CANCEL_FLAG_FILE = os.path.join(PROJECTS_DIR, "cancel_job.flag")
if not os.path.exists(PROJECTS_DIR): os.makedirs(PROJECTS_DIR)
# --- 状態読み込み関数 ---
def get_queue():
    if not os.path.exists(QUEUE_FILE): return []
    try:
        with open(QUEUE_FILE, 'r') as f: return json.load(f)
    except (IOError, json.JSONDecodeError): return []
def get_current_job():
    if not os.path.exists(CURRENT_JOB_FILE): return None
    try:
        with open(CURRENT_JOB_FILE, 'r') as f: return json.load(f)
    except (IOError, json.JSONDecodeError): return None
# --- UI設定 ---
st.set_page_config(page_title="Universal MD Simulator", layout="wide")
st.title("🧪 Universal MD Simulator")
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Simulation Dashboard", "📂 Project Browser", "🛠️ CIF Structure Editor", "🔬 Structure Optimizer"])
# --- サイドバー ---
with st.sidebar:
    st.header("1. Model Selection")
    selected_model = st.selectbox("Select ML Force Field", ["CHGNet", "CHGNet_r2SCAN", "MatterSim", "Orb", "NequipOLM", "MatRIS"])
    sim_mode = st.selectbox("Simulation Mode", ["Realistic (ISIF=3)", "Legacy (Orthorombic)"])
    st.header("2. Structure Input")
    uploaded_files = st.file_uploader("Upload CIF Files for NPT Simulation", type=["cif"], accept_multiple_files=True)
    st.header("3. Project Settings")
    default_prefix = datetime.now().strftime("%Y%m%d")
    project_prefix = st.text_input("Project Name Prefix", value=default_prefix)
    st.header("4. NPT Simulation Parameters")
    with st.expander("NPT Simulation Details", expanded=True):
        magmom_specie = st.text_input("Species for Magmom Tracking", "Co") if selected_model == "CHGNet" else None
        
        temp_start, temp_end = st.number_input("Start Temp (K)", 1), st.number_input("End Temp (K)", 1000)
        temp_step, eq_steps = st.number_input("Temp Step (K)", 1,100,5), st.number_input("Steps per Temp", 2000)
        n_gpu_jobs = st.slider("Parallel Jobs", 1, 8, 3)
        # ✅ --- ここに追加 --- (冷却オプションのトグル)
        enable_cooling = st.checkbox("Enable Cooling After Heating (same rate)")

    if st.button("➕ Add NPT Jobs to Queue", type="primary", use_container_width=True):
        if not project_prefix: st.sidebar.error("Project Name Prefix cannot be empty.")
        elif not uploaded_files: st.sidebar.error("Please upload at least one CIF file for NPT simulation.")
        else:
            queue = get_queue()
            for uploaded_file in uploaded_files:
                # Read atoms to check count
                try:
                    # We need to save the file temporarily to read it with ASE
                    temp_path = os.path.join(PROJECTS_DIR, f"temp_{uploaded_file.name}")
                    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    atoms_temp = read(temp_path)
                    os.remove(temp_path)
                    
                    total_atoms = len(atoms_temp)
                    
                    # Atomic count constraint check
                    if total_atoms > 1080:
                        st.error(f"❌ REJECTED: `{uploaded_file.name}` has too many atoms ({total_atoms}). Limit is 1080.")
                        continue
                    elif total_atoms > 320 and n_gpu_jobs > 1:
                        st.warning(f"⚠️ `{uploaded_file.name}` ({total_atoms} atoms) requires Single Job mode (Parallel Jobs=1) due to memory constraints. Overriding setting.")
                        effective_gpu_jobs = 1
                    else:
                        effective_gpu_jobs = n_gpu_jobs
                except Exception as e:
                    st.error(f"Error reading `{uploaded_file.name}`: {e}")
                    continue

                base_filename = os.path.splitext(uploaded_file.name)[0]
                final_project_name = f"{project_prefix}_{base_filename}_{selected_model}_NPT"
                job_info = {
                    "job_type": "full_simulation", # 🔑 NPTジョブのタイプを明記
                    "original_filename": uploaded_file.name,
                    "project_name": final_project_name,
                    "model": selected_model,
                    "sim_mode": sim_mode,
                    "params": {
                        "magmom_specie": magmom_specie,
                        "temp_range": (temp_start, temp_end, temp_step),
                        "eq_steps": eq_steps,
                        "n_gpu_jobs": effective_gpu_jobs,
                        "enable_cooling": enable_cooling
                    }
                }
                queue.append(job_info)
                with open(os.path.join(PROJECTS_DIR, uploaded_file.name), "wb") as f: f.write(uploaded_file.getbuffer())
            
            with open(QUEUE_FILE, 'w') as f: json.dump(queue, f)
            notify.send_to_discord(f"✅ {len(uploaded_files)} NPT job(s) added to queue.")
            st.rerun()
    st.header("5. Control")
    if st.button("🔄 Refresh Status"): st.rerun()

    # --- Maintenance Section ---
    st.title("⚠️ Maintenance")
    if st.button("Force Restart All Processes"):
        # Use session state to manage the confirmation flow
        st.session_state.confirm_restart = True

    # If the button has been clicked, show the confirmation dialog
    if 'confirm_restart' in st.session_state and st.session_state.confirm_restart:
        st.warning("**Are you sure?** This will delete all queued and running jobs and restart the entire application.")
        
        # Use columns for side-by-side buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Confirm Restart"):
                st.session_state.confirm_restart = False
                
                # The restart script is in the root of the workspace
                restart_script_path = "/workspace/force_restart.sh"
                
                try:
                    # Ensure the script is executable
                    subprocess.run(["chmod", "+x", restart_script_path], check=True, capture_output=True, text=True)
                    
                    # Execute the script in the background
                    # We use Popen so the Streamlit app doesn't wait for it to finish
                    subprocess.Popen([restart_script_path], cwd="/workspace")
                    
                    st.success("Restart command issued! The application will restart shortly.")
                    st.info("You may need to refresh this page in 20-30 seconds.")
                    
                    # Optional: A small delay to let the user read the message
                    time.sleep(5)
                    # Clear the confirmation state and rerun to reset the UI
                    st.session_state.confirm_restart = False
                    st.rerun()

                except FileNotFoundError:
                    st.error(f"Error: Restart script not found at {restart_script_path}")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error making restart script executable: {e.stderr}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

        with col2:
            if st.button("❌ Cancel"):
                st.session_state.confirm_restart = False
                st.rerun()
# --- タブ1: Simulation Dashboard ---
with tab1:
    st.header("Job Queue Status")
    current_job = get_current_job(); queue = get_queue()
    if current_job:
        st_autorefresh(interval=60 * 1000, key="data_refresher")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("▶️ Now Running")
        if current_job:
            job_type_str = "Optimization" if current_job.get("job_type") == "optimize_only" else "NPT Simulation"
            st.info(f"**{current_job['project_name']}** ({job_type_str})")
            
            if st.button("🛑 Stop Current Job", type="secondary", use_container_width=True):
                with open(CANCEL_FLAG_FILE, "w") as f:
                    f.write("cancel")
                st.warning("Cancellation request sent. The worker will stop the current job shortly.")
                time.sleep(1)
                st.rerun()
        else:
            st.info("No job is currently running.")
            if os.path.exists(CANCEL_FLAG_FILE):
                os.remove(CANCEL_FLAG_FILE)

    with col2:
        st.subheader("📋 Waiting Queue")
        if queue:
            queue_df = pd.DataFrame([{"Project": job['project_name'], "Model": job['model']} for job in queue])
            st.dataframe(queue_df, use_container_width=True)
            
            # --- Delete Jobs from Queue ---
            with st.expander("🗑️ Delete Jobs from Queue"):
                selected_jobs = st.multiselect(
                    "Select jobs to remove:",
                    options=list(range(len(queue))),
                    format_func=lambda i: f"#{i}: {queue[i]['project_name']}"
                )
                if st.button("Remove Selected Jobs", type="primary"):
                    if selected_jobs:
                        new_queue = [job for i, job in enumerate(queue) if i not in selected_jobs]
                        with open(QUEUE_FILE, 'w') as f:
                            json.dump(new_queue, f)
                        st.success(f"Removed {len(selected_jobs)} jobs from queue.")
                        st.rerun()
        else:
            st.info("Queue is empty.")
    st.subheader("📈 Live Monitoring (NPT only)")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("##### Lattice Parameters (a, b, c)"); chart_placeholder_abc = st.empty()
    with col_chart2:
        st.markdown("##### Volume"); chart_placeholder_v = st.empty()
    if current_job and current_job.get("job_type") != "optimize_only":
        try:
            df = pd.read_csv(REALTIME_DATA_FILE)
            fig_abc, ax_abc = plt.subplots(); ax_abc.plot(df.index, df[['a_lengths', 'b_lengths', 'c_lengths']]); ax_abc.legend(['a','b','c']); ax_abc.set_xlabel("Steps"); ax_abc.set_ylabel("Å"); ax_abc.grid(True); chart_placeholder_abc.pyplot(fig_abc, clear_figure=True); plt.close(fig_abc)
            fig_v, ax_v = plt.subplots(); ax_v.plot(df.index, df['volumes'], color='g'); ax_v.set_xlabel("Steps"); ax_v.set_ylabel("Å³"); ax_v.grid(True); chart_placeholder_v.pyplot(fig_v, clear_figure=True); plt.close(fig_v)
        except Exception:
            chart_placeholder_abc.info("Waiting for first batch to complete...")
# --- タブ2: Project Browser ---
with tab2:
    st.header("📂 Saved Project Browser")
    projects = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
    if not projects: st.info("No projects found.")
    else:
        selected_project = st.selectbox("Select a project", sorted(projects, reverse=True))
        if selected_project:
            project_path = os.path.join(PROJECTS_DIR, selected_project)
            st.subheader(f"Results for: `{selected_project}`")

            # ✅ --- ここから復活 --- (実行時間とプロット表示のコードを再追加)
            # 実行時間を表示
            time_file = os.path.join(project_path, "execution_time.txt")
            if os.path.exists(time_file):
                with open(time_file, "r") as f:
                    st.metric("Total Calculation Time", f"{f.read()} seconds")
            
            # 温度依存性のプロット画像を表示
            st.subheader("Temperature-Dependent Properties")
            npt_vs_temp_png = os.path.join(project_path, "npt_vs_temp.png")
            if os.path.exists(npt_vs_temp_png):
                st.image(npt_vs_temp_png)
            else:
                st.info("NPT summary plot not found for this project (or it was an optimization-only job).")
            # ✅ --- ここまで復活 ---

            st.subheader("Download Artifacts")
            
            # ダウンロードファイルリスト (変更なし)
            files_to_download = {
                "Full Data (CSV)": "npt_summary_full.csv",
                "Statistical Summary (CSV)": "npt_summary_stats.csv",
                "Last Step Data (CSV)": "npt_last_steps.csv",
                "Magmoms per Atom (CSV)": "magmoms_per_atom.csv",
                "Trajectory (XYZ)": "trajectory.xyz",
                "Smoothed Traj (XYZ)": "trajectory_smoothed.xyz",
                "Optimized Structure (CIF)": "optimized_structure.cif"
            }
            
            available_files = {label: filename for label, filename in files_to_download.items() if os.path.exists(os.path.join(project_path, filename))}
            
            if available_files:
                num_files = len(available_files)
                cols = st.columns(num_files if num_files <= 5 else 5)
                col_idx = 0
                for label, filename in available_files.items():
                    current_col = cols[col_idx % 5]
                    filepath = os.path.join(project_path, filename)
                    with current_col, open(filepath, "rb") as f:
                        st.download_button(label, f.read(), file_name=filename, use_container_width=True)
                    col_idx += 1
            else:
                st.warning("No downloadable files found for this project.")
            
            # ✅ --- Analysis Results ZIP Section ---
            st.subheader("📊 Batch Analysis Results")
            
            # List generated plots
            analysis_plots = [f for f in os.listdir(project_path) if f.endswith('.png') and "npt_vs_temp" not in f]
            if analysis_plots:
                with st.expander(f"View Generated Analysis Plots ({len(analysis_plots)})"):
                    # Grid layout for images
                    p_cols = st.columns(3)
                    for i, p in enumerate(sorted(analysis_plots)):
                        with p_cols[i % 3]:
                            st.image(os.path.join(project_path, p), caption=p, use_container_width=True)

            # ZIP Creation
            zip_base_name = "analysis_results"
            zip_filepath = os.path.join(project_path, f"{zip_base_name}.zip")
            
            col_zip1, col_zip2 = st.columns([1, 2])
            with col_zip1:
                if st.button("📦 Create/Update Analysis ZIP"):
                    # Select files to include (png, csv, smoothed traj)
                    files_to_zip = [f for f in os.listdir(project_path) if f.endswith('.png') or f.endswith('.csv') or f.endswith('smoothed.xyz')]
                    
                    if not files_to_zip:
                        st.warning("No analysis files found to zip.")
                    else:
                        # Create a temp dir to zip structure
                        temp_zip_dir = os.path.join(project_path, "temp_analysis_pack")
                        if os.path.exists(temp_zip_dir): shutil.rmtree(temp_zip_dir)
                        os.makedirs(temp_zip_dir)
                        
                        for f in files_to_zip:
                            shutil.copy2(os.path.join(project_path, f), os.path.join(temp_zip_dir, f))
                        
                        shutil.make_archive(os.path.join(project_path, zip_base_name), 'zip', temp_zip_dir)
                        shutil.rmtree(temp_zip_dir)
                        st.success(f"ZIP created with {len(files_to_zip)} files!")
                        st.rerun()

            with col_zip2:
                if os.path.exists(zip_filepath):
                    with open(zip_filepath, "rb") as f:
                        st.download_button("⬇️ Download All Analysis (ZIP)", f.read(), file_name=f"{selected_project}_analysis.zip", mime="application/zip", use_container_width=True)
            # ---------------------------------------

            if st.button(f"🗑️ Delete Project '{selected_project}'", use_container_width=True):
                shutil.rmtree(project_path); st.success(f"Project '{selected_project}' deleted."); st.rerun()
# --- タブ3: CIF Structure Editor ---
# --- タブ3: CIF Structure Editor ---
with tab3:
    st.header("🛠️ CIF Structure Editor")
    st.write("Upload a CIF file to replace a percentage of one element with another.")
    editor_file = st.file_uploader("Upload CIF for Editing", type=["cif"], key="editor_uploader")
    
    if editor_file:
        # カラムを5つに増やしてモード選択を追加
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1: replace_from = st.text_input("Element to Replace", "Bi")
        with col2: replace_to = st.text_input("Replace With", "Mg")
        with col3: percentage = st.number_input("Percentage (%)", 0.0, 100.0, 10.0, 1.0)
        # 新しいオプション: 置換モードの選択
        with col4: mode_select = st.selectbox("Mode", ["Random", "Sequential"], help="Random: Randomly select sites.\nSequential: Select sites from top to bottom.")
        with col5: st.write(""); st.write(""); generate_button = st.button("Generate", type="primary")
        
        if generate_button:
            input_content = editor_file.getvalue().decode("utf-8")
            
            # モードの文字列を小文字にして関数に渡す ('Random' -> 'random')
            modified_content, log_messages = cif_edit.modify_cif_content(
                input_content, 
                replace_from, 
                replace_to, 
                percentage / 100.0, 
                mode=mode_select.lower()
            )
            
            st.subheader("Processing Log")
            for msg in log_messages:
                if "Error" in msg: st.error(msg)
                elif "Warning" in msg: st.warning(msg)
                else: st.info(msg)
            
            if "Success" in " ".join(log_messages):
                st.subheader("Download Modified File")
                # ファイル名にモードも含めると管理しやすいです
                mode_suffix = "seq" if mode_select == "Sequential" else "rnd"
                new_filename = f"{os.path.splitext(editor_file.name)[0]}_{replace_from}2{replace_to}_{int(percentage)}pct_{mode_suffix}.cif"
                st.download_button(label=f"Download '{new_filename}'", data=modified_content, file_name=new_filename, mime="chemical/x-cif")
with tab4:
    st.header("🔬 Structure Optimizer")
    st.write("Upload a CIF file to perform only structure optimization and get the final CIF.")
    opt_file = st.file_uploader("Upload CIF for Optimization", type=["cif"], key="optimizer_uploader")
   
    col1, col2 = st.columns(2)
    with col1:
        opt_model = st.selectbox("Select ML Force Field", ["CHGNet", "CHGNet_r2SCAN", "MatterSim", "Orb", "NequipOLM", "MatRIS"], key="optimizer_model")
    with col2:
        opt_prefix = st.text_input("Project Name Prefix", value=datetime.now().strftime("%Y%m%d"), key="optimizer_prefix")
    
    if st.button("⚙️ Optimize Structure & Add to Queue", type="primary", use_container_width=True):
        if not opt_prefix:
            st.error("Project Name Prefix cannot be empty.")
        elif not opt_file:
            st.error("Please upload a CIF file for optimization.")
        else:
            # Read atoms to check count
            try:
                temp_path = os.path.join(PROJECTS_DIR, f"temp_opt_{opt_file.name}")
                with open(temp_path, "wb") as f: f.write(opt_file.getbuffer())
                atoms_temp = read(temp_path)
                os.remove(temp_path)
                
                total_atoms = len(atoms_temp)
                
                if total_atoms > 1080:
                    st.error(f"❌ REJECTED: `{opt_file.name}` has too many atoms ({total_atoms}). Limit is 1080.")
                    st.stop()
            except Exception as e:
                st.error(f"Error reading `{opt_file.name}`: {e}")
                st.stop()

            base_filename = os.path.splitext(opt_file.name)[0]
            final_project_name = f"{opt_prefix}_{base_filename}_{opt_model}_OPT"
           
            job_info = {
                "job_type": "optimize_only", # 🔑 このジョブの種類を定義
                "original_filename": opt_file.name,
                "project_name": final_project_name,
                "model": opt_model,
            }
           
            queue = get_queue()
            queue.append(job_info)
           
            # アップロードされたファイルを保存
            with open(os.path.join(PROJECTS_DIR, opt_file.name), "wb") as f:
                f.write(opt_file.getbuffer())
               
            # キューを更新
            with open(QUEUE_FILE, 'w') as f:
                json.dump(queue, f)
           
            notify.send_to_discord(f"✅ 1 Optimization job (`{final_project_name}`) added to queue.")
            st.success(f"Job '{final_project_name}' added to the queue!")
            st.rerun()