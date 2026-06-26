# api.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import shutil
import pandas as pd
from typing import List
import torch

app = FastAPI(title="Universal MD Simulator API")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECTS_DIR = "/workspace/simulation_projects"
QUEUE_FILE = os.path.join(PROJECTS_DIR, "queue.json")
CURRENT_JOB_FILE = os.path.join(PROJECTS_DIR, "current_job.json")
REALTIME_DATA_FILE = os.path.join(PROJECTS_DIR, "realtime_data.csv")
CANCEL_FLAG_FILE = os.path.join(PROJECTS_DIR, "cancel_job.flag")

# Ensure projects directory exists
os.makedirs(PROJECTS_DIR, exist_ok=True)

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

def save_queue(queue):
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=4)

@app.get("/api/status")
def get_status():
    queue = get_queue()
    current_job = get_current_job()
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "current_job": current_job,
        "queue": queue
    }

@app.post("/api/jobs/npt")
async def add_npt_jobs(
    files: List[UploadFile] = File(...),
    model: str = Form(...),
    sim_mode: str = Form(...),
    project_prefix: str = Form(...),
    magmom_specie: str = Form(""),
    temp_start: int = Form(...),
    temp_end: int = Form(...),
    temp_step: int = Form(...),
    eq_steps: int = Form(...),
    n_gpu_jobs: int = Form(1),
    enable_cooling: bool = Form(False)
):
    import ase.io
    queue = get_queue()
    added_jobs = 0
    rejected_jobs = []
    
    for file in files:
        temp_path = os.path.join(PROJECTS_DIR, f"temp_{file.filename}")
        try:
            # Save uploaded file temporarily
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            
            # Read atoms to check count constraints
            atoms_temp = ase.io.read(temp_path)
            total_atoms = len(atoms_temp)
            
            if total_atoms > 1080:
                rejected_jobs.append(f"{file.filename} (rejected: {total_atoms} atoms > 1080 limit)")
                os.remove(temp_path)
                continue
                
            effective_gpu_jobs = n_gpu_jobs
            if total_atoms > 320 and n_gpu_jobs > 1:
                effective_gpu_jobs = 1
                
            # Copy temp file to permanent location in simulation_projects/
            perm_path = os.path.join(PROJECTS_DIR, file.filename)
            shutil.copy2(temp_path, perm_path)
            os.remove(temp_path)
            
            base_filename = os.path.splitext(file.filename)[0]
            final_project_name = f"{project_prefix}_{base_filename}_{model}_NPT"
            
            job_info = {
                "job_type": "full_simulation",
                "original_filename": file.filename,
                "project_name": final_project_name,
                "model": model,
                "sim_mode": sim_mode,
                "params": {
                    "magmom_specie": magmom_specie,
                    "temp_range": [temp_start, temp_end, temp_step],
                    "eq_steps": eq_steps,
                    "n_gpu_jobs": effective_gpu_jobs,
                    "enable_cooling": enable_cooling
                }
            }
            queue.append(job_info)
            added_jobs += 1
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Error reading {file.filename}: {str(e)}")
            
    save_queue(queue)
    
    try:
        import notifications as notify
        notify.send_to_discord(f"✅ {added_jobs} NPT job(s) added to queue.")
    except Exception:
        pass
        
    return {
        "message": f"Successfully queued {added_jobs} jobs. {len(rejected_jobs)} files rejected.",
        "rejected": rejected_jobs
    }

@app.post("/api/jobs/optimize")
async def add_optimize_job(
    files: List[UploadFile] = File(...),
    model: str = Form(...),
    project_prefix: str = Form(...)
):
    import ase.io
    queue = get_queue()
    added_jobs = 0
    rejected_jobs = []
    
    for file in files:
        temp_path = os.path.join(PROJECTS_DIR, f"temp_opt_{file.filename}")
        try:
            with open(temp_path, "wb") as f:
                f.write(await file.read())
                
            atoms_temp = ase.io.read(temp_path)
            total_atoms = len(atoms_temp)
            
            if total_atoms > 1080:
                rejected_jobs.append(f"{file.filename} (rejected: {total_atoms} atoms > 1080 limit)")
                os.remove(temp_path)
                continue
                
            perm_path = os.path.join(PROJECTS_DIR, file.filename)
            shutil.copy2(temp_path, perm_path)
            os.remove(temp_path)
            
            base_filename = os.path.splitext(file.filename)[0]
            final_project_name = f"{project_prefix}_{base_filename}_{model}_OPT"
            
            job_info = {
                "job_type": "optimize_only",
                "original_filename": file.filename,
                "project_name": final_project_name,
                "model": model,
            }
            queue.append(job_info)
            added_jobs += 1
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Error reading {file.filename}: {str(e)}")
            
    save_queue(queue)
    
    try:
        import notifications as notify
        notify.send_to_discord(f"✅ Queued {added_jobs} Optimization job(s) to queue.")
    except Exception:
        pass
        
    return {
        "message": f"Successfully queued {added_jobs} optimization jobs. {len(rejected_jobs)} files rejected.",
        "rejected": rejected_jobs
    }

@app.get("/api/jobs/realtime")
def get_realtime():
    if not os.path.exists(REALTIME_DATA_FILE):
        return []
    try:
        df = pd.read_csv(REALTIME_DATA_FILE)
        return df.to_dict(orient="records")
    except Exception:
        return []

@app.post("/api/jobs/cancel")
def cancel_job():
    try:
        with open(CANCEL_FLAG_FILE, "w") as f:
            f.write("cancel")
        return {"message": "Cancellation flag set."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

class QueueDeleteRequest(BaseModel):
    index: int

@app.delete("/api/jobs/queue")
def delete_queue_job(req: QueueDeleteRequest):
    queue = get_queue()
    if 0 <= req.index < len(queue):
        removed = queue.pop(req.index)
        save_queue(queue)
        return {"message": f"Removed job '{removed['project_name']}' from queue."}
    else:
        raise HTTPException(status_code=400, detail="Invalid queue index.")

@app.get("/api/projects")
def list_projects():
    if not os.path.exists(PROJECTS_DIR):
        return {"projects": []}
    projects = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
    return {"projects": sorted(projects, reverse=True)}

@app.get("/api/projects/{name}")
def get_project_details(name: str):
    project_path = os.path.join(PROJECTS_DIR, name)
    if not os.path.exists(project_path) or not os.path.isdir(project_path):
        raise HTTPException(status_code=404, detail="Project not found.")
        
    execution_time = "N/A"
    time_file = os.path.join(project_path, "execution_time.txt")
    if os.path.exists(time_file):
        try:
            with open(time_file, "r") as f:
                execution_time = f.read().strip()
        except:
            pass
            
    all_files = os.listdir(project_path)
    plots = [f for f in all_files if f.endswith('.png')]
    
    files_to_download = {
        "npt_summary_full.csv",
        "npt_summary_stats.csv",
        "npt_last_steps.csv",
        "magmoms_per_atom.csv",
        "trajectory.xyz",
        "trajectory_smoothed.xyz",
        "optimized_structure.cif"
    }
    available_files = [f for f in all_files if f in files_to_download]
    zip_exists = "analysis_results.zip" in all_files
    
    return {
        "name": name,
        "execution_time": execution_time,
        "plots": sorted(plots),
        "files": sorted(available_files),
        "zip_exists": zip_exists
    }

@app.get("/api/projects/{name}/files/{filename}")
def download_project_file(name: str, filename: str):
    if ".." in name or ".." in filename or name.startswith("/") or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid path.")
        
    file_path = os.path.join(PROJECTS_DIR, name, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path)

@app.post("/api/projects/{name}/zip")
def zip_project_files(name: str):
    project_path = os.path.join(PROJECTS_DIR, name)
    if not os.path.exists(project_path) or not os.path.isdir(project_path):
        raise HTTPException(status_code=404, detail="Project not found.")
        
    try:
        files_to_zip = [f for f in os.listdir(project_path) if f.endswith('.png') or f.endswith('.csv') or f.endswith('smoothed.xyz')]
        if not files_to_zip:
            raise HTTPException(status_code=400, detail="No files found to zip.")
            
        temp_zip_dir = os.path.join(project_path, "temp_analysis_pack")
        if os.path.exists(temp_zip_dir):
            shutil.rmtree(temp_zip_dir)
        os.makedirs(temp_zip_dir)
        
        for f in files_to_zip:
            shutil.copy2(os.path.join(project_path, f), os.path.join(temp_zip_dir, f))
            
        shutil.make_archive(os.path.join(project_path, "analysis_results"), 'zip', temp_zip_dir)
        shutil.rmtree(temp_zip_dir)
        return {"message": "ZIP package updated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/projects/{name}")
def delete_project(name: str):
    project_path = os.path.join(PROJECTS_DIR, name)
    if not os.path.exists(project_path) or not os.path.isdir(project_path):
        raise HTTPException(status_code=404, detail="Project not found.")
    try:
        shutil.rmtree(project_path)
        return {"message": f"Project '{name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")

@app.post("/api/cif/edit")
async def edit_cif(
    file: UploadFile = File(...),
    operation: str = Form(...),
    replace_from: str = Form(...),
    replace_to: str = Form(""),
    percentage: float = Form(...),
    mode: str = Form(...)
):
    try:
        input_content = (await file.read()).decode("utf-8")
        import cif_editor as cif_edit
        
        is_vacancy = (operation == "Vacancy")
        modified_content, log_messages = cif_edit.modify_cif_content(
            input_content,
            replace_from,
            replace_to,
            percentage / 100.0,
            mode=mode.lower(),
            is_vacancy=is_vacancy
        )
        
        mode_suffix = "seq" if mode.lower() == "sequential" else "rnd"
        op_suffix = "vac" if is_vacancy else f"2{replace_to}"
        base_name = os.path.splitext(file.filename)[0]
        new_filename = f"{base_name}_{replace_from}{op_suffix}_{int(percentage)}pct_{mode_suffix}.cif"
        
        return {
            "new_filename": new_filename,
            "content": modified_content,
            "logs": log_messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CIF: {str(e)}")

@app.post("/api/maintenance/restart")
def maintenance_restart():
    restart_script_path = "/workspace/force_restart.sh"
    import subprocess
    import os
    try:
        subprocess.run(["chmod", "+x", restart_script_path], check=True, capture_output=True, text=True)
        # Detach child process using a new session group (os.setsid) and redirect stdio
        subprocess.Popen(
            [restart_script_path],
            cwd="/workspace",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        return {"message": "Restart script triggered."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute restart script: {str(e)}")

# Mount React static files (must be at the bottom)
FRONTEND_DIST = "frontend/dist"
if not os.path.exists(FRONTEND_DIST):
    FRONTEND_DIST = "/opt/frontend/dist"

if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")
    
    # Catch-all route to serve index.html for React router
    @app.get("/{path_name:path}")
    def serve_frontend(path_name: str):
        index_path = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
else:
    @app.get("/")
    def index_fallback():
        return {"message": "FastAPI is running, but React frontend dist directory was not found. Please build the frontend."}
