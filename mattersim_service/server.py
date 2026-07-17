import os
import threading
import torch

# ASE constraints monkeypatch for older MatterSim compatibility with newer ASE versions
import ase.constraints
if not hasattr(ase.constraints, 'full_3x3_to_voigt_6_stress'):
    try:
        import ase.stress
        ase.constraints.full_3x3_to_voigt_6_stress = ase.stress.full_3x3_to_voigt_6_stress
    except ImportError:
        pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from mattersim.forcefield import MatterSimCalculator
from ase import Atoms


app = FastAPI(title="MatterSim Service")

# Cache to store loaded MatterSimCalculators
_CALC_CACHE = {}
eval_lock = threading.Lock()

class StructureRequest(BaseModel):
    numbers: List[int]
    positions: List[List[float]]
    cell: List[List[float]]
    pbc: List[bool]
    model_name: Optional[str] = "MatterSim-v1.0.0-1M"
    device: Optional[str] = "cuda"

def get_calculator(model_name: str, device: str):
    # Ensure model name ends with .pth if it's a file name
    if not model_name.endswith(".pth"):
        model_name = f"{model_name}.pth"
    
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cache_key = (model_name, device)
    if cache_key in _CALC_CACHE:
        return _CALC_CACHE[cache_key]

    print(f"Initializing MatterSimCalculator with load_path={model_name} on device={device}")
    try:
        calc = MatterSimCalculator(load_path=model_name, device=device)
        _CALC_CACHE[cache_key] = calc
        return calc
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try fallback to default init (without load_path) if it's 1M
        if "1M" in model_name:
            try:
                calc = MatterSimCalculator(device=device)
                _CALC_CACHE[cache_key] = calc
                return calc
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Failed to load default Mattersim model: {str(e2)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cached_models": list(_CALC_CACHE.keys())
    }

@app.post("/evaluate")
def evaluate(req: StructureRequest):
    try:
        calc = get_calculator(req.model_name, req.device)
        atoms = Atoms(
            numbers=req.numbers,
            positions=req.positions,
            cell=req.cell,
            pbc=req.pbc
        )
        atoms.calc = calc
        
        with eval_lock:
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces().tolist()
            stress = atoms.get_stress(voigt=True).tolist()
            
        return {
            "energy": energy,
            "forces": forces,
            "stress": stress
        }
    except Exception as e:
        print(f"Error evaluating structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))
