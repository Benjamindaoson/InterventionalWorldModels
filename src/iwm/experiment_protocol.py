"""Research metadata and scene-level split leakage validation."""
from dataclasses import dataclass,asdict
import hashlib,json
@dataclass(frozen=True)
class Experiment:
    experiment_id:str
    hypothesis:str
    dataset_version:str
    split:str
    model:str
    seed:int
    training_budget_gpu_hours:float
    def manifest(self):
        d=asdict(self)
        d["fingerprint"]=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
        return d
def validate_scene_splits(splits):
    seen={}
    for split,ids in splits.items():
        for scene in ids:
            if scene in seen:raise ValueError(f"scene leakage: {scene} in {seen[scene]} and {split}")
            seen[scene]=split
    return len(seen)
