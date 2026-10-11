"""Optional CPU MuJoCo physics smoke-test generator; not yet executed."""
import json
from pathlib import Path
import numpy as np
def simulate(force,friction,steps=80,dt=.005):
    try:import mujoco
    except ImportError as exc:raise RuntimeError("Install mujoco first") from exc
    xml=f'''<mujoco><option timestep="{dt}" gravity="0 0 -9.81"/><worldbody><geom type="plane" size="2 2 .1" friction="{friction} .005 .0001"/><body name="cube" pos="0 0 .055"><freejoint/><geom type="box" size=".05 .05 .05" mass="1" friction="{friction} .005 .0001"/></body></worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    mujoco.mj_forward(model,data);body=model.body("cube").id
    initial=data.xpos[body].copy();positions=[]
    for _ in range(steps):
        data.xfrc_applied[body,0]=force
        mujoco.mj_step(model,data)
        positions.append(data.xpos[body].copy().tolist())
    displacement=float(np.linalg.norm(np.asarray(positions[-1])[:2]-initial[:2]))
    return dict(force=force,friction=friction,positions=positions,displacement=displacement,moved=displacement>.005)
def generate(output="results/mujoco_physics.jsonl"):
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w") as f:
        for friction in (.3,.5,.7):
            for force in (1.,3.,5.,7.):
                f.write(json.dumps(simulate(force,friction))+"\n")
    return path
if __name__=="__main__":generate()
