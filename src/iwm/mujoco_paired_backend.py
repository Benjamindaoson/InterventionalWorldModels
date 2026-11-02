"""Optional, real-physics MuJoCo adapter for the paired replay contract.

This code is unexecuted until a compatible MuJoCo runtime is available.
Real engine runs must be kept separate from deterministic toy contract tests.
"""
from __future__ import annotations

import math


class MuJoCoCubeBackend:
    def __init__(self, friction: float = 0.5, timestep: float = 0.005):
        if not math.isfinite(friction) or friction <= 0 or timestep <= 0:
            raise ValueError("invalid physical parameters")
        import mujoco
        import numpy as np
        self.mujoco, self.np = mujoco, np
        xml = f'''<mujoco>
<option timestep="{timestep}" gravity="0 0 -9.81"/>
<worldbody>
  <geom type="plane" size="2 2 0.1" friction="{friction} 0.005 0.0001"/>
  <body name="cube" pos="0 0 0.05">
    <freejoint/>
    <geom type="box" size="0.05 0.05 0.05" mass="1" friction="{friction} 0.005 0.0001"/>
  </body>
</worldbody>
</mujoco>'''
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.state_size = mujoco.mj_stateSize(self.model, self.spec)

    def reset(self, seed: int) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.xfrc_applied[:] = 0
        self.mujoco.mj_forward(self.model, self.data)

    def snapshot(self) -> bytes:
        state = self.np.empty(self.state_size, dtype=self.np.float64)
        self.mujoco.mj_getState(self.model, self.data, state, self.spec)
        return state.astype("<f8", copy=False).tobytes()

    def restore(self, state: bytes) -> None:
        saved = self.np.frombuffer(state, dtype="<f8").copy()
        if len(saved) != self.state_size:
            raise ValueError("incorrect MuJoCo state length")
        self.mujoco.mj_setState(self.model, self.data, saved, self.spec)
        self.data.xfrc_applied[:] = 0
        self.mujoco.mj_forward(self.model, self.data)

    def execute(self, action: tuple[float, ...], horizon: int) -> dict:
        if len(action) != 3:
            raise ValueError("MuJoCo cube action must be [Fx,Fy,Fz]")
        start = self.data.xpos[self.body_id].copy()
        for _ in range(horizon):
            self.data.xfrc_applied[self.body_id, :3] = action
            self.mujoco.mj_step(self.model, self.data)
        displacement = float(self.np.linalg.norm(self.data.xpos[self.body_id, :2] - start[:2]))
        return {"displacement": displacement, "event": displacement > 0.005,
                "final_state": self.data.xpos[self.body_id].tolist(), "steps": horizon}
