"""Optional, real-physics MuJoCo adapter for the paired replay contract.

This code is unexecuted until a compatible MuJoCo runtime is available.
Real engine runs must be kept separate from deterministic toy contract tests.
"""
from __future__ import annotations

import math


class MuJoCoCubeBackend:
    def __init__(self, friction: float = 0.5, timestep: float = 0.005,
                 mass: float = 1.0, box_half_size: float = 0.05):
        if (not math.isfinite(friction) or friction <= 0 or timestep <= 0 or
                not math.isfinite(mass) or mass <= 0 or
                not math.isfinite(box_half_size) or box_half_size <= 0):
            raise ValueError("invalid physical parameters")
        import mujoco
        import numpy as np
        self.mujoco, self.np = mujoco, np
        xml = f'''<mujoco>
<option timestep="{timestep}" gravity="0 0 -9.81"/>
<worldbody>
  <geom type="plane" size="2 2 0.1" friction="{friction} 0.005 0.0001"/>
  <camera name="track" pos="0 -1.0 0.8" xyaxes="1 0 0 0 0.65 0.76" fovy="45"/>
  <body name="cube" pos="0 0 0.05">
    <freejoint/>
    <geom type="box" size="{box_half_size} {box_half_size} {box_half_size}" mass="{mass}" friction="{friction} 0.005 0.0001"/>
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
        self.data.qpos[0] = (seed % 13) * 0.015
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

    def trajectory(self, action: tuple[float, ...], horizon: int, renderer=None,
                   camera: str | None = None) -> dict:
        """Record actual MuJoCo state at every step after an exact restore."""
        if len(action) != 3 or horizon < 1:
            raise ValueError("expected 3D force and positive horizon")
        positions, velocities, contacts, frames = [], [], [], []
        if renderer is not None:
            if camera is None:
                renderer.update_scene(self.data)
            else:
                renderer.update_scene(self.data, camera=camera)
            frames.append(renderer.render().copy())
        for _ in range(horizon):
            self.data.xfrc_applied[self.body_id, :3] = action
            self.mujoco.mj_step(self.model, self.data)
            positions.append(self.data.xpos[self.body_id].copy().tolist())
            velocities.append(self.data.cvel[self.body_id, 3:].copy().tolist())
            contacts.append(int(self.data.ncon > 0))
            if renderer is not None:
                if camera is None:
                    renderer.update_scene(self.data)
                else:
                    renderer.update_scene(self.data, camera=camera)
                frames.append(renderer.render().copy())
        return {"position": positions, "velocity": velocities, "contact": contacts,
                "rgb": frames}
