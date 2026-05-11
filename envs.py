import os
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(ROOT_DIR, "Scene.xml")
GRASP_STATES_PATH = os.path.join(ROOT_DIR, "outputs", "grasp_states.npy")

HOME_ARM_QPOS = np.array([0.0, -0.2, 0.2, 0.0, 0.0], dtype=np.float64)
ARM_CTRL_IDS = [0, 1, 2, 3, 4]
GRIPPER_CTRL_ID = 5
N_ARM = 5

GRIPPER_OPEN_CTRL = -0.17453
GRIPPER_CLOSE_CTRL = 1.74533
RETURN_STEPS = 500
MAX_JOINT_DELTA = 0.02


def ensure_parent_dir(path: str) -> None:
    # Create the parent directory for a file path if it does not exist.
    os.makedirs(os.path.dirname(path), exist_ok=True)


def freeze_arm(data: mujoco.MjData) -> None:
    # Hold the arm joints at their current positions while another control changes.
    for ctrl_id in ARM_CTRL_IDS:
        data.ctrl[ctrl_id] = float(data.qpos[ctrl_id])


def do_gripper_motion(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    target_ctrl: float,
    n_steps: int = 60,
) -> None:
    # Move the gripper gradually to avoid sudden jumps in the simulation.
    start = float(data.ctrl[GRIPPER_CTRL_ID])
    for step in range(1, n_steps + 1):
        blend = step / n_steps
        freeze_arm(data)
        data.ctrl[GRIPPER_CTRL_ID] = start + blend * (target_ctrl - start)
        mujoco.mj_step(model, data)


def smoothstep(x: float) -> float:
    # Smooth interpolation used for the return-home motion.
    return x * x * (3.0 - 2.0 * x)


def do_return_home(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    home_arm_qpos: np.ndarray = HOME_ARM_QPOS,
    gripper_ctrl: float = GRIPPER_CLOSE_CTRL,
) -> None:
    # Return the arm to its home pose with a smooth multi-step transition.
    start_arm_qpos = data.qpos[:N_ARM].copy()
    total_delta = home_arm_qpos - start_arm_qpos
    max_range = np.max(np.abs(total_delta))
    n_steps = max(RETURN_STEPS, int(max_range / MAX_JOINT_DELTA) + 1)

    for step in range(1, n_steps + 1):
        blend = smoothstep(step / n_steps)
        target = start_arm_qpos + blend * total_delta
        for joint_idx in range(N_ARM):
            data.ctrl[joint_idx] = float(target[joint_idx])
        data.ctrl[GRIPPER_CTRL_ID] = gripper_ctrl
        mujoco.mj_step(model, data)


def load_saved_states(path: str):
    # Load saved grasp states from disk, or return an empty list if none exist.
    if not os.path.exists(path):
        return []
    states = np.load(path, allow_pickle=True)
    return list(states)


@dataclass
class ValveModelRefs:
    # MuJoCo index references reused during environment steps.
    valve_qpos_adr: int
    valve_qvel_adr: int
    gripper_qpos_adr: int
    lever_geom_id: int
    gripper_geom_ids: set


class ValveTaskBase(gym.Env):
    # Shared MuJoCo utilities used by both the reach and rotation stages.
    metadata = {"render_modes": []}

    def __init__(self, model_path: str = SCENE_PATH, frame_skip: int = 20):
        super().__init__()
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.n_arm = N_ARM
        self.n_act = self.model.nu
        self.refs = self._build_refs()
        self.max_steps = 250
        self.step_count = 0

    def _build_refs(self) -> ValveModelRefs:
        # Resolve frequently used MuJoCo ids once at initialization time.
        lever_geom_id = self.model.geom("lever_geom").id
        gripper_bodies = {"gripper", "moving_jaw_so101_v1"}
        gripper_geom_ids = set()

        for geom_id in range(self.model.ngeom):
            body_name = self.model.body(self.model.geom_bodyid[geom_id]).name
            if body_name in gripper_bodies:
                if self.model.geom_contype[geom_id] != 0 or self.model.geom_conaffinity[geom_id] != 0:
                    gripper_geom_ids.add(geom_id)

        return ValveModelRefs(
            valve_qpos_adr=self.model.joint("valve_hinge").qposadr[0],
            valve_qvel_adr=self.model.joint("valve_hinge").dofadr[0],
            gripper_qpos_adr=self.model.joint("gripper").qposadr[0],
            lever_geom_id=lever_geom_id,
            gripper_geom_ids=gripper_geom_ids,
        )

    def _lever_contact_count(self) -> int:
        count = 0
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self.refs.lever_geom_id and geom2 in self.refs.gripper_geom_ids:
                count += 1
            elif geom2 == self.refs.lever_geom_id and geom1 in self.refs.gripper_geom_ids:
                count += 1
        return count

    def _gripper_q(self) -> float:
        return float(self.data.qpos[self.refs.gripper_qpos_adr])

    def _valve_angle(self) -> float:
        return float(self.data.qpos[self.refs.valve_qpos_adr])

    def _valve_velocity(self) -> float:
        return float(self.data.qvel[self.refs.valve_qvel_adr])

    def _ee_pos(self) -> np.ndarray:
        return self.data.site("gripperframe").xpos.copy()

    def _grasp_pos(self) -> np.ndarray:
        return self.data.site("lever_grasp").xpos.copy()

    def _dist_to_grasp(self) -> float:
        return float(np.linalg.norm(self._grasp_pos() - self._ee_pos()))

    def _apply_arm_action(self, action: np.ndarray, gripper_ctrl: float) -> None:
        ctrl_range = self.model.actuator_ctrlrange
        for idx in range(N_ARM):
            lo, hi = ctrl_range[idx]
            self.data.ctrl[idx] = lo + (float(action[idx]) + 1.0) * 0.5 * (hi - lo)
        self.data.ctrl[GRIPPER_CTRL_ID] = gripper_ctrl

    def _fallback_reset_pose(self, gripper_ctrl: float) -> None:
        self.data.qpos[:N_ARM] = HOME_ARM_QPOS
        self.data.qpos[GRIPPER_CTRL_ID] = gripper_ctrl
        self.data.qpos[self.refs.valve_qpos_adr] = 0.0
        self.data.qvel[self.refs.valve_qvel_adr] = 0.0
        for idx in range(N_ARM):
            self.data.ctrl[idx] = float(HOME_ARM_QPOS[idx])
        self.data.ctrl[GRIPPER_CTRL_ID] = gripper_ctrl
        mujoco.mj_forward(self.model, self.data)


class ReachOnlyEnv(ValveTaskBase):
    # Stage 1 environment: move the gripper to the valve lever and save successful states.
    def __init__(
        self,
        model_path: str = SCENE_PATH,
        save_path: str = GRASP_STATES_PATH,
        max_saved: int = 1000,
    ):
        self.save_path = save_path
        self.max_saved = max_saved
        self.reach_dist = 0.03
        self.reach_bonus = 30.0
        self.w_dist = 1.4
        self.w_prog = 6.0
        self.w_act = 0.05
        self.time_penalty = 0.002
        self.prev_dist = None
        super().__init__(model_path=model_path, frame_skip=20)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_ARM,), dtype=np.float32)
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[:6].copy()
        qvel = self.data.qvel[:6].copy()
        ee_pos = self._ee_pos()
        grasp_pos = self._grasp_pos()
        rel_vec = grasp_pos - ee_pos
        dist = float(np.linalg.norm(rel_vec))
        return np.concatenate([qpos, qvel, ee_pos, grasp_pos, rel_vec, [dist]]).astype(np.float32)

    def _save_grasp_state(self) -> None:
        # Store successful reach states so Stage 2 can start near the lever.
        ensure_parent_dir(self.save_path)
        states = load_saved_states(self.save_path)
        states.append(
            {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "ctrl": self.data.ctrl.copy(),
            }
        )
        if len(states) > self.max_saved:
            states = states[-self.max_saved:]
        np.save(self.save_path, states)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Start near the home pose with small noise to diversify approach trajectories.
        noise = np.random.uniform(-0.05, 0.05, size=N_ARM)
        self.data.qpos[:N_ARM] = HOME_ARM_QPOS + noise
        self.data.qpos[GRIPPER_CTRL_ID] = GRIPPER_CLOSE_CTRL
        self.data.qpos[self.refs.valve_qpos_adr] = 0.0
        self.data.qvel[self.refs.valve_qvel_adr] = 0.0

        for idx in range(N_ARM):
            self.data.ctrl[idx] = float(self.data.qpos[idx])
        self.data.ctrl[GRIPPER_CTRL_ID] = GRIPPER_CLOSE_CTRL
        mujoco.mj_forward(self.model, self.data)

        # Open the gripper after reset so the reach policy approaches with a ready pose.
        # do_gripper_motion(self.data, self.model, GRIPPER_OPEN_CTRL, n_steps=60)

        self.step_count = 0
        self.prev_dist = self._dist_to_grasp()
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        self._apply_arm_action(action, GRIPPER_OPEN_CTRL)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        dist = self._dist_to_grasp()
        progress = self.prev_dist - dist
        self.prev_dist = dist

        reward = (
            -self.w_dist * dist
            + self.w_prog * progress
            - self.w_act * float(np.sum(np.square(action)))
            - self.time_penalty
        )

        terminated = dist <= self.reach_dist
        if terminated:
            reward += self.reach_bonus

            # Important bridge to Stage 2:
            # close the gripper after a successful reach, then save that state.
            do_gripper_motion(self.data, self.model, GRIPPER_CLOSE_CTRL, n_steps=70)
            mujoco.mj_forward(self.model, self.data)
            self._save_grasp_state()

        info = {
            "dist": dist,
            "grasp_contacts": self._lever_contact_count(),
            "gripper_q": self._gripper_q(),
            "is_success": bool(terminated),
        }

        truncated = self.step_count >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, info


class RotateValveEnv(ValveTaskBase):
    # Stage 2 environment: start from a grasp state and learn to rotate the valve.
    def __init__(
        self,
        model_path: str = SCENE_PATH,
        grasp_states_path: str = GRASP_STATES_PATH,
    ):
        self.grasp_states_path = grasp_states_path

        self.success_angle = 1.40
        self.grasp_dist = 0.06
        self.gripper_closed = 1.0
        self.min_grasp_contacts = 1

        self.w_dist = 2.0
        self.w_rot_delta = 30.0
        self.w_rvel = 2.0
        self.w_grip_keep = 2.0
        self.w_backtrack = 12.0
        self.w_ungrasped_motion = 5.0
        self.w_act = 0.05
        self.time_penalty = 0.05
        self.terminal_bonus = 50.0

        self.prev_valve_angle = 0.0
        self.best_valve_angle = 0.0
        self.prev_dist = None
        super().__init__(model_path=model_path, frame_skip=20)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_ARM,), dtype=np.float32)
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[:6].copy()
        qvel = self.data.qvel[:6].copy()
        ee_pos = self._ee_pos()
        grasp_pos = self._grasp_pos()
        rel_vec = grasp_pos - ee_pos
        dist = float(np.linalg.norm(rel_vec))
        return np.concatenate(
            [
                qpos,
                qvel,
                ee_pos,
                grasp_pos,
                rel_vec,
                [dist, self._valve_angle(), self._valve_velocity(), self._gripper_q()],
            ]
        ).astype(np.float32)

    def _sample_grasp_state(self) -> bool:
        # Sample one previously saved grasp state to initialize the rotation task.
        states = load_saved_states(self.grasp_states_path)
        if not states:
            return False

        state = states[np.random.randint(len(states))]
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        self.data.ctrl[:] = state["ctrl"]

        self.data.qpos[self.refs.valve_qpos_adr] = 0.0
        self.data.qvel[self.refs.valve_qvel_adr] = 0.0
        self.data.ctrl[GRIPPER_CTRL_ID] = GRIPPER_CLOSE_CTRL

        mujoco.mj_forward(self.model, self.data)
        return True

    def _has_stable_grasp(self, dist: float, gripper_q: float, contact_count: int) -> bool:
        # Require proximity, contact, and a physically closed gripper before rewarding rotation.
        return (
            dist < self.grasp_dist
            and gripper_q >= self.gripper_closed
            and contact_count >= self.min_grasp_contacts
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Prefer a saved successful grasp; fall back to a fixed pose if none exist yet.
        if not self._sample_grasp_state():
            self._fallback_reset_pose(gripper_ctrl=GRIPPER_CLOSE_CTRL)

        self.step_count = 0
        self.prev_dist = self._dist_to_grasp()
        self.prev_valve_angle = self._valve_angle()
        self.best_valve_angle = self.prev_valve_angle
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        self._apply_arm_action(action, GRIPPER_CLOSE_CTRL)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        dist = self._dist_to_grasp()
        valve_angle = self._valve_angle()
        valve_vel = self._valve_velocity()
        gripper_q = self._gripper_q()
        contact_count = self._lever_contact_count()
        has_stable_grasp = self._has_stable_grasp(dist, gripper_q, contact_count)

        raw_angle_delta = valve_angle - self.prev_valve_angle
        angle_delta = max(valve_angle - self.best_valve_angle, 0.0)
        backtrack = max(-raw_angle_delta, 0.0)
        self.best_valve_angle = max(self.best_valve_angle, valve_angle)
        self.prev_valve_angle = valve_angle
        self.prev_dist = dist

        act_penalty = self.w_act * float(np.sum(np.square(action)))
        grip_penalty = self.w_grip_keep * max(self.gripper_closed - gripper_q, 0.0)

        reward = -self.time_penalty - act_penalty - grip_penalty
        reward += -self.w_dist * dist
        reward += -self.w_backtrack * backtrack

        # Rotation rewards only count when the policy maintains a real closed grasp.
        if has_stable_grasp:
            reward += self.w_rot_delta * angle_delta
            reward += self.w_rvel * max(valve_vel, 0.0)
        else:
            reward -= self.w_ungrasped_motion * abs(raw_angle_delta)

        terminated = valve_angle >= self.success_angle and has_stable_grasp
        if terminated:
            reward += self.terminal_bonus
            # Move back toward a neat final pose after a successful turn.
            do_return_home(
                self.data,
                self.model,
                home_arm_qpos=HOME_ARM_QPOS,
                gripper_ctrl=GRIPPER_CLOSE_CTRL,
            )
            mujoco.mj_forward(self.model, self.data)

        truncated = self.step_count >= self.max_steps
        info = {
            "is_success": bool(terminated),
            "dist": dist,
            "valve_angle": valve_angle,
            "best_valve_angle": self.best_valve_angle,
            "valve_velocity": valve_vel,
            "grasp_contacts": contact_count,
            "has_stable_grasp": bool(has_stable_grasp),
            "gripper_q": gripper_q,
        }
        return self._get_obs(), float(reward), terminated, truncated, info
