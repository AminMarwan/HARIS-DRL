import argparse
import json
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from stable_baselines3 import SAC

from envs import GRASP_STATES_PATH, ReachOnlyEnv, RotateValveEnv


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RECORDING_DIR = os.path.join(OUTPUT_DIR, "recordings")


def resolve_model(*candidates: str) -> str:
    # Return the first available model path from a preferred list of filenames.
    for candidate in candidates:
        path = os.path.join(MODEL_DIR, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find any of these models in {MODEL_DIR}: {candidates}")


def make_frame(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    # Render the current MuJoCo scene into a standalone image frame.
    renderer.update_scene(data)
    return renderer.render().copy()


def run_episode(env, model, renderer, max_steps: int, fps: int, frame_skip: int) -> tuple[list[np.ndarray], dict]:
    # Reset the environment and record one full policy rollout.
    obs, _ = env.reset()
    return run_episode_from_current_state(env, model, renderer, obs, max_steps, fps, frame_skip)


def run_episode_from_current_state(
    env,
    model,
    renderer,
    obs,
    max_steps: int,
    fps: int,
    frame_skip: int,
) -> tuple[list[np.ndarray], dict]:
    # Record an episode starting from the environment's current internal state.
    frames = [make_frame(renderer, env.model, env.data)]
    done = False
    truncated = False
    steps = 0
    total_reward = 0.0
    info = {}

    while not done and not truncated and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        # Capture roughly one frame per control step for readable playback.
        frames.append(make_frame(renderer, env.model, env.data))

    info = dict(info)
    info["episode_reward"] = total_reward
    info["episode_steps"] = steps
    info["video_fps"] = fps
    info["sim_frame_skip"] = frame_skip
    return frames, info


def transfer_state_between_envs(source_env, target_env) -> np.ndarray:
    # Copy the MuJoCo state from Agent 1's environment into Agent 2's environment
    target_env.data.qpos[:] = source_env.data.qpos.copy()
    target_env.data.qvel[:] = source_env.data.qvel.copy()
    target_env.data.ctrl[:] = source_env.data.ctrl.copy()
    mujoco.mj_forward(target_env.model, target_env.data)
    return target_env._get_obs()


def save_video(path: str, frames: list[np.ndarray], fps: int) -> None:
    # Write a list of rendered frames to an MP4 file.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)


def record_stage(stage: str, episodes: int, width: int, height: int, fps: int) -> list[dict]:
    # Record standalone rollouts for one stage only
    if stage == "agent1":
        env = ReachOnlyEnv(save_path=GRASP_STATES_PATH)
        model_path = resolve_model("agent1_reach_best.zip", "agent1_reach_final.zip")
    else:
        env = RotateValveEnv(grasp_states_path=GRASP_STATES_PATH)
        model_path = resolve_model("agent2_rotate_best.zip", "agent2_rotate_final.zip")

    model = SAC.load(model_path)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    results = []

    for ep in range(episodes):
        frames, info = run_episode(
            env=env,
            model=model,
            renderer=renderer,
            max_steps=env.max_steps,
            fps=fps,
            frame_skip=env.frame_skip,
        )
        video_path = os.path.join(RECORDING_DIR, f"{stage}_episode_{ep + 1}.mp4")
        save_video(video_path, frames, fps=fps)
        info["video_path"] = video_path
        info["stage"] = stage
        results.append(info)

    renderer.close()
    env.close()
    return results


def record_combined(episodes: int, width: int, height: int, fps: int) -> list[dict]:
    # Record Agent 1 followed immediately by Agent 2 in a single combined video."""
    agent1_env = ReachOnlyEnv(save_path=GRASP_STATES_PATH)
    agent2_env = RotateValveEnv(grasp_states_path=GRASP_STATES_PATH)

    agent1_model = SAC.load(resolve_model("agent1_reach_best.zip", "agent1_reach_final.zip"))
    agent2_model = SAC.load(resolve_model("agent2_rotate_best.zip", "agent2_rotate_final.zip"))

    renderer1 = mujoco.Renderer(agent1_env.model, height=height, width=width)
    renderer2 = mujoco.Renderer(agent2_env.model, height=height, width=width)
    results = []

    for ep in range(episodes):
        frames_1, info_1 = run_episode(
            env=agent1_env,
            model=agent1_model,
            renderer=renderer1,
            max_steps=agent1_env.max_steps,
            fps=fps,
            frame_skip=agent1_env.frame_skip,
        )

        # Start Agent 2 from the exact physical state where Agent 1 finished.
        agent2_obs = transfer_state_between_envs(agent1_env, agent2_env)
        # Reset stage-specific trackers so Agent 2 computes rewards from its own rollout.
        agent2_env.step_count = 0
        agent2_env.prev_dist = agent2_env._dist_to_grasp()
        agent2_env.prev_valve_angle = agent2_env._valve_angle()

        frames_2, info_2 = run_episode_from_current_state(
            env=agent2_env,
            model=agent2_model,
            renderer=renderer2,
            obs=agent2_obs,
            max_steps=agent2_env.max_steps,
            fps=fps,
            frame_skip=agent2_env.frame_skip,
        )

        combined_frames = frames_1 + frames_2
        video_path = os.path.join(RECORDING_DIR, f"combined_episode_{ep + 1}.mp4")
        save_video(video_path, combined_frames, fps=fps)

        results.append(
            {
                "stage": "combined",
                "episode": ep + 1,
                "video_path": video_path,
                "agent1_success": bool(info_1.get("is_success", False)),
                "agent2_success": bool(info_2.get("is_success", False)),
                "agent1_reward": float(info_1.get("episode_reward", 0.0)),
                "agent2_reward": float(info_2.get("episode_reward", 0.0)),
                "agent2_final_angle": float(info_2.get("valve_angle", np.nan)),
            }
        )

    renderer1.close()
    renderer2.close()
    agent1_env.close()
    agent2_env.close()
    return results


def main() -> None:
    # Parse CLI arguments, record videos, and save a JSON summary.
    parser = argparse.ArgumentParser(description="Record test runs for downloaded SAC models.")
    parser.add_argument("--mode", choices=["agent1", "agent2", "combined"], default="combined")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(RECORDING_DIR, exist_ok=True)

    if args.mode == "combined":
        results = record_combined(args.episodes, args.width, args.height, args.fps)
    else:
        results = record_stage(args.mode, args.episodes, args.width, args.height, args.fps)

    # Save per-episode metadata so recordings can be reviewed without replaying videos.
    summary_path = os.path.join(RECORDING_DIR, f"{args.mode}_recording_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved recordings to: {RECORDING_DIR}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
