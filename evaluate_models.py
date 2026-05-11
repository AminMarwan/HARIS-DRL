import json
import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from envs import GRASP_STATES_PATH, ReachOnlyEnv, RotateValveEnv


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")


def run_policy(model_path: str, env, episodes: int = 20) -> dict:
    # Run a trained policy for a fixed number of episodes and aggregate metrics.
    model = SAC.load(model_path)

    rewards = []
    successes = []
    final_dist = []
    final_angle = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        info = {}

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        successes.append(float(info.get("is_success", 0.0)))
        final_dist.append(float(info.get("dist", np.nan)))
        final_angle.append(float(info.get("valve_angle", np.nan)))

    return {
        "episodes": episodes,
        "reward_mean": float(np.nanmean(rewards)),
        "reward_std": float(np.nanstd(rewards)),
        "success_rate": float(np.nanmean(successes)),
        "final_dist_mean": float(np.nanmean(final_dist)),
        "final_angle_mean": float(np.nanmean(final_angle)),
    }


def save_eval_plot(results: dict, output_path: str) -> None:
    # Create a compact bar chart summary for evaluation results.
    labels = list(results.keys())
    success = [results[label]["success_rate"] for label in labels]
    rewards = [results[label]["reward_mean"] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(labels, success)
    axes[0].set_title("Evaluation Success Rate")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, rewards)
    axes[1].set_title("Evaluation Mean Reward")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    # Evaluate the latest available saved models for both training stages.
    os.makedirs(EVAL_DIR, exist_ok=True)

    results = {}

    # Prefer the best checkpoint when it exists, otherwise fall back to the final model.
    agent1_model = os.path.join(MODEL_DIR, "agent1_reach_best.zip")
    if not os.path.exists(agent1_model):
        agent1_model = os.path.join(MODEL_DIR, "agent1_reach_final.zip")

    if os.path.exists(agent1_model):
        env = ReachOnlyEnv(save_path=GRASP_STATES_PATH)
        results["agent1_reach"] = run_policy(agent1_model, env)
        env.close()

    # Agent 2 is evaluated separately because it uses a different environment setup.
    agent2_model = os.path.join(MODEL_DIR, "agent2_rotate_best.zip")
    if not os.path.exists(agent2_model):
        agent2_model = os.path.join(MODEL_DIR, "agent2_rotate_final.zip")

    if os.path.exists(agent2_model):
        env = RotateValveEnv(grasp_states_path=GRASP_STATES_PATH)
        results["agent2_rotate"] = run_policy(agent2_model, env)
        env.close()

    with open(os.path.join(EVAL_DIR, "evaluation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    # Only create the figure when at least one model was found and evaluated.
    if results:
        save_eval_plot(results, os.path.join(EVAL_DIR, "evaluation_summary.png"))


if __name__ == "__main__":
    main()
