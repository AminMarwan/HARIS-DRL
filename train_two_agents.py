import json
import os

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from callbacks import StageMetricsCallback
from envs import GRASP_STATES_PATH, ReachOnlyEnv, RotateValveEnv, load_saved_states
from evaluate_models import main as run_evaluation
from plotting import generate_all_figures


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

CONFIG = {
    "agent1_timesteps": 1_000_000,
    "agent2_timesteps": 1_000_000,
    "agent1_extra_block": 50_000,
    "max_extra_agent1_timesteps": 500_000,
    "min_grasp_states": 60,
    "learning_rate": 3e-4,
    "buffer_size": 500_000,
    "learning_starts": 10_000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "net_arch": [256, 256],
}


def ensure_dirs() -> None:
    # Create the output folders used by training, plotting, and evaluation.
    for path in [OUTPUT_DIR, MODEL_DIR, LOG_DIR, FIGURE_DIR]:
        os.makedirs(path, exist_ok=True)


def save_config() -> None:
    # Save the current experiment configuration for reproducibility.
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(CONFIG, handle, indent=2)


def build_sac(env, device: str) -> SAC:
    # Build the SAC model with the shared hyperparameters for both agents.
    return SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=CONFIG["learning_rate"],
        buffer_size=CONFIG["buffer_size"],
        learning_starts=CONFIG["learning_starts"],
        batch_size=CONFIG["batch_size"],
        gamma=CONFIG["gamma"],
        tau=CONFIG["tau"],
        ent_coef="auto",
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_kwargs=dict(net_arch=CONFIG["net_arch"]),
    )


def train_agent1(device: str) -> None:
    # Train the reach policy and collect successful grasp states for stage 2.
    print("=" * 70)
    print("AGENT 1: Reach the valve lever")
    print("=" * 70)

    env = Monitor(ReachOnlyEnv(save_path=GRASP_STATES_PATH))
    model = build_sac(env, device=device)
    callback = StageMetricsCallback(
        stage_name="agent1_reach",
        model_dir=MODEL_DIR,
        log_dir=LOG_DIR,
        stop_threshold=0.80,
        patience=2,
        metric_name="success",
    )

    model.learn(total_timesteps=CONFIG["agent1_timesteps"], callback=callback, log_interval=10)
    model.save(os.path.join(MODEL_DIR, "agent1_reach_final"))

    extra_timesteps = 0
    # Agent 2 samples from these saved states, so make sure enough of them exist.
    while len(load_saved_states(GRASP_STATES_PATH)) < CONFIG["min_grasp_states"]:
        if extra_timesteps >= CONFIG["max_extra_agent1_timesteps"]:
            raise RuntimeError(
                f"Agent 1 produced only {len(load_saved_states(GRASP_STATES_PATH))} reach states. "
                "Increase training time or inspect the reach reward setup."
            )
        print("Collecting more reach states before Agent 2 starts...")
        model.learn(
            total_timesteps=CONFIG["agent1_extra_block"],
            callback=callback,
            log_interval=10,
            reset_num_timesteps=False,
        )
        extra_timesteps += CONFIG["agent1_extra_block"]

    env.close()


def train_agent2(device: str) -> None:
    # Train the valve-rotation policy from the saved grasp-state starts.
    print("=" * 70)
    print("AGENT 2: Rotate the valve after reaching it")
    print("=" * 70)

    env = Monitor(RotateValveEnv(grasp_states_path=GRASP_STATES_PATH))
    model = build_sac(env, device=device)
    callback = StageMetricsCallback(
        stage_name="agent2_rotate",
        model_dir=MODEL_DIR,
        log_dir=LOG_DIR,
        stop_threshold=0.75,
        patience=2,
        metric_name="valve_angle",
    )

    model.learn(total_timesteps=CONFIG["agent2_timesteps"], callback=callback, log_interval=10)
    model.save(os.path.join(MODEL_DIR, "agent2_rotate_final"))
    env.close()


def main() -> None:
    # Run the full two-stage training pipeline and generate outputs.
    ensure_dirs()
    save_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Train the reach stage first because it produces the starting states for stage 2.
    train_agent1(device=device)
    generate_all_figures(LOG_DIR, FIGURE_DIR)

    train_agent2(device=device)
    generate_all_figures(LOG_DIR, FIGURE_DIR)
    # Evaluate after both stages complete so the final bundle contains metrics and plots.
    run_evaluation()

    print("\nTraining bundle finished.")
    print(f"Models  : {MODEL_DIR}")
    print(f"Logs    : {LOG_DIR}")
    print(f"Figures : {FIGURE_DIR}")
    print(f"States  : {GRASP_STATES_PATH}")


if __name__ == "__main__":
    main()
