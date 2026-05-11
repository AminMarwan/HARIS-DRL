import json
import os
import csv

import matplotlib.pyplot as plt
import pandas as pd


EPISODE_COLUMNS = [
    "timesteps",
    "episode",
    "episode_reward",
    "episode_length",
    "success",
    "dist",
    "grasp_contacts",
    "gripper_q",
    "valve_angle",
    "best_valve_angle",
    "has_stable_grasp",
]


def _read_episode_csv(csv_path: str) -> pd.DataFrame:
    # Fall back to a tolerant reader for logs whose header changed mid-run.
    try:
        return pd.read_csv(csv_path)
    except pd.errors.ParserError:
        rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            for row in reader:
                item = {}
                for idx, value in enumerate(row):
                    if idx < len(header):
                        column = header[idx]
                    elif idx < len(EPISODE_COLUMNS):
                        column = EPISODE_COLUMNS[idx]
                    else:
                        column = f"extra_{idx}"
                    item[column] = value
                rows.append(item)
        return pd.DataFrame(rows)


def _to_numeric(series: pd.Series) -> pd.Series:
    # Convert a series to numeric values and coerce invalid entries to NaN.
    return pd.to_numeric(series, errors="coerce")


def _rolling(series: pd.Series, window: int = 25) -> pd.Series:
    # Apply a rolling mean to smooth noisy episode-level metrics.
    numeric = _to_numeric(series)
    return numeric.rolling(window=window, min_periods=1).mean()


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize known metric columns so plotting code can assume numeric inputs.
    numeric_cols = [
        "timesteps",
        "episode",
        "episode_reward",
        "episode_length",
        "success",
        "dist",
        "grasp_contacts",
        "gripper_q",
        "valve_angle",
        "best_valve_angle",
        "has_stable_grasp",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timesteps" in df.columns:
        df = df.dropna(subset=["timesteps"])

    return df.reset_index(drop=True)


def plot_stage_curves(csv_path: str, figure_dir: str, stage_name: str) -> None:
    # Generate per-stage training curves from the callback CSV log.
    os.makedirs(figure_dir, exist_ok=True)
    df = _read_episode_csv(csv_path)
    if df.empty:
        return

    df = _clean_df(df)
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    axes[0].plot(df["timesteps"], _rolling(df["episode_reward"]), linewidth=2)
    axes[0].set_title(f"{stage_name} Reward")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Episode reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["timesteps"], _rolling(df["success"]), linewidth=2)
    axes[1].set_title(f"{stage_name} Success Rate")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Success")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["timesteps"], _rolling(df["episode_length"]), linewidth=2)
    axes[2].set_title(f"{stage_name} Episode Length")
    axes[2].set_xlabel("Timesteps")
    axes[2].set_ylabel("Steps")
    axes[2].grid(True, alpha=0.3)

    # Reach training tracks distance, while rotation training also logs valve angle.
    metric_col = "valve_angle" if df["valve_angle"].notna().any() else "dist"
    axes[3].plot(df["timesteps"], _rolling(df[metric_col]), linewidth=2)
    axes[3].set_title(f"{stage_name} {'Valve Angle' if metric_col == 'valve_angle' else 'Reach Distance'}")
    axes[3].set_xlabel("Timesteps")
    axes[3].set_ylabel(metric_col.replace("_", " ").title())
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, f"{stage_name}_training_curves.png"), dpi=200)
    plt.close(fig)

    if "grasp_contacts" in df.columns and df["grasp_contacts"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(df["timesteps"], _rolling(df["grasp_contacts"]), linewidth=2)
        axes[0].set_title(f"{stage_name} Lever Contacts")
        axes[0].set_xlabel("Timesteps")
        axes[0].set_ylabel("Contacts")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df["timesteps"], _rolling(df["gripper_q"]), linewidth=2)
        axes[1].set_title(f"{stage_name} Gripper Joint")
        axes[1].set_xlabel("Timesteps")
        axes[1].set_ylabel("Gripper qpos")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(figure_dir, f"{stage_name}_grasp_metrics.png"), dpi=200)
        plt.close(fig)


def plot_training_summary(log_dir: str, figure_dir: str) -> None:
    # Create a small summary figure across the two training stages.
    os.makedirs(figure_dir, exist_ok=True)
    summaries = []

    for name in ["agent1_reach", "agent2_rotate"]:
        summary_path = os.path.join(log_dir, f"{name}_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                summaries.append(json.load(handle))

    if not summaries:
        return

    labels = [item["stage_name"] for item in summaries]
    success = [item.get("rolling_success_rate", 0.0) for item in summaries]
    steps = [item.get("timesteps", 0) for item in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(labels, success)
    axes[0].set_title("Final Rolling Success Rate")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, steps)
    axes[1].set_title("Training Timesteps")
    axes[1].set_ylabel("Timesteps")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "training_summary.png"), dpi=200)
    plt.close(fig)


def generate_all_figures(log_dir: str, figure_dir: str) -> None:
    # Generate every available training figure from the saved logs.
    stage_files = {
        "agent1_reach": os.path.join(log_dir, "agent1_reach_episodes.csv"),
        "agent2_rotate": os.path.join(log_dir, "agent2_rotate_episodes.csv"),
    }

    for stage_name, csv_path in stage_files.items():
        if os.path.exists(csv_path):
            plot_stage_curves(csv_path, figure_dir, stage_name)

    plot_training_summary(log_dir, figure_dir)
