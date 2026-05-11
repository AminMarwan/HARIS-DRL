import argparse
import copy
import csv
import json
import os
import shutil
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch

import envs
import evaluate_models
import train_two_agents


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_ROOT_DIR = os.path.join(ROOT_DIR, "sweep_outputs")

# Sweep one parameter at a time around the baseline configuration in train_two_agents.py.
DEFAULT_SWEEP_VALUES = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "batch_size": [128, 256, 512],
    "gamma": [0.95, 0.99, 0.995],
    "tau": [0.002, 0.005, 0.01],
}

STAGE_NAMES = ("agent1_reach", "agent2_rotate")


def _rolling(series: pd.Series, window: int = 25) -> pd.Series:
    # Smooth noisy episode metrics so comparison plots are easier to read.
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rolling(window=window, min_periods=1).mean()


def _sanitize_value(value: Any) -> str:
    # Convert parameter values into filename-safe labels.
    return str(value).replace(".", "p").replace("-", "m")


def _run_label(param_name: str, value: Any) -> str:
    # Build a readable name for one parameter-value run.
    return f"{param_name}_{_sanitize_value(value)}"


def _run_dir(param_name: str, value: Any) -> str:
    # Place each run under its parameter group inside sweep_outputs.
    return os.path.join(SWEEP_ROOT_DIR, param_name, _run_label(param_name, value))


def _comparison_dir() -> str:
    # Store cross-run tables and plots in one shared comparison folder.
    return os.path.join(SWEEP_ROOT_DIR, "comparisons")


def _set_run_paths(run_dir: str) -> None:
    # Redirect training, environment, and evaluation outputs into this sweep run.
    output_dir = os.path.join(run_dir, "outputs")
    model_dir = os.path.join(output_dir, "models")
    log_dir = os.path.join(output_dir, "logs")
    figure_dir = os.path.join(output_dir, "figures")
    eval_dir = os.path.join(output_dir, "evaluation")
    grasp_states_path = os.path.join(output_dir, "grasp_states.npy")

    train_two_agents.OUTPUT_DIR = output_dir
    train_two_agents.MODEL_DIR = model_dir
    train_two_agents.LOG_DIR = log_dir
    train_two_agents.FIGURE_DIR = figure_dir
    train_two_agents.GRASP_STATES_PATH = grasp_states_path

    envs.GRASP_STATES_PATH = grasp_states_path

    evaluate_models.OUTPUT_DIR = output_dir
    evaluate_models.MODEL_DIR = model_dir
    evaluate_models.EVAL_DIR = eval_dir
    evaluate_models.GRASP_STATES_PATH = grasp_states_path


def _prepare_run_dir(run_dir: str, overwrite: bool) -> None:
    # Create the run folder, optionally clearing old results first.
    if overwrite and os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)


def _load_json(path: str) -> dict:
    # Read a JSON file used for saved summaries or sweep configuration.
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: dict | list) -> None:
    # Write structured results while creating the parent folder if needed.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_sweep_values(json_path: str | None) -> dict[str, list[Any]]:
    # Use defaults unless the user provides a custom sweep JSON file.
    if json_path is None:
        return copy.deepcopy(DEFAULT_SWEEP_VALUES)

    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Sweep configuration JSON must contain an object mapping parameter names to lists.")

    # Require each parameter to have at least one value to test.
    for key, values in data.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Sweep values for '{key}' must be a non-empty list.")

    return data


def _apply_config_overrides(
    base_config: dict[str, Any],
    param_name: str,
    value: Any,
    timesteps_scale: float,
) -> None:
    # Copy the baseline config, change one parameter, and scale training length.
    config = copy.deepcopy(base_config)
    config[param_name] = value

    for key in ("agent1_timesteps", "agent2_timesteps", "agent1_extra_block", "max_extra_agent1_timesteps"):
        config[key] = max(1, int(config[key] * timesteps_scale))

    train_two_agents.CONFIG.clear()
    train_two_agents.CONFIG.update(config)


def _run_training_bundle(device: str, stage_mode: str) -> None:
    # Run the requested training stages and regenerate plots after each stage.
    train_two_agents.ensure_dirs()
    train_two_agents.save_config()

    train_two_agents.train_agent1(device=device)
    train_two_agents.generate_all_figures(train_two_agents.LOG_DIR, train_two_agents.FIGURE_DIR)

    if stage_mode in {"agent2", "both"}:
        train_two_agents.train_agent2(device=device)
        train_two_agents.generate_all_figures(train_two_agents.LOG_DIR, train_two_agents.FIGURE_DIR)
        evaluate_models.main()
    elif stage_mode == "agent1":
        evaluate_models.main()


def _collect_run_summary(run_dir: str, param_name: str, value: Any, stage_mode: str) -> dict[str, Any]:
    # Gather the final training and evaluation metrics for one sweep run.
    output_dir = os.path.join(run_dir, "outputs")
    log_dir = os.path.join(output_dir, "logs")
    eval_dir = os.path.join(output_dir, "evaluation")

    summary: dict[str, Any] = {
        "run_name": _run_label(param_name, value),
        "parameter": param_name,
        "value": value,
        "run_dir": run_dir,
        "stage_mode": stage_mode,
    }

    # Merge per-stage training summaries into a flat row for CSV output.
    for stage_name in STAGE_NAMES:
        summary_path = os.path.join(log_dir, f"{stage_name}_summary.json")
        if os.path.exists(summary_path):
            stage_summary = _load_json(summary_path)
            for key, stage_value in stage_summary.items():
                summary[f"{stage_name}_{key}"] = stage_value

    # Add evaluation metrics if evaluation produced a summary file.
    eval_summary_path = os.path.join(eval_dir, "evaluation_summary.json")
    if os.path.exists(eval_summary_path):
        eval_summary = _load_json(eval_summary_path)
        for stage_name, metrics in eval_summary.items():
            for key, stage_value in metrics.items():
                summary[f"{stage_name}_eval_{key}"] = stage_value

    return summary


def _write_summary_table(rows: list[dict[str, Any]]) -> None:
    # Write all run summaries into one comparison CSV.
    if not rows:
        return

    output_path = os.path.join(_comparison_dir(), "sweep_summary.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames: list[str] = []
    # Preserve the first-seen column order across rows with different metrics.
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric_comparison(
    run_records: list[dict[str, Any]],
    stage_name: str,
    metric_key: str,
    title: str,
    y_label: str,
    output_path: str,
) -> None:
    # Plot one metric across all tested values for a single parameter.
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for record in run_records:
        csv_path = os.path.join(record["run_dir"], "outputs", "logs", f"{stage_name}_episodes.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if "timesteps" not in df.columns or metric_key not in df.columns:
            continue

        df["timesteps"] = pd.to_numeric(df["timesteps"], errors="coerce")
        df[metric_key] = pd.to_numeric(df[metric_key], errors="coerce")
        df = df.dropna(subset=["timesteps", metric_key])
        if df.empty:
            continue

        # Plot a rolling average so short-term reward noise does not dominate.
        ax.plot(
            df["timesteps"],
            _rolling(df[metric_key]),
            linewidth=2,
            label=str(record["value"]),
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    # Save only figures that had at least one valid run to show.
    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Value")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_parameter_comparisons(param_name: str, run_records: list[dict[str, Any]], stage_mode: str) -> None:
    # Generate the standard comparison figures for one swept parameter.
    comparisons_dir = _comparison_dir()

    _plot_metric_comparison(
        run_records,
        stage_name="agent1_reach",
        metric_key="episode_reward",
        title=f"Agent 1 Reward Sweep: {param_name}",
        y_label="Episode reward",
        output_path=os.path.join(comparisons_dir, f"{param_name}_agent1_reward.png"),
    )
    _plot_metric_comparison(
        run_records,
        stage_name="agent1_reach",
        metric_key="success",
        title=f"Agent 1 Success Sweep: {param_name}",
        y_label="Success",
        output_path=os.path.join(comparisons_dir, f"{param_name}_agent1_success.png"),
    )

    if stage_mode in {"agent2", "both"}:
        _plot_metric_comparison(
            run_records,
            stage_name="agent2_rotate",
            metric_key="episode_reward",
            title=f"Agent 2 Reward Sweep: {param_name}",
            y_label="Episode reward",
            output_path=os.path.join(comparisons_dir, f"{param_name}_agent2_reward.png"),
        )
        _plot_metric_comparison(
            run_records,
            stage_name="agent2_rotate",
            metric_key="valve_angle",
            title=f"Agent 2 Valve Angle Sweep: {param_name}",
            y_label="Valve angle",
            output_path=os.path.join(comparisons_dir, f"{param_name}_agent2_valve_angle.png"),
        )


def run_sweep(
    sweep_values: dict[str, list[Any]],
    stage_mode: str,
    timesteps_scale: float,
    overwrite: bool,
) -> list[dict[str, Any]]:
    # Run each parameter sweep while restoring the original config at the end.
    base_config = copy.deepcopy(train_two_agents.CONFIG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows: list[dict[str, Any]] = []

    print(f"Using device: {device}")
    print(f"Stage mode: {stage_mode}")
    print(f"Timesteps scale: {timesteps_scale}")

    for param_name, values in sweep_values.items():
        # Keep sweeps limited to known training configuration keys.
        if param_name not in base_config:
            raise KeyError(f"Unknown config key '{param_name}'.")

        print("\n" + "=" * 80)
        print(f"SWEEPING PARAMETER: {param_name}")
        print("=" * 80)

        param_rows: list[dict[str, Any]] = []

        for value in values:
            # Train and evaluate one run with this parameter value.
            run_dir = _run_dir(param_name, value)
            print(f"\nRunning {param_name}={value}")
            print(f"Output directory: {run_dir}")

            _prepare_run_dir(run_dir, overwrite=overwrite)
            _set_run_paths(run_dir)
            _apply_config_overrides(base_config, param_name, value, timesteps_scale)
            _run_training_bundle(device=device, stage_mode=stage_mode)

            summary = _collect_run_summary(run_dir, param_name, value, stage_mode)
            _write_json(os.path.join(run_dir, "run_summary.json"), summary)
            param_rows.append(summary)
            all_rows.append(summary)

        # Compare values for this parameter before moving to the next one.
        _plot_parameter_comparisons(param_name, param_rows, stage_mode)

    train_two_agents.CONFIG.clear()
    train_two_agents.CONFIG.update(base_config)
    return all_rows


def parse_args() -> argparse.Namespace:
    # Define the command-line controls for running faster or custom sweeps.
    parser = argparse.ArgumentParser(
        description="Run one-parameter-at-a-time sweeps using train_two_agents.py."
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default=None,
        help="Optional JSON file mapping config keys to the list of values to test.",
    )
    parser.add_argument(
        "--stage-mode",
        choices=["agent1", "agent2", "both"],
        default="both",
        help="Choose whether to train Agent 1 only or the full two-stage pipeline.",
    )
    parser.add_argument(
        "--timesteps-scale",
        type=float,
        default=1.0,
        help="Scale training timesteps for faster experiments, for example 0.2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing run folders before rerunning the sweep.",
    )
    return parser.parse_args()


def main() -> None:
    # Load sweep settings, run all experiments, and write the final summary table.
    args = parse_args()
    sweep_values = _load_sweep_values(args.sweep_config)
    rows = run_sweep(
        sweep_values=sweep_values,
        stage_mode=args.stage_mode,
        timesteps_scale=args.timesteps_scale,
        overwrite=args.overwrite,
    )
    _write_summary_table(rows)

    summary_path = os.path.join(_comparison_dir(), "sweep_summary.csv")
    print("\nSweep finished.")
    print(f"Sweep root : {SWEEP_ROOT_DIR}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
