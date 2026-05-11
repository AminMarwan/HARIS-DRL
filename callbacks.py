import csv
import json
import os
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


BASE_METRIC_COLUMNS = [
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


class StageMetricsCallback(BaseCallback):
    # Log per-episode metrics, save summaries, and stop early when performance is stable.
    def __init__(
        self,
        stage_name: str,
        model_dir: str,
        log_dir: str,
        save_freq: int = 25_000,
        check_freq: int = 10_000,
        stop_threshold: float = 0.90,
        patience: int = 2,
        min_episodes_for_stop: int = 30,
        metric_name: str = "success_rate",
    ):
        super().__init__()
        self.stage_name = stage_name
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.check_freq = check_freq
        self.stop_threshold = stop_threshold
        self.patience = patience
        self.min_episodes_for_stop = min_episodes_for_stop
        self.metric_name = metric_name

        self.csv_path = os.path.join(log_dir, f"{stage_name}_episodes.csv")
        self.summary_path = os.path.join(log_dir, f"{stage_name}_summary.json")
        self.success_window = deque(maxlen=100)
        self.metric_window = deque(maxlen=100)
        self.episode_count = 0
        self._checks_above = 0
        self._header_written = False
        self.early_stopped = False

    def _on_training_start(self) -> None:
        # Create output folders before training begins.
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self._ensure_csv_header()

    def _fieldnames(self) -> list[str]:
        # Keep one stable CSV schema even when the selected early-stop metric changes.
        fieldnames = list(BASE_METRIC_COLUMNS)
        if self.metric_name not in fieldnames:
            fieldnames.append(self.metric_name)
        return fieldnames

    def _ensure_csv_header(self) -> None:
        # Rewrite older logs if new metric columns were added after training started.
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            return

        fieldnames = self._fieldnames()
        with open(self.csv_path, "r", newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))

        if not rows or rows[0] == fieldnames:
            return

        old_header = rows[0]
        migrated_rows = []
        for row in rows[1:]:
            item = {name: "" for name in fieldnames}
            for idx, value in enumerate(row):
                if idx < len(old_header) and old_header[idx] in item:
                    item[old_header[idx]] = value
                elif idx < len(fieldnames):
                    item[fieldnames[idx]] = value
            migrated_rows.append(item)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(migrated_rows)

    def _append_row(self, row: dict) -> None:
        # Append one finished episode to the CSV log.
        write_header = (not os.path.exists(self.csv_path)) or os.path.getsize(self.csv_path) == 0
        with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames(), extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)


    def _save_summary(self, extra: dict | None = None) -> None:
        # Write the latest rolling metrics to a compact JSON summary.
        summary = {
            "stage_name": self.stage_name,
            "timesteps": int(self.num_timesteps),
            "episodes": int(self.episode_count),
            "rolling_success_rate": float(np.mean(self.success_window)) if self.success_window else 0.0,
            f"rolling_{self.metric_name}": float(np.mean(self.metric_window)) if self.metric_window else 0.0,
        }
        if extra:
            summary.update(extra)
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue

            # Stable-Baselines adds the "episode" entry when a monitored episode ends.
            self.episode_count += 1
            success = float(info.get("is_success", 0.0))
            metric_value = float(info.get(self.metric_name, success))
            row = {
                "timesteps": int(self.num_timesteps),
                "episode": int(self.episode_count),
                "episode_reward": float(info["episode"]["r"]),
                "episode_length": float(info["episode"]["l"]),
                "success": success,
                "dist": float(info.get("dist", np.nan)),
                "grasp_contacts": float(info.get("grasp_contacts", np.nan)),
                "gripper_q": float(info.get("gripper_q", np.nan)),
                "valve_angle": float(info.get("valve_angle", np.nan)),
                "best_valve_angle": float(info.get("best_valve_angle", np.nan)),
                "has_stable_grasp": float(info.get("has_stable_grasp", np.nan)),
                self.metric_name: metric_value,
            }
            self._append_row(row)
            self.success_window.append(success)
            self.metric_window.append(metric_value)

        if self.num_timesteps % self.check_freq == 0 and self.success_window:
            # Evaluate stopping conditions using recent performance rather than one episode.
            success_rate = float(np.mean(self.success_window))
            metric_rate = float(np.mean(self.metric_window)) if self.metric_window else 0.0

            print(f"\n[{self.stage_name} | step {self.num_timesteps}]")
            print(f"  episodes         : {self.episode_count}")
            print(f"  success rate     : {success_rate:.3f}")
            print(f"  {self.metric_name:16}: {metric_rate:.3f}")

            if len(self.success_window) >= self.min_episodes_for_stop and success_rate >= self.stop_threshold:
                self._checks_above += 1
                print(f"  early stop check : {self._checks_above}/{self.patience}")
                if self._checks_above >= self.patience:
                    best_path = os.path.join(self.model_dir, f"{self.stage_name}_best")
                    self.model.save(best_path)
                    self.early_stopped = True
                    self._save_summary({"best_model_path": best_path, "early_stopped": True})
                    return False
            else:
                self._checks_above = 0

            self._save_summary({"early_stopped": False})

        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            # Keep periodic checkpoints in case training is interrupted or later inspection is needed.
            ckpt_path = os.path.join(self.model_dir, f"{self.stage_name}_ckpt_{self.num_timesteps}")
            self.model.save(ckpt_path)
            print(f"  checkpoint saved : {ckpt_path}")

        return True

    def _on_training_end(self) -> None:
        # Persist the final summary, whether training finished normally or early.
        self._save_summary({"early_stopped": self.early_stopped})
