"""
Plot training progress from Ray Tune progress.csv logs.

Usage:
    python scripts/plot_training_progress.py --trial_dir <path_to_trial_directory>
    python scripts/plot_training_progress.py --experiment_dir <path_to_experiment_directory>

Examples:
    # Plot a single trial:
    python scripts/plot_training_progress.py --trial_dir "C:/Users/tejas/ray_results/TestSandbox14/CustomPPO_l2rpn_case14_sandbox_train_run_100k_a1ae2_2026-03-11_12-56-33"

    # Plot all trials in an experiment:
    python scripts/plot_training_progress.py --experiment_dir "C:/Users/tejas/ray_results/TestSandbox14"
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_progress(trial_dir: str) -> pd.DataFrame:
    """Load progress.csv from a single trial directory."""
    csv_path = os.path.join(trial_dir, "progress.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No progress.csv found in {trial_dir}")
    df = pd.read_csv(csv_path)
    return df


def plot_single_trial(df: pd.DataFrame, trial_name: str, save_path: str = None):
    """Generate plots for a single trial."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Training Progress: {trial_name}", fontsize=14, fontweight="bold")

    # 1. Episode length (grid2op_end_mean) vs training iteration
    ax = axes[0, 0]
    if "custom_metrics/grid2op_end_mean" in df.columns:
        ax.plot(df["training_iteration"], df["custom_metrics/grid2op_end_mean"],
                marker=".", markersize=3, label="Train ep_len (mean)")
        # Add cumulative mean
        cumul_mean = df["custom_metrics/grid2op_end_mean"].expanding().mean()
        ax.plot(df["training_iteration"], cumul_mean,
                linewidth=2, color="red", label="Cumulative mean")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Grid2Op Episode Length (steps)")
    ax.set_title("Episode Length vs Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Episode length vs timesteps
    ax = axes[0, 1]
    if "custom_metrics/grid2op_end_mean" in df.columns and "timesteps_total" in df.columns:
        ax.plot(df["timesteps_total"], df["custom_metrics/grid2op_end_mean"],
                marker=".", markersize=3, label="Train ep_len (mean)")
        cumul_mean = df["custom_metrics/grid2op_end_mean"].expanding().mean()
        ax.plot(df["timesteps_total"], cumul_mean,
                linewidth=2, color="red", label="Cumulative mean")
    ax.set_xlabel("Total Agent Timesteps")
    ax.set_ylabel("Grid2Op Episode Length (steps)")
    ax.set_title("Episode Length vs Timesteps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Episode reward vs training iteration
    ax = axes[1, 0]
    if "episode_reward_mean" in df.columns:
        ax.plot(df["training_iteration"], df["episode_reward_mean"],
                marker=".", markersize=3, color="green", label="Mean reward")
        cumul_mean_rw = df["episode_reward_mean"].expanding().mean()
        ax.plot(df["training_iteration"], cumul_mean_rw,
                linewidth=2, color="darkred", label="Cumulative mean")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward vs Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Evaluation episode length (if available)
    ax = axes[1, 1]
    eval_cols = [c for c in df.columns if "evaluation" in c and "grid2op_end_mean" in c]
    if eval_cols:
        # These appear only on eval iterations, so drop NaN
        eval_col = eval_cols[0] if eval_cols else None
    else:
        eval_col = None

    if eval_col and eval_col in df.columns:
        eval_df = df.dropna(subset=[eval_col])
        ax.plot(eval_df["training_iteration"], eval_df[eval_col],
                marker="o", markersize=4, color="purple", label="Eval ep_len (mean)")
    if "custom_metrics/grid2op_end_mean" in df.columns:
        ax.plot(df["training_iteration"], df["custom_metrics/grid2op_end_mean"],
                alpha=0.3, color="blue", label="Train ep_len (mean)")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title("Train vs Eval Episode Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training progress from Ray logs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trial_dir", type=str, help="Path to a single trial directory")
    group.add_argument("--experiment_dir", type=str, help="Path to experiment directory (plots all trials)")
    parser.add_argument("--save", action="store_true", help="Save plots as PNG files")
    args = parser.parse_args()

    if args.trial_dir:
        trial_dirs = [args.trial_dir]
    else:
        trial_dirs = [
            os.path.join(args.experiment_dir, d)
            for d in os.listdir(args.experiment_dir)
            if os.path.isdir(os.path.join(args.experiment_dir, d))
            and os.path.exists(os.path.join(args.experiment_dir, d, "progress.csv"))
        ]
        if not trial_dirs:
            print(f"No trial directories with progress.csv found in {args.experiment_dir}")
            return

    for trial_dir in trial_dirs:
        trial_name = os.path.basename(trial_dir)
        print(f"\n--- Processing: {trial_name} ---")
        try:
            df = load_progress(trial_dir)
            save_path = os.path.join(trial_dir, "training_progress.png") if args.save else None
            plot_single_trial(df, trial_name, save_path)
        except Exception as e:
            print(f"Error processing {trial_name}: {e}")


if __name__ == "__main__":
    main()
