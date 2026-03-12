# RL4PNC — Setup & Usage Guide

Complete guide to set up, train, evaluate, and visualize reinforcement learning agents for power grid topology control.

---

## 1. Clone the Repository

```bash
git clone https://github.com/tejaswikandula22/PPO_rl4pnc.git
cd rl4pnc
```

---

## 2. Create Conda Environment

Requires **Python 3.10**.

```bash
conda create -n rl4pnc python=3.10 -y
conda activate rl4pnc
```

---

## 3. Install Dependencies

### Option A: Using requirements.txt (recommended)

```bash
pip install -r requirements.txt
```

### Option B: Using pip editable install

```bash
pip install -e .
```

### LightSim2Grid (if installation fails)

If `lightsim2grid` fails to install via pip, build from source:

```bash
git clone https://github.com/BDonnot/lightsim2grid.git
cd lightsim2grid
git checkout v0.9.2
git submodule init
git submodule update
make
pip install -U pybind11
pip install -U .
cd ..
```

### GPU Support (optional)

If you have an NVIDIA GPU, install PyTorch with CUDA **before** running `pip install -r requirements.txt`:

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
```

---

## 4. Configure the Training

Edit the config file at `configs/l2rpn_case14_sandbox/ppo_baseline.yaml`.

**Required change** — set `lib_dir` to your project path:

```yaml
environment:
  env_config:
    lib_dir: /path/to/your/rl4pnc/   # <-- change this
```

Key parameters you may want to adjust:

| Parameter | Location | Description |
|---|---|---|
| `nb_timesteps` | `setup` | Total training steps (e.g., 50000, 500000) |
| `folder_name` | `setup` | Name for the experiment (used in log directory) |
| `action_space` | `environment.env_config` | Action space: `tennet`, `medha`, `medha_reversed`, `assym`, `d3qn2022` |
| `reward_class` | `environment.env_config.grid2op_kwargs` | Reward function: `!AlphaZeroRW`, `!LossReward`, `!RewardRho` |
| `num_rollout_workers` | `rollouts` | Parallel data collection workers (set based on CPU cores) |
| `evaluation_num_workers` | `evaluation` | Parallel evaluation workers |

---

## 5. Train an Agent

```bash
conda activate rl4pnc

python scripts/train_ppo_baseline.py \
    -f configs/l2rpn_case14_sandbox/ppo_baseline.yaml \
    -wd /path/to/your/rl4pnc/ \
    -s 0 \
    -j my_first_run
```

| Flag | Description |
|---|---|
| `-f` | Path to YAML config file |
| `-wd` | Working directory (project root — used to locate action spaces, scaling arrays) |
| `-s` | Seed offset (added to the seed in config) |
| `-j` | Job ID — a unique name appended to the trial folder |
| `-o` | *(optional)* Train with an opponent enabled |

**Training logs** are saved to: `~/ray_results/<folder_name>/`

The terminal will print live progress each iteration showing episode length and reward.

---

## 6. Plot Training Progress

```bash
# Show plots (no save)
python scripts/plot_training_progress.py \
    --experiment_dir ~/ray_results/<folder_name>

# Save plots as PNG
python scripts/plot_training_progress.py \
    --experiment_dir ~/ray_results/<folder_name> \
    --save
```

Generates 4 plots: episode length vs iteration, episode length vs timesteps, reward vs iteration, and train vs eval comparison.

To plot a specific trial only:

```bash
python scripts/plot_training_progress.py \
    --trial_dir ~/ray_results/<folder_name>/<trial_folder>
```

---

## 7. Evaluate a Trained Agent

First, find your trial folder name:

```bash
# Linux/Mac
ls ~/ray_results/<folder_name>/

# Windows
dir %USERPROFILE%\ray_results\<folder_name>\
```

The trial folder looks like: `CustomPPO_<env>_<job_id>_<hash>_<date>_<time>`

### Evaluate RL agent (best checkpoint)

```bash
python scripts/agent_evaluation.py \
    -a rl \
    -c test \
    -p ~/ray_results/<folder_name> \
    -l /path/to/your/rl4pnc/ \
    -t "<trial_folder_name>" \
    -j eval_best \
    -at 0.95 \
    -lr \
    -b
```

### Evaluate RL agent (last checkpoint)

```bash
python scripts/agent_evaluation.py \
    -a rl \
    -c test \
    -p ~/ray_results/<folder_name> \
    -l /path/to/your/rl4pnc/ \
    -t "<trial_folder_name>" \
    -j eval_last \
    -at 0.95 \
    -lr
```

### Evaluate heuristic baseline (DoNothing / RhoGreedy)

```bash
python scripts/agent_evaluation.py \
    -a heur \
    -c test \
    -p ~/ray_results/<folder_name> \
    -l /path/to/your/rl4pnc/ \
    -j heur_baseline \
    -at 0.95 \
    -lr
```

| Flag | Description |
|---|---|
| `-a` | Agent type: `rl` (trained model) or `heur` (heuristic baseline) |
| `-c` | Chronics split: `test`, `train`, `val`, or `""` (all) |
| `-p` | Path to results directory |
| `-l` | Library directory (project root) |
| `-t` | Trial folder name (for RL agents only) |
| `-b` | Use best checkpoint (highest validation score during training) |
| `-j` | Job ID for this evaluation run |
| `-at` | Rho activation threshold (agent only acts when max rho > this value) |
| `-lr` | Enable line reconnection heuristic |
| `-ld` | Enable line disconnection heuristic |
| `-rt` | Reset topology threshold (0 = disabled) |
| `-s` | Simulate action before executing |

---

## 8. Evaluate All Agents in a Directory

```bash
python scripts/multiple_agent_analysis.py \
    -p ~/ray_results/<folder_name> \
    -l /path/to/your/rl4pnc/ \
    -c test \
    -j eval_all \
    -at 0.95 \
    -lr \
    -w 4
```

| Flag | Description |
|---|---|
| `-w` | Number of parallel workers (default: 16, reduce for local machines) |
| `-q` | Generate quick overview plots per agent |
| `-f` | Filter agents by name substring |

---

## 9. Summarize Evaluation Results

```bash
python scripts/summarize_evaluation_data.py \
    -p ~/ray_results/<folder_name> \
    -c 14 \
    -m 8064
```

Outputs: `summarized_data.csv`, `boxplot_data.csv`, and `box_plots_agents_rules_opponent.svg`.

---

## 10. TensorBoard

```bash
tensorboard --logdir ~/ray_results/<folder_name>
```

Open `http://localhost:6006` in your browser.

---

## Additional Commands

### Generate Scaling Arrays

Run a DoNothing agent across all chronics to collect observation statistics for input normalization:

```bash
python scripts/generate_scaling_array.py -d data/scaling_arrays -e 14 -a -t
```

| Flag | Description |
|---|---|
| `-d` | Output directory for .npy files |
| `-e` | Environment: `14` (case14), `5` (case5), or `14_rel` (case14 realistic) |
| `-a` | Collect from all episodes |
| `-t` | Use test environments |

### Generate Action Spaces

```bash
# Generate tennet action space
python scripts/develop_action_spaces.py -e l2rpn_case14_sandbox -a tennet -s data/action_spaces/

# Generate medha with rho filtering and greedy selection
python scripts/develop_action_spaces.py -e l2rpn_case14_sandbox -a medha -s data/action_spaces/ \
    -dn -sh "" -rf 1.0 -w 4 -g -ps 100 -i 200
```

| Flag | Description |
|---|---|
| `-e` | Environment name |
| `-a` | Strategy: `tennet`, `medha`, `assym`, `d3qn2022`, `alphazero`, `curriculumagent` |
| `-s` | Save path for action space JSON |
| `-rf` | Rho filter threshold (remove actions causing rho > this) |
| `-g` | Apply greedy filtering |
| `-w` | Number of workers |

### Generate Train/Val/Test Split

```bash
python scripts/generate_train_val_test_split.py \
    -e l2rpn_case14_sandbox \
    -p ~/data_grid2op/ \
    -t 10 \
    -v 10
```

Splits chronics into 80% train, 10% validation, 10% test.

### Split Chronics into Per-Day Scenarios

```bash
python scripts/generate_per_day_scenarios.py \
    -e l2rpn_case14_sandbox_train \
    -p ~/data_grid2op/ \
    -d 2
```

---

## Project Structure

```
rl4pnc/
├── configs/                    # YAML configuration files
│   └── l2rpn_case14_sandbox/
│       ├── ppo_baseline.yaml           # Local training config
│       └── ppo_baseline_batchjob.yaml  # Production/HPC config
├── data/
│   ├── action_spaces/          # Pre-built action space JSONs
│   ├── observations_dn/        # Do-nothing observation baselines
│   └── scaling_arrays/         # Min-max scaling arrays for normalization
├── scripts/
│   ├── train_ppo_baseline.py           # Train PPO agent
│   ├── agent_evaluation.py             # Evaluate single agent
│   ├── multiple_agent_analysis.py      # Batch evaluate all agents
│   ├── summarize_evaluation_data.py    # Aggregate evaluation results
│   ├── plot_training_progress.py       # Visualize training curves
│   ├── develop_action_spaces.py        # Create action spaces
│   ├── generate_scaling_array.py       # Generate normalization arrays
│   ├── generate_train_val_test_split.py
│   └── generate_per_day_scenarios.py
├── src/rl4pnc/
│   ├── algorithms/             # CustomPPO (multi-agent batch counting)
│   ├── evaluation/             # Evaluation agent wrappers
│   ├── experiments/            # Callbacks, rewards, YAML parsing, utils
│   ├── grid2op_env/            # Environment wrapper, observation converter
│   └── multi_agent/            # SelectAgentPolicy, DoNothingPolicy
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quick Reference

```bash
# 1. Setup
git clone https://github.com/EricavanderSar/rl4pnc-survey.git && cd rl4pnc
conda create -n rl4pnc python=3.10 -y && conda activate rl4pnc
pip install -r requirements.txt

# 2. Train (update -wd and lib_dir in config to your path)
python scripts/train_ppo_baseline.py -f configs/l2rpn_case14_sandbox/ppo_baseline.yaml -wd /path/to/rl4pnc/ -s 0 -j run_1

# 3. Plot
python scripts/plot_training_progress.py --experiment_dir ~/ray_results/<folder_name>

# 4. Evaluate (replace <TRIAL_FOLDER> with actual folder from step 2)
python scripts/agent_evaluation.py -a rl -c test -p ~/ray_results/<folder_name> -l /path/to/rl4pnc/ -t "<TRIAL_FOLDER>" -j eval -at 0.95 -lr -b

# 5. Summarize
python scripts/summarize_evaluation_data.py -p ~/ray_results/<folder_name> -c 14 -m 8064
```
