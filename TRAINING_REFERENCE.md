# RL4PNC Training Reference

> **Environment:** l2rpn_case14_sandbox | **Hardware:** AMD Ryzen AI 9 HX (12C/24T)  
> **Workspace:** `C:\Users\tejas\OneDrive\Documents\Major Project\rl4pnc\`  
> **Ray Results:** `C:\Users\tejas\ray_results\`

---

## 1. Config Comparison: Original vs Current vs Production

### Setup

| Parameter | Original | Current (ppo_baseline.yaml) | Production (batchjob.yaml) |
|---|---|---|---|
| `nb_timesteps` | 100,000 | **50,000** | 500,000 |
| `folder_name` | `"TestSandbox14"` | **`"l2rpn_case14_50k"`** | `"Case14_SurveyPaperRainbow"` |
| `experiment_name` | `"TestSandbox14"` | **`"l2rpn_case14_50k"`** | `"Case14_SurveyPaperRainbow"` |
| `num_samples` | 1 | 1 | 32 |
| `seed` | 9 | 9 | 14 |

### Training Hyperparameters

| Parameter | Original | Current | Production |
|---|---|---|---|
| `gamma` | 0.99 | 0.99 | 0.99 |
| `lr` | 0.0001 | 0.0001 | 0.0001 |
| `entropy_coeff` | 0.005 | **0.01** | 0.01 |
| `kl_coeff` | 0.0 | **0.2** | 0.2 |
| `clip_param` | 0.3 | 0.3 | 0.3 |
| `lambda` | 0.95 | 0.95 | 0.95 |
| `vf_loss_coeff` | 0.5 | 0.5 | 0.5 |
| `vf_clip_param` | 100 | 100 | 100 |
| `num_sgd_iter` | 6 | **10** | 15 |
| `sgd_minibatch_size` | 64 | **128** | 256 |
| `train_batch_size` | 256 | **512** | 1024 |
| `batch_mode` | complete_episodes | complete_episodes | complete_episodes |
| `fcnet_hiddens` | [256,256,256] | [256,256,256] | [256,256,256] |
| `fcnet_activation` | relu | relu | relu |
| `post_fcnet_hiddens` | [256] | **removed** | *(not set)* |

### Environment Config

| Parameter | Original | Current | Production |
|---|---|---|---|
| `action_space` | medha_reversed | **tennet** | tennet |
| `reward_class` | `!LossReward` | **`!AlphaZeroRW`** | `!AlphaZeroRW` |
| `rho_threshold` | 0.99 | **0.95** | 0.95 |
| `g2op_input` | `['r']` | **`['v_l', 'a', 'r', 't']`** | `['v_l', 'a', 'r', 't']` |
| `custom_input` | `["d"]` | **`[""]`** | `[""]` |
| `danger` | 0.9 | 0.9 | 0.9 |
| `use_ffw` | True | **False** | False |
| `reset_topo` | 300 | **0** | 0 |
| `line_reco` | True | True | True |
| `line_disc` | True | **False** | False |
| `n_history` | 1 | 1 | 1 |
| `penalty_game_over` | 0 | 0 | 0 |
| `reward_finish` | 0 | 0 | 0 |

### Observation Space Breakdown (current & production)

`g2op_input: ['v_l', 'a', 'r', 't']` expands to:

| Attribute | Size | Scaling File | Status |
|---|---|---|---|
| `v_ex` | 20 | `v_ex.npy` (2, 20) | Normalized |
| `v_or` | 20 | `v_or.npy` (2, 20) | Normalized |
| `a_ex` | 20 | `a_ex.npy` (2, 20) | Normalized |
| `a_or` | 20 | `a_or.npy` (2, 20) | Normalized |
| `rho` | 20 | *(no file — skipped)* | Raw (~0–1.5) |
| `topo_vect` | 57 | *(no file — skipped)* | Raw (1 or 2) |

Total grid2op obs = **157 values**. No custom_input ("d" danger disabled).

### Action Space

| Space | # Actions | Substations |
|---|---|---|
| medha_reversed (original) | 65 + 1 = 66 | Subs 1, 3, 4, 5, 8 |
| **tennet (current & production)** | **73 + 1 = 74** | Subs 1, 3, 4, 5, 8 |

### Evaluation

| Parameter | Original | Current | Production |
|---|---|---|---|
| `evaluation_interval` | 5 | 5 | 10 |
| `evaluation_duration` | 100 episodes | 100 episodes | 100 episodes |
| `evaluation_num_workers` | 2 | **6** | 15 |
| `eval explore` | False | False | True |

### Resources / Rollouts

| Parameter | Original | Current | Production |
|---|---|---|---|
| `num_rollout_workers` | 2 | **10** | 16 |
| `num_learner_workers` | 0 | 0 | 4 |
| `num_workers (scaling)` | — | — | 4 |

---

## 2. Commands Reference

### Training

```powershell
# Activate environment first
conda activate rl4pnc

# Train with current config (50k timesteps)
python scripts/train_ppo_baseline.py -f configs/l2rpn_case14_sandbox/ppo_baseline.yaml -wd "C:\Users\tejas\OneDrive\Documents\Major Project\rl4pnc" -s 0 -j run_50k

# Train with opponent
python scripts/train_ppo_baseline.py -f configs/l2rpn_case14_sandbox/ppo_baseline.yaml -wd "C:\Users\tejas\OneDrive\Documents\Major Project\rl4pnc" -s 0 -j run_50k -o
```

| Flag | Description |
|---|---|
| `-f` | Path to YAML config file |
| `-wd` | Working directory (lib_dir for action spaces, scaling arrays) |
| `-s` | Seed offset (added to config seed) |
| `-j` | Job ID — appended to trial name |
| `-o` | Enable opponent during training |

Logs saved to: `C:\Users\tejas\ray_results\l2rpn_case14_50k\`

---

### Plot Training Progress

```powershell
# Show plots for all trials in an experiment (no save)
python scripts/plot_training_progress.py --experiment_dir "C:\Users\tejas\ray_results\l2rpn_case14_50k"

# Show plot for a single trial
python scripts/plot_training_progress.py --trial_dir "C:\Users\tejas\ray_results\l2rpn_case14_50k\CustomPPO_..."

# Save plots as PNG
python scripts/plot_training_progress.py --experiment_dir "C:\Users\tejas\ray_results\l2rpn_case14_50k" --save
```

Generates 4 subplots: episode length vs iteration, episode length vs timesteps, reward vs iteration, train vs eval overlay.

---

### Evaluate a Single Agent

```powershell
# Evaluate trained RL agent on test chronics
python scripts/agent_evaluation.py -a rl -c test -p "C:\Users\tejas\ray_results\l2rpn_case14_50k" -l "C:\Users\tejas\OneDrive\Documents\Major Project\rl4pnc" -t "CustomPPO_<trial_folder_name>" -j eval_test -at 0.95 -lr

# Evaluate with best checkpoint instead of latest
python scripts/agent_evaluation.py -a rl -c test -p "C:\Users\tejas\ray_results\l2rpn_case14_50k" -l "C:\Users\tejas\OneDrive\Documents\Major Project\rl4pnc" -t "CustomPPO_<trial_folder_name>" -j eval_best -at 0.95 -lr -b

# Evaluate heuristic (DoNothing / RhoGreedy) baseline
python scripts/agent_evaluation.py -a heur -c test -p "C:\Users\tejas\ray_results\l2rpn_case14_50k" -l "C:\Users\tejas\OneDrive\Documents\Major Project\rl4pnc" -j heur_baseline -at 0.95 -lr
```

| Flag | Description |
|---|---|
| `-a` | Agent type: `rl` or `heur` |
| `-c` | Chronics split: `test`, `train`, `val`, or `""` (all) |
| `-p` | Path to results / agent directory |
| `-l` | Library directory (project root) |
| `-t` | Trial folder name (for RL agents) |
| `-b` | Use best checkpoint (default: latest) |
| `-j` | Job ID for this evaluation run |
| `-at` | Rho activation threshold (default: 0.95) |
| `-lr` | Enable line reconnection heuristic |
| `-ld` | Enable line disconnection heuristic |
| `-rt` | Reset topology threshold (0 = disabled) |
| `-s` | Simulate action before executing |

---

### Evaluate All Agents in a Directory

```powershell
# Evaluate all RL agents in parallel (4 workers for local machine)
python scripts/multiple_agent_analysis.py -p "C:\Users\tejas\ray_results\l2rpn_case14_50k" -l "C:\Users\tejas\OneDrive\Documents\Major Project\rl4pnc" -c test -j eval_all -at 0.95 -lr -w 4

# Filter agents by name
python scripts/multiple_agent_analysis.py -p "C:\Users\tejas\ray_results\l2rpn_case14_50k" -l "C:\Users\tejas\OneDrive\Documents\Major Project\rl4pnc" -c test -j eval_filtered -at 0.95 -lr -w 4 -f "run_50k"
```

| Flag | Description |
|---|---|
| `-w` | Number of parallel workers (default: 16, use 4 locally) |
| `-q` | Generate quick overview plots per agent |
| `-f` | Filter agents by name substring |
| *(others same as single evaluation)* | |

---

### Summarize Evaluation Results

```powershell
python scripts/summarize_evaluation_data.py -p "C:\Users\tejas\ray_results\l2rpn_case14_50k" -c 14 -m 8064
```

| Flag | Description |
|---|---|
| `-p` | Path with evaluation results |
| `-c` | Case size (14 for case14) |
| `-m` | Max environment steps per episode (8064) |

Outputs: `summarized_data.csv`, `boxplot_data.csv`, `box_plots_agents_rules_opponent.svg`

---

### TensorBoard

```powershell
tensorboard --logdir "C:\Users\tejas\ray_results\l2rpn_case14_50k"
```

---

## 3. Utility Commands

### Generate Scaling Arrays

```powershell
python scripts/generate_scaling_array.py -d data/scaling_arrays -e 14 -a -t
```

| Flag | Description |
|---|---|
| `-d` | Output directory for .npy files |
| `-e` | Environment: `14`, `5`, or `14_rel` |
| `-a` | Collect from ALL episodes |
| `-t` | Use test environments |

### Generate Action Spaces

```powershell
# Generate tennet action space
python scripts/develop_action_spaces.py -e l2rpn_case14_sandbox -a tennet -s data/action_spaces/

# Generate medha with rho filtering and greedy selection
python scripts/develop_action_spaces.py -e l2rpn_case14_sandbox -a medha -s data/action_spaces/ -dn -sh "" -rf 1.0 -w 4 -g -ps 100 -i 200
```

### Generate Train/Val/Test Split

```powershell
python scripts/generate_train_val_test_split.py -e l2rpn_case14_sandbox -p "C:\Users\tejas\data_grid2op" -t 10 -v 10
```

### Split Chronics into Per-Day Scenarios

```powershell
python scripts/generate_per_day_scenarios.py -e l2rpn_case14_sandbox_train -p "C:\Users\tejas\data_grid2op" -d 2
```

---

## 4. Key Paths

| Resource | Path |
|---|---|
| Project root | `C:\Users\tejas\OneDrive\Documents\Major Project\rl4pnc\` |
| Training config | `configs/l2rpn_case14_sandbox/ppo_baseline.yaml` |
| Production config | `configs/l2rpn_case14_sandbox/ppo_baseline_batchjob.yaml` |
| Action spaces | `data/action_spaces/l2rpn_case14_sandbox/` |
| Scaling arrays | `data/scaling_arrays/l2rpn_case14_sandbox/` |
| Observation norms | `data/observations_dn/l2rpn_case14_sandbox/` |
| Ray results | `C:\Users\tejas\ray_results\l2rpn_case14_50k\` |
| Callback (live tracking) | `src/rl4pnc/experiments/callback.py` |
| Plot script | `scripts/plot_training_progress.py` |

---

## 5. Quick Start (Copy-Paste)

```powershell
# 1. Train
conda activate rl4pnc; python scripts/train_ppo_baseline.py -f configs/l2rpn_case14_sandbox/ppo_baseline.yaml -wd "C:\Users\tejas\OneDrive\Documents\Major Project\rl4pnc" -s 0 -j run_50k

# 2. Plot results
python scripts/plot_training_progress.py --experiment_dir "C:\Users\tejas\ray_results\l2rpn_case14_50k"

# 3. Evaluate best agent
#    Trial folder = CustomPPO_<env>_<job_id>_<hash>_<timestamp> (created by Ray during training)
#    List available trials:  dir "C:\Users\tejas\ray_results\l2rpn_case14_50k"
python scripts/agent_evaluation.py -a rl -c test -p "C:\Users\tejas\ray_results\l2rpn_case14_50k" -l "C:\Users\tejas\OneDrive\Documents\Major Project\rl4pnc" -t "CustomPPO_l2rpn_case14_sandbox_train_run_50k_a4ab6_2026-03-12_13-31-15" -j eval_test -at 0.95 -lr -b

# 4. Summarize
python scripts/summarize_evaluation_data.py -p "C:\Users\tejas\ray_results\l2rpn_case14_50k" -c 14 -m 8064
```
