# Product Requirements Document (PRD)
## Reinforcement Learning Power Grid Control Platform

### 1. Project Overview
**Purpose:**
The goal is to build a local web-based platform UI that manages the entire Reinforcement Learning (RL) workflow for power grid control. The system trains an RL agent to maintain grid stability by performing actions such as toggling power lines and modifying bus topology. This platform will serve as a comprehensive interface over an existing Python training and evaluation pipeline. The platform is environment-agnostic — it accepts Grid2Op-compatible file formats but is not tied to any specific named environment so that users can train agents on their own custom power network configurations. For initial development and testing, the L2RPN Case14 Sandbox environment (Grid2Op, LightSim2Grid, Stable-Baselines3) will be used.

**Scope:**
The platform covers the complete RL lifecycle, moving progressively from environment configuration to deploying and monitoring trained models. It is fully local, requiring no cloud infrastructure, and designed to run from Visual Studio Code. The platform is single-user with no authentication; all state persists across server restarts.

### 2. Technology Stack
*   **Frontend**: Svelte, Tailwind CSS (for modern UI components), Chart.js or D3.js (for charts and graphs). Built as a static SPA and served by Nginx.
*   **Backend**: Python with FastAPI. Exposes REST APIs and integrates with existing RL training scripts. Asynchronous training jobs via Python multiprocessing or a background job queue.
*   **Database**: PostgreSQL for storing metadata (datasets, model configurations, training results, deployments). Runs as a dedicated Docker container with a named volume for data persistence.
*   **Communication**: REST APIs (standard operations) and WebSockets (real-time updates, logs, live monitoring).
*   **Reverse Proxy**: Nginx — acts as the single entry point on **port 8080**, routing requests to the appropriate backend service and serving the frontend static assets.
*   **Containerisation**: Docker and Docker Compose. Every service (Nginx, Frontend build, Backend API, PostgreSQL) runs in its own container, orchestrated by a single `docker-compose.yml` at the project root.
*   **Execution Environment**: Local execution only. The entire platform is started with `docker compose up` and accessed at `http://localhost:8080`.

### 3. Platform Structure (User Interface)

The platform consists of six major pages arranged as a **linear wizard**. Users progress sequentially from Page 1 through Page 6. A **top progress bar** (numbered step indicator) is always visible, showing the current step and highlighting completed steps. Each page includes a **"Next"** button (enabled only when required inputs on the current page are valid/saved) to advance to the next step. Completed steps in the progress bar are **not** clickable — users must navigate forward only. Pages beyond the current step are locked until prerequisites are satisfied (e.g., Page 2 is inaccessible until Page 1's configuration is saved).

#### Page 1: Power Network Setup and Dataset Upload
*   **Purpose:** Allow users to upload chronics training data and the network topology, then visualize the resulting power grid and inspect dataset statistics.
*   **Features:**
    *   **Chronics Data Upload (Zip):** Upload a single `.zip` archive that contains a top-level folder named `chronics/`. Inside `chronics/`, each subfolder represents one episode (e.g., `chronics/0001/`, `chronics/0002/`, …). Each episode folder contains multiple CSV files describing the episode's time-series data (e.g., `load_p.csv`, `prod_p.csv`, `load_q.csv`, `prod_v.csv`, `maintenance.csv`, `hazards.csv`). On upload the backend extracts the archive, validates the expected folder structure, verifies that each episode folder contains the required CSV files, stores the extracted files locally, and saves metadata in PostgreSQL.
    *   **Automatic Chronics Split (Train / Validation / Test):** After successful validation, the backend automatically partitions chronics into three subsets:
        *   **Training Set:** Used for PPO optimization.
        *   **Validation Set:** Used for periodic policy evaluation during training to monitor generalization, compare checkpoints, and select the best-performing model.
        *   **Testing Set (Single Chronic):** Exactly one chronic reserved for real-time inference and live monitoring in Page 6.
        Split assignments are persisted as dataset metadata and reused consistently across training, validation, and testing workflows.
    *   **Grid Topology Upload:** Upload a JSON file describing the power network. Two formats are accepted:
        *   **Grid2Op-compatible `grid.json`:** A pandapower-format JSON file as used by Grid2Op environments, containing bus, line, generator, load, and transformer tables. The backend parses this directly to extract all network elements.
        *   **Simplified custom JSON:** A lighter-weight schema for non-Grid2Op users, containing a `buses` array (each with `id`, `type`: generator/load/junction) and a `lines` array (each with `from_bus`, `to_bus`, and optional electrical parameters). The backend normalises this into the same internal representation.
        The backend auto-detects the format, validates the file, extracts nodes (buses), edges (transmission lines), and element assignments (generators, loads), and stores the parsed topology in PostgreSQL.
    *   **Network Visualization:** After uploading the topology file, the frontend renders an interactive D3.js force-directed graph of the power network. Nodes represent buses (colour-coded by type: generator bus, load bus, junction), edges represent transmission lines. The graph is read-only by default but supports pan/zoom.
    *   **Dataset Summary Panel:** Displays parsed metadata from the uploaded chronics — including network name, total number of buses and lines, number of generators and loads. Additionally shows the **total number of episodes** (derived from counting the subfolders inside the `chronics/` directory) and the **number of timesteps per episode** (derived by reading one representative episode folder and counting the rows in one of its CSV files, e.g., `load_p.csv`).
    *   **Save Configuration:** Validated topology and chronics dataset references are stored in PostgreSQL for use in subsequent training and evaluation pages.

#### Page 2: Training Configuration
*   **Purpose:** Configure RL parameters, define environment spaces, name the training job, and launch training.
*   **Layout:** Two tabs — **Agent** and **Environment** — with a global controls section below the tabs.

    *   **Tab 1 — Agent (Hyperparameter Configuration):**
        *   Training algorithm is **PPO only**.
        *   Configure actor-critic MLP architecture used by the policy and value networks (default from pipeline: `fcnet_hiddens: [256, 256, 256]`, activation: `relu`).
        *   Set PPO hyperparameters (defaults mirror current pipeline):
            *   `nb_timesteps: 50_000`
            *   `gamma: 0.99`
            *   `lr: 0.0001`
            *   `clip_param: 0.3`
            *   `lambda: 0.95` (GAE)
            *   `entropy_coeff: 0.01`
            *   `vf_loss_coeff: 0.5`
            *   `vf_clip_param: 100`
            *   `kl_coeff: 0.2`
            *   `num_sgd_iter: 10`
            *   `sgd_minibatch_size: 128`
            *   `train_batch_size: 512`
            *   `batch_mode: complete_episodes`
            *   `checkpoint_freq: 10`
            *   `seed: 9`
        *   Defaults must mirror the existing pipeline. Inline validation highlights out-of-range values.

    *   **Tab 2 — Environment (Action & Observation Space):**
        *   **Action Space (Topology Interface):** The backend loads predefined actions from `tennet.json`. Each action is a **composite action** (one action = multiple topology modifications), for example: `S03 -> Bus1`, `S04 -> Bus2`, `Disconnect L07`.
        *   **Interactive Action Creation UI:**
            *   Users view the grid topology graph and select a substation.
            *   A connectivity panel opens showing connected lines and busbar options (Busbar A / Busbar B).
            *   The selected bus assignment (and optional line disconnect) is added to the current action set.
            *   Users repeat this interaction to build custom composite actions.
            *   Users can accept suggested actions, reject actions, or create custom actions.
        *   **Connectivity Visualization:** When a substation is selected, connected lines are visually highlighted and the panel shows an A/B busbar mapping view for each connected element.
        *   **Observation Space:** Multi-select for grid state features to include PPO inputs such as `rho`, topology state (`topo_vect`), line status, and generator/load state features.
        *   **Reward Function:** Select and configure the reward signal used during training. Available predefined reward components:
            *   **Line Overflow Penalty:** Penalises the agent when line loading (`rho`) exceeds safe thresholds. Configurable weight.
            *   **Survival / Timestep Bonus:** Grants a positive reward for each timestep the agent keeps the grid alive. Configurable weight.
            *   **Combined Weighted Sum:** A composite reward that sums the above components using user-specified weights. The UI displays a normalised weight editor (sliders or numeric inputs) so users can balance penalty vs. bonus.
        *   **Environment Controls:** Configure `rho_threshold` (default `0.95`), `n_history` (default `1`), `danger` (default `0.9`), `line_reco` (default `True`), `line_disc` (default `False`), and `reset_topo` (default `0`).
        *   Link to the dataset/topology configured on Page 1 to ensure consistency.

    *   **Global Controls (outside tabs, always visible):**
        *   **Training Job Name (mandatory):** A text input where the user must provide a unique, descriptive name for the training run (e.g., `ppo-case14-lr0.0001`). The field is required — the Launch Training button remains disabled until a valid name is entered. The backend rejects duplicate names.
        *   **Launch Training** button submits the full configuration (agent + environment + job name) to the backend as an async job.

#### Page 3: Training Jobs
*   **Purpose:** View and monitor all submitted training jobs in one place.
*   **Features:**
    *   Lists all submitted training jobs with their name, status (queued, running, completed, failed, cancelled), and submission timestamp.
    *   Selecting a running or completed job opens its **live log panel**: a real-time, WebSocket-streamed terminal-style log showing current timestep, episode reward, episode length, elapsed time, and loss values.
    *   **Cancel / Stop:** A running or queued job can be cancelled by the user. The backend gracefully terminates the training process, saves any partial checkpoint if possible, and marks the job status as `cancelled`.
    *   Completed jobs display a final summary (total timesteps, best reward, wall-clock training time) and a link to the saved model checkpoint.
    *   Failed jobs display the captured error traceback.
    *   Cancelled jobs display the timestep at which cancellation occurred and a link to the partial checkpoint (if saved).

#### Page 4: PPO Agent Evaluation and Comparison
*   **Purpose:** Evaluate PPO-trained models on the uploaded training chronics and compare their performance.
*   **Features:**
    *   **Training & Validation Data Usage:** PPO training uses the **training chronics set**. During training, policy checkpoints are periodically evaluated on the **validation set**.
    *   **Automatic Validation Loop:** Validation runs automatically in the training workflow to monitor generalization, compare policy checkpoints, and select the best-performing model. This process does not modify the training environment state.
    *   **PPO Agent Description:** The deployed/evaluated agent is an actor-critic PPO policy that selects topology actions to keep the grid stable and prevent line overload. The policy network outputs action probabilities over the available action set, while the value network estimates expected return for the current grid state.
    *   **Observation Space (Agent Input):** Includes line loading values (`rho`), topology state, and generator/load state features.
    *   **Action Space (Agent Output):** Combines predefined composite actions loaded from `tennet.json` and user-defined composite actions created in the interface.
    *   **PPO Optimization Signals:** Training/evaluation summaries expose PPO-specific signals including clipped surrogate objective behavior, value loss, entropy term, and advantage estimation trends.
    *   **Model Selection:** Choose single or multiple stored models (from completed or cancelled training jobs that produced a checkpoint) for evaluation.
    *   **Episode Count Configuration:** User specifies how many episodes from the training chronics to evaluate on (default: all available episodes). Episodes are sampled from the chronics uploaded on Page 1.
    *   **Evaluation Metrics:** Display average reward, reward variance, avg/max/min survival steps, and action usage statistics.
    *   **Visualizations:** Reward distributions, survival step histograms, and action frequency distributions.
    *   **Model Comparison:** Side-by-side tables and graphs comparing multiple models' metrics and learning curves.
    *   **Retraining:** Capability to quickly adjust parameters — pre-fills Page 2 with the selected model's configuration — and launch a new training run.

#### Page 5: Model Deployment
*   **Purpose:** Deploy a single trained RL model into an inference environment. Only one deployment may be active at a time; starting a new deployment stops the previous one.
*   **Features:**
    *   **Deployment Configuration:** Select a trained model from completed/cancelled jobs, set inference frequency (timestep interval), and logging verbosity.
    *   **Deployment Chronics Upload:** Upload a separate `.zip` archive of chronics (same folder structure as Page 1) to use as the inference scenario. This allows evaluation/deployment on data distinct from training chronics. The backend validates the zip identically to Page 1.
    *   **Deployment Process:** Backend loads the selected model, initializes the environment using the uploaded deployment chronics and the grid topology from Page 1, and runs continuous inference in a background process.
    *   **Deployment Status:** Display active/stopped status, start time, current timestep, and general execution health.
    *   **Stop Deployment:** A button to gracefully stop the running deployment at any time.

#### Page 6: Live Monitoring Dashboard
*   **Purpose:** Real-time monitoring of the deployed RL agent managing the grid.
*   **Features:**
    *   **Testing Chronic Execution:** Live monitoring runs a **single testing chronic** (reserved during dataset split) in real time using the trained PPO policy.
    *   **Real-Time Simulation Behavior:**
        *   At each timestep, the system evaluates the current grid state and computes line loading values (`rho`).
        *   If no overload is present, the action is `do_nothing`.
        *   If overload is detected, the PPO agent proposes a topology action to stabilize the grid.
    *   **WebSocket Updates:** Stream timesteps, chosen actions, reward values, and environment status.
    *   **Dynamic Charts:** Continuously updating lines/bars for reward over time, power flow stats, and line loading.
    *   **Agent Action Timeline:** A timeline chart with x-axis = timestep and y-axis = action event. The timeline must explicitly distinguish:
        *   agent suggestions,
        *   user-modified actions,
        *   final applied actions,
        *   and `do_nothing`.
        Example events: `t=10 -> do_nothing`, `t=11 -> agent_suggested_S03_bus1`, `t=11 -> user_modified_action`, `t=12 -> do_nothing`.
    *   **Grid Topology Stabilization View:**
        *   Step 1: Show current network topology, current `rho` values, and highlight overloaded lines (`rho` above threshold).
        *   Step 2: Show the PPO agent's proposed topology action.
        *   Step 3: After user approval/modification, apply action and show updated topology with overload reduction/removal highlighted.
    *   **Human-in-the-Loop Action Control:** When the agent proposes an action other than `do_nothing`, the UI shows the suggestion and requires user decision:
        *   **Approve:** Apply suggested action directly.
        *   **Modify:** Edit suggested action with the topology interface, then apply modified action.
        *   **Reject:** Discard suggested action.
        Action suggestion workflow:
        *   Agent detects overload.
        *   Agent proposes composite topology action.
        *   Suggestion is displayed in UI.
        *   User approves, modifies, or rejects.
    *   **Event Logs:** Scrolling console mapping agent actions, grid events, warnings, and errors.
    *   **Grid Visualization:** Live graphical visual of topology changes, line loading, overloaded-line highlighting, and bus states for the entire single testing chronic replay.

### 4. System Architecture & Workflow
*   **Frontend:** Uploads chronics/data, configures PPO training parameters, creates composite topology actions, and monitors training/evaluation/deployment.
*   **Backend (FastAPI):** Loads `tennet.json`, validates datasets/topology, handles API requests, creates training jobs, pushes jobs to queue, and records state/logs in PostgreSQL.
*   **Workers:** Execute PPO training processes; each worker handles one training job and can scale horizontally.
*   **Queue (RabbitMQ):** Distributes training jobs from backend to available workers, enabling multiple concurrent training jobs.
*   **Database (PostgreSQL):** Stores job metadata, configuration parameters, statuses, and logs.
*   **Storage (Local Filesystem):** Stores uploaded chronics, extracted datasets, training outputs, and model checkpoints.
*   **WebSockets:** Stream real-time training and deployment events to the frontend.

### 4.1 Containerisation & Infrastructure (Docker / Docker Compose / Nginx)

The entire platform is containerised and orchestrated with **Docker Compose**. All services are exposed to the user through a single **Nginx** reverse proxy on **port 8080**.

#### 4.1.1 Services (docker-compose.yml)

| Service      | Image / Build Context        | Internal Port | Purpose |
|-------------|------------------------------|---------------|---------|
| **nginx**    | `nginx:alpine` + custom conf | 80 (mapped → host **8080**) | Reverse proxy & static file server |
| **frontend** | Multi-stage build (Node → Nginx serves static) | — (build-only, output copied to nginx) | Svelte SPA build |
| **backend**  | Python 3.11 + FastAPI         | 8000          | REST API, PPO job orchestration, WebSocket server |
| **worker**   | Python 3.11 worker image      | —             | PPO training execution (horizontally scalable) |
| **rabbitmq** | `rabbitmq:3-management-alpine`| 5672          | Job queue for backend-to-worker training dispatch |
| **db**       | `postgres:16-alpine`          | 5432          | PostgreSQL database |

*   **frontend** is a multi-stage Docker build: the first stage installs Node dependencies and runs `npm run build` to produce static assets; the final stage is not a running container — the built files are copied into a shared Docker volume (or directly into the nginx image) so Nginx can serve them.
*   **backend** container mounts a shared volume for uploaded datasets, model checkpoints, and logs (`/app/data`). It connects to `rabbitmq` and `db` over the internal Docker network.
*   **worker** container(s) consume queued PPO jobs from `rabbitmq`, run training, persist checkpoints/logs to shared storage, and update job status in PostgreSQL.
*   **db** uses a named Docker volume (`pgdata`) so that all PostgreSQL data survives `docker compose down` / `docker compose up` cycles.
*   A shared Docker network (`app-net`) connects all services internally.

#### 4.1.2 Nginx Routing Rules

Nginx listens on port 80 inside the container (mapped to **host port 8080**) and routes as follows:

| Path Pattern               | Target                     | Notes |
|---------------------------|----------------------------|-------|
| `/api/**`                  | `http://backend:8000/api/**` | All REST API calls (proxy_pass) |
| `/ws/**`                   | `http://backend:8000/ws/**`  | WebSocket connections (proxy_pass with Upgrade headers) |
| `/**` (everything else)    | Static files from `/usr/share/nginx/html/` | Svelte SPA; falls back to `index.html` for client-side routing |

*   Nginx is configured with `proxy_http_version 1.1`, `Upgrade`, and `Connection` headers for WebSocket support.
*   Large file uploads (chronics zip) are supported via `client_max_body_size` set to an appropriate limit (e.g., 500 MB).
*   Gzip compression is enabled for static assets.

#### 4.1.3 Volumes

| Volume Name  | Mount Point (container) | Purpose |
|-------------|-------------------------|---------|
| `pgdata`     | `/var/lib/postgresql/data` (db) | Persistent PostgreSQL storage |
| `app-data`   | `/app/data` (backend)    | Uploaded datasets, extracted chronics, model checkpoints, logs |
| `static`     | `/usr/share/nginx/html` (nginx) | Built frontend static assets |

#### 4.1.4 Environment Variables

Configured via a `.env` file at the project root (loaded by Docker Compose):

*   `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` — database credentials.
*   `DATABASE_URL` — connection string for the backend (e.g., `postgresql://user:pass@db:5432/powerplant`).
*   `BACKEND_PORT` — internal API port (default `8000`).
*   `NGINX_PORT` — host-exposed port (default `8080`).

#### 4.1.5 Startup & Shutdown

*   **Start:** `docker compose up --build` from the project root builds all images and starts all services. The platform is accessible at `http://localhost:8080`.
*   **Stop:** `docker compose down` stops all containers. Data persists in named volumes.
*   **Full reset:** `docker compose down -v` removes volumes, clearing all stored data.

### 5. Database Design (PostgreSQL)

All data persists across server restarts. The database and local file storage together form the durable state of the platform.

*   **Datasets Table:** Upload info, file paths, chronics metadata (episode count, timesteps per episode), and grid topology references.
*   **Models Table:** Training configurations, hyper-parameters, reward function settings, job name, job status, and `.zip`/`.pt` checkpoint file locations.
*   **Evaluations Table:** Performance metrics, survival steps, reward records, and episode count — linked to one or more models.
*   **Deployments Table:** Tracking the single active deployment instance and historical deployment records (model used, chronics used, start/stop times, final status).

### 5.1 System Workflow
1. User uploads chronics/data from the frontend.
2. Backend receives files.
3. Backend validates chronics structure and auto-splits episodes into training set, validation set, and one single testing chronic.
4. Files are stored in local storage and split metadata is logged in PostgreSQL.
5. User configures PPO training parameters and action space.
6. User starts a training job.
7. Backend creates a job record in PostgreSQL.
8. Backend pushes the job to the RabbitMQ queue.
9. A worker pulls the queued job.
10. PPO training runs on training chronics, with periodic validation on validation chronics for checkpoint comparison and model selection.
11. Training logs and status updates are continuously written to PostgreSQL.
12. Frontend fetches/streams logs and visualizes training progress in real time.
13. Selected/best model is executed on the single testing chronic for real-time monitoring, including agent suggestions and user-approved/modified applied actions.

### 6. System Requirements & Non-Functional Requirements
*   **Local Processing:** Ensures the platform works completely without cloud dependencies. The only prerequisite on the host machine is Docker and Docker Compose.
*   **Single User / No Authentication:** The platform assumes a single local user. No login, sessions, or role-based access control is required.
*   **Containerised Deployment:** All services run inside Docker containers orchestrated by Docker Compose. A single `docker compose up --build` command starts the entire platform. No manual dependency installation is required on the host beyond Docker.
*   **Single Entry Point:** All traffic (frontend, API, WebSocket) flows through an Nginx reverse proxy on **port 8080**. No other ports need to be exposed to the host.
*   **Persistence:** All uploaded datasets, trained model checkpoints, job history, evaluation results, and deployment records persist across container restarts via PostgreSQL (named Docker volume) and a shared data volume for file storage.
*   **Asynchronous Processing:** Long-running training jobs and deployments must not block the API. Background worker processes handle training; the API remains responsive.
*   **Resilience:** Handle training and simulation failures gracefully, piping error logs to the frontend via WebSockets. Cancelled jobs should save partial checkpoints where possible.
*   **UX/UI:** Linear wizard navigation with a top progress bar. Clear visual feedback for validation states, progress indicators for training, and easily interpretable charts. Each page gates progression until required inputs are completed.