# VLN Benchmark Integration Guide for IsaacLab Arena

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Component Details](#component-details)
   - [H1 Embodiment](#h1-embodiment)
   - [Matterport Background](#matterport-background)
   - [VLN Task](#vln-task)
   - [VLN Metrics](#vln-metrics)
   - [VLNEnvWrapper (Low-Level Policy Bridge)](#vlnenvwrapper)
   - [RSL-RL Loader](#rsl-rl-loader)
   - [VLN Server-Side Policy (VLM)](#vln-server-side-policy)
   - [VLN Client-Side Policy](#vln-client-side-policy)
   - [VLN Environment Builder](#vln-environment-builder)
5. [Control Pipeline](#control-pipeline)
6. [How to Run](#how-to-run)
7. [Dataset Format](#dataset-format)
8. [Configuration Reference](#configuration-reference)
9. [Known Limitations & TODOs](#known-limitations--todos)
10. [Migration Notes from NaVILA-Bench](#migration-notes-from-navila-bench)

---

## Overview

This package (`isaaclab_arena_vln`) adds Vision-Language Navigation (VLN)
benchmark support to IsaacLab Arena. It enables evaluation of VLN agents
(e.g., NaVILA) navigating Matterport 3D indoor scenes on the Unitree H1
humanoid robot.

**Key features:**
- R2R-style VLN episode management (instruction + start/goal + reference path)
- Standard VLN metrics: SPL, Success Rate, Path Length, Distance-to-Goal
- Two-level control: VLM (high-level navigation) + RSL-RL (low-level locomotion)
- Remote VLM policy via IsaacLab Arena's ZeroMQ remote-policy framework
- Demo planner for environment testing without a VLM

**Branch:** `feature/vln-benchmark` (based on `socket_for_policy`)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     VLN Evaluation Pipeline                         │
│                                                                     │
│  ┌──────────────────┐          ┌──────────────────────────────────┐ │
│  │  VLM Server       │          │  Isaac Sim Client                │ │
│  │  (GPU Machine)    │          │  (Simulation Machine)            │ │
│  │                   │          │                                  │ │
│  │  ┌─────────────┐ │  ZeroMQ  │  ┌────────────────────────────┐ │ │
│  │  │ NaVILA/LLaVA│ │ ◄──────► │  │ VlnClientSidePolicy       │ │ │
│  │  │ Model       │ │          │  │   ↓ velocity cmd [vx,vy,ω] │ │ │
│  │  └─────────────┘ │          │  ├────────────────────────────┤ │ │
│  │                   │          │  │ VLNEnvWrapper              │ │ │
│  │  VlnServerSide   │          │  │   ↓ inject cmd to obs buf  │ │ │
│  │  Policy           │          │  │   ↓ run low-level policy   │ │ │
│  └──────────────────┘          │  │   ↓ joint position targets │ │ │
│                                 │  ├────────────────────────────┤ │ │
│                                 │  │ RSL-RL Locomotion Policy   │ │ │
│                                 │  │   (pre-trained, 50 Hz)     │ │ │
│                                 │  ├────────────────────────────┤ │ │
│                                 │  │ Isaac Sim ManagerBasedEnv  │ │ │
│                                 │  │   H1 Robot + Matterport    │ │ │
│                                 │  │   (physics at 200 Hz)      │ │ │
│                                 │  └────────────────────────────┘ │ │
│                                 └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow per step:**
1. Camera RGB image is extracted from the simulation.
2. Image is sent to the remote VLM server via ZeroMQ.
3. VLM generates a navigation command (e.g., "turn left 45 degrees").
4. Command is parsed into velocity `[vx, vy, yaw_rate]` + duration.
5. Velocity command is injected into the proprioceptive observation buffer.
6. The pre-trained RSL-RL locomotion policy converts it to joint position targets.
7. Isaac Sim steps the physics simulation.
8. VLN metrics (position recording) are updated.

---

## File Structure

```
isaaclab_arena_vln/
│
├── __init__.py                          # Package docstring
├── vln_environment.py                   # VLNBenchmarkEnvironment (ExampleEnvironmentBase)
│                                        #   - Composes scene + embodiment + task
│                                        #   - Provides CLI arguments
│
├── assets/
│   ├── __init__.py
│   └── matterport_background.py         # MatterportBackground
│                                        #   - Inherits from Background (Object)
│                                        #   - Loads Matterport USD as static scene
│
├── embodiments/
│   ├── __init__.py
│   └── h1_vln.py                        # H1VlnEmbodiment
│                                        #   - H1VlnSceneCfg: robot articulation, contacts, lights
│                                        #   - H1VlnCameraCfg: pelvis-mounted 512x512 RGB camera
│                                        #   - H1VlnObservationsCfg: policy + proprio groups
│                                        #   - H1VlnActionCfg: JointPositionAction for all joints
│                                        #   - H1VlnCommandsCfg: velocity command generator
│                                        #   - H1VlnEventCfg: joint reset on episode start
│                                        #   - modify_env_cfg: sets dt=0.005, decimation=4
│
├── tasks/
│   ├── __init__.py
│   └── vln_task.py                      # VlnNavTask (TaskBase)
│                                        #   - VlnEpisodeCfg / VlnBenchmarkCfg: episode data
│                                        #   - read_episodes(): loads gzipped JSON R2R dataset
│                                        #   - vln_success_term(): goal-reached termination
│                                        #   - reset_robot_and_goal_from_vln_dataset(): event
│                                        #   - Registers 4 VLN metrics
│
├── metrics/
│   ├── __init__.py
│   └── vln_metrics.py                   # VLN standard metrics
│                                        #   - RobotPositionRecorder: logs [x,y,z] each step
│                                        #   - PathLengthMetric: cumulative Euclidean distance
│                                        #   - DistanceToGoalMetric: geodesic dist via KDTree
│                                        #   - SuccessMetric: dist < radius → success
│                                        #   - SPLMetric: Success * shortest / max(shortest, actual)
│
├── policy/
│   ├── __init__.py
│   ├── vln_policy.py                    # VlnPolicy (ClientSidePolicy / PolicyBase)
│   │                                    #   - Composite policy: remote VLM + local RSL-RL
│   │                                    #   - Works with Arena's policy_runner.py
│   │                                    #   - Lazy-loads RSL-RL, does warmup, returns joint actions
│   │                                    #   - Per-episode instruction tracking via env.extras
│   │
│   ├── vln_server_side_policy.py        # VlnServerSidePolicy (ServerSidePolicy)
│   │                                    #   - Loads NaVILA/LLaVA model at init
│   │                                    #   - Maintains image history buffer
│   │                                    #   - Runs VLM inference → text → velocity cmd
│   │                                    #   - parse_vlm_output_to_velocity(): text parser
│   │
│   ├── vln_client_side_policy.py        # VlnClientSidePolicy (ClientSidePolicy)
│   │                                    #   - Returns velocity commands directly (no RSL-RL)
│   │                                    #   - For velocity-controlled robots or testing
│   │
│   ├── vln_env_wrapper.py               # VLNEnvWrapper (low-level building block)
│   │                                    #   - Bridges high-level vel cmds ↔ low-level joints
│   │                                    #   - update_command(): injects vel into obs buffer
│   │                                    #   - Warmup: 200 steps with zero velocity
│   │
│   └── rslrl_loader.py                  # load_navila_low_level_policy()
│                                        #   - Utility for loading RSL-RL checkpoint
│
├── scripts/
│   ├── __init__.py
│   └── test_vln_server.py               # Minimal test client for ZeroMQ server
│
└── VLN_BENCHMARK_GUIDE.md               # This document
```

**Modified existing files:**
- `setup.py` — added `"isaaclab_arena_vln*"` to `find_packages(include=...)`
- `isaaclab_arena_environments/cli.py` — registered `VLNBenchmarkEnvironment`

---

## Component Details

### H1 Embodiment

**File:** `isaaclab_arena_vln/embodiments/h1_vln.py`

The `H1VlnEmbodiment` configures the Unitree H1 humanoid for VLN:

| Component | Details |
|-----------|---------|
| **Robot USD** | `h1_minimal.usd` from Isaac Nucleus |
| **DOF** | 19 joints (5 per leg, 1 torso, 4 per arm) |
| **Actuators** | `ImplicitActuatorCfg` with PD gains |
| **Actions** | `JointPositionActionCfg` — all joints, scale=0.5 |
| **Physics** | dt=0.005 (200Hz), decimation=4 → 50Hz control |
| **Camera** | 512×512 RGB on pelvis, PinholeCameraCfg |

**Proprioceptive observation vector (concatenated):**
```
Index   Dim   Content
0-2     3     base_ang_vel
3-5     3     projected_gravity
6-8     3     velocity_commands  ← VLNEnvWrapper injects [vx, vy, ω] here
9-27    19    joint_pos_rel
28-46   19    joint_vel_rel
47-65   19    last_action
────────────
Total   ~66
```

**Joint initial positions (standing pose):**
```python
{
    "left_hip_pitch_joint": -0.28,   "right_hip_pitch_joint": -0.28,
    "left_knee_joint": 0.79,         "right_knee_joint": 0.79,
    "left_ankle_joint": -0.52,       "right_ankle_joint": -0.52,
    "left_shoulder_pitch_joint": 0.28, "right_shoulder_pitch_joint": 0.28,
    "left_elbow_joint": 0.52,        "right_elbow_joint": 0.52,
    # hip_yaw, hip_roll, shoulder_roll, shoulder_yaw, torso = 0.0
}
```

### Matterport Background

**File:** `isaaclab_arena_vln/assets/matterport_background.py`

Inherits from `Background` (which inherits from `Object` with `ObjectType.BASE`).
Loads a Matterport 3D scene USD as a static, non-physics background at
`/World/matterport`.

### VLN Task

**File:** `isaaclab_arena_vln/tasks/vln_task.py`

`VlnNavTask` inherits from `TaskBase` and provides:

- **Episode management:** Reads `vln_ce_isaac_v1.json.gz`, converts to
  `VlnEpisodeCfg` objects, samples sequentially.
- **Termination:** `vln_success_term` (position_tolerance=0.3m) + time_out.
- **Events:** `reset_robot_and_goal_from_vln_dataset` — on reset, teleports
  the robot to the episode start pose and stores instruction/goal in
  `env.extras`.
- **Metrics:** Registers PathLength, DistanceToGoal, Success, SPL.

**Episode data stored in `env.extras` after reset:**
```python
env.extras["current_goal_pos"]         # np.array shape (3,)
env.extras["current_instruction"]      # str
env.extras["current_reference_path"]   # np.array shape (K, 3)
env.extras["current_scene_id"]         # str
env.extras["current_episode_id"]       # int
```

### VLN Metrics

**File:** `isaaclab_arena_vln/metrics/vln_metrics.py`

All metrics use `RobotPositionRecorder` (records `root_pos_w` each step).

| Metric | Formula |
|--------|---------|
| **PathLength** | `mean_i( Σ_t ‖pos_t - pos_{t-1}‖ )` |
| **DistanceToGoal** | `mean_i( geodesic_dist(final_pos, goal) )` via KDTree on GT waypoints |
| **Success** | `mean_i( 1 if DTG_i < radius else 0 )` |
| **SPL** | `mean_i( S_i × d_i / max(d_i, l_i) )` where S=success, d=shortest, l=actual |

### VLNEnvWrapper

**File:** `isaaclab_arena_vln/policy/vln_env_wrapper.py`

This is the critical bridge between the high-level VLM and the low-level
locomotion policy.

**Key methods:**

| Method | Description |
|--------|-------------|
| `reset()` | Resets env, runs 200 warmup steps with zero velocity to stabilize H1 |
| `step(action)` | Takes `[vx, vy, ω]`, injects into obs, runs low-level policy, steps sim |
| `update_command(cmd)` | Writes velocity into `obs[:, 6:9]` (history wrapper) or `obs[:, 9:12]` |
| `_check_same_pos()` | Detects stuck robot (same position for 1000 steps) |
| `set_stop_called()` | Signals episode termination from VLM "STOP" command |

**Command injection detail:**
```
With RslRlVecEnvHistoryWrapper:
  obs layout: [policy_obs(~66), proprio_history(~66 × H)]
  velocity_commands are at obs[:, 6:9]
  Also update: env.proprio_obs_buf[:, -1, 6:9]

Without history wrapper (standard RslRlVecEnvWrapper):
  obs layout: [base_ang_vel(3), gravity(3), vel_cmd(3), joints..., actions...]
  velocity_commands are at obs[:, 9:12]
```

### RSL-RL Loader

**File:** `isaaclab_arena_vln/policy/rslrl_loader.py`

```python
vln_env = load_navila_low_level_policy(
    env=env,                          # Isaac Lab ManagerBasedRLEnv
    log_root_path="/path/to/logs",    # RSL-RL training log root
    agent_cfg_yaml="/path/to/agent.yaml",
    policy_run_name="run_001",
    policy_checkpoint_id=0,           # model_0.pt
    task_name="h1",
    max_length=10000,
)
# Returns VLNEnvWrapper with loaded locomotion policy
```

**Expected directory structure for RSL-RL logs:**
```
{log_root_path}/rsl_rl/{experiment_name}/{policy_run_name}/models/model_0.pt
```

### VLN Server-Side Policy

**File:** `isaaclab_arena_vln/policy/vln_server_side_policy.py`

Runs on the GPU machine. Wraps a NaVILA / LLaVA model.

**VLM text → velocity command parsing:**
```
"turn left 45"    → [0.0, 0.0, +π/6],  dur=1.5s
"turn left 30"    → [0.0, 0.0, +π/6],  dur=1.0s
"turn left 15"    → [0.0, 0.0, +π/6],  dur=0.5s
"turn right 45"   → [0.0, 0.0, -π/6],  dur=1.5s
"move forward 75" → [0.5, 0.0, 0.0],   dur=1.5s
"move forward 50" → [0.5, 0.0, 0.0],   dur=1.0s
"move forward 25" → [0.5, 0.0, 0.0],   dur=0.5s
"stop"            → [0.0, 0.0, 0.0],   dur=0.0s
```

### VLN Client-Side Policy

**File:** `isaaclab_arena_vln/policy/vln_client_side_policy.py`

Implements NaVILA-style command scheduling:
- Query VLM once → get velocity + duration
- Hold velocity for `duration` seconds (converted to N env steps)
- Only query VLM again when duration expires
- On "STOP", signals `VLNEnvWrapper.set_stop_called(True)`

### VLN Environment Builder

**File:** `isaaclab_arena_vln/vln_environment.py`

Follows the `ExampleEnvironmentBase` pattern:

```python
class VLNBenchmarkEnvironment(ExampleEnvironmentBase):
    name = "VLN_Benchmark"

    def get_env(self, args_cli):
        background = MatterportBackground(usd_path=args_cli.usd_path)
        embodiment = H1VlnEmbodiment(enable_cameras=True)
        task = VlnNavTask(robot=embodiment, r2r_dataset_path=args_cli.r2r_dataset_path)
        scene = Scene(assets=[background])
        return IsaacLabArenaEnvironment(
            name=self.name, scene=scene, embodiment=embodiment, task=task
        )
```

---

## Control Pipeline

### Step-by-step walkthrough of one evaluation step:

```
1. VlnClientSidePolicy.get_action(env, obs)
   │
   ├─ Is it time to query the VLM? (step_count >= target_step)
   │   ├─ YES: Pack camera obs → ZeroMQ → VLM server
   │   │       VLM returns "turn left 45" → parse → [0, 0, π/6], dur=1.5s
   │   │       Set target_step = step_count + int(1.5 / 0.02) = step_count + 75
   │   └─ NO:  Reuse last velocity command
   │
   ├─ Return velocity cmd tensor [1, 3]
   │
2. VLNEnvWrapper.step(action=[vx, vy, ω])
   │
   ├─ update_command(): obs[:, 6:9] = [vx, vy, ω]
   │
   ├─ low_level_policy(obs) → joint_pos_targets [1, 19]
   │
   ├─ env.step(joint_pos_targets)
   │   └─ Isaac Sim: physics step (dt=0.005 × 4 = 0.02s)
   │       Robot moves, camera renders new frame
   │
   ├─ Extract camera RGB from info["observations"]["camera_obs"]
   │
   ├─ Check termination:
   │   ├─ env done (time_out or success)?
   │   ├─ stuck for 1000 steps?
   │   ├─ max_length reached?
   │   └─ STOP called?
   │
   └─ Return (camera_obs, reward, done, info)

3. If done → reset → sample next episode → teleport robot → clear history
```

---

## How to Run

### Prerequisites

1. IsaacLab Arena installed with `feature/vln-benchmark` branch
2. RSL-RL installed (`pip install rsl-rl`)
3. A pre-trained RSL-RL low-level locomotion checkpoint
4. Matterport USD scene files
5. VLN-CE-Isaac dataset (`vln_ce_isaac_v1.json.gz`)
6. NaVILA/LLaVA model checkpoint (for the VLM server Docker image)

### Start the VLM Server

Build and run the VLM server Docker container:

```bash
# Uses docker/Dockerfile.vln_server + docker/run_vln_server.sh
# Same pattern as docker/run_gr00t_server.sh
bash docker/run_vln_server.sh -m /path/to/navila-llama3-8b-8f
```

### Run Evaluation

Uses Arena's standard ``policy_runner.py`` with ``VlnPolicy``:

```bash
python -m isaaclab_arena.evaluation.policy_runner \
    --headless --num_envs 1 \
    --policy_type isaaclab_arena_vln.policy.vln_policy.VlnPolicy \
    --remote_host localhost --remote_port 5555 \
    --ll_checkpoint_path /path/to/rsl_rl/model_0.pt \
    --ll_agent_cfg /path/to/agent.yaml \
    --num_episodes 10 \
    VLN_Benchmark \
    --usd_path /path/to/matterport_usd/5q7pvUzZiYa/5q7pvUzZiYa.usd \
    --r2r_dataset_path /path/to/vln_ce_isaac_v1.json.gz
```

``VlnPolicy`` internally handles:
  - VLM query scheduling (holds velocity commands for their duration)
  - Per-episode instruction updates (reads from ``env.extras``)
  - RSL-RL low-level policy loading and warmup
  - Velocity command injection and joint action generation

---

## Dataset Format

### VLN-CE-Isaac Dataset (`vln_ce_isaac_v1.json.gz`)

```json
{
  "episodes": [
    {
      "episode_id": "12345",
      "scene_id": "matterport/5q7pvUzZiYa",
      "start_position": [9.1, 3.8, 1.1],
      "start_rotation": [0.7, 0.0, 0.0, 0.0],
      "reference_path": [
        [9.1, 3.8, 1.1],
        [8.5, 4.2, 1.1],
        [7.0, 5.0, 1.1]
      ],
      "instruction": {
        "instruction_text": "Walk through the hallway and turn left at the kitchen.",
        "instruction_tokens": ["Walk", "through", "the", ...]
      },
      "goals": [{"position": [7.0, 5.0, 1.1], "radius": 3.0}],
      "trajectory_id": "..."
    }
  ]
}
```

### Matterport USD Files

Expected path structure:
```
matterport_usd/
├── 5q7pvUzZiYa/
│   └── 5q7pvUzZiYa.usd
├── 1LXtFkjw3qL/
│   └── 1LXtFkjw3qL.usd
└── ...
```

---

## Configuration Reference

### H1 VLN Embodiment Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Sim dt | 0.005 s (200 Hz) | h1_matterport_base_cfg.py |
| Decimation | 4 | h1_matterport_base_cfg.py |
| Control freq | 50 Hz | 200/4 |
| Camera resolution | 512 × 512 | NaVILA default |
| Camera mount | pelvis | NaVILA default |
| Camera FOV | ~54° horizontal | h1_matterport_base_cfg.py |
| Joint action scale | 0.5 | NaVILA default |
| Default offset | True | Uses init joint pos as offset |

### VLN Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| episode_length_s | 60.0 | Max seconds per episode |
| success_radius | 3.0 | Goal success distance (meters) |
| position_tolerance | 0.3 | Termination distance (meters) |

### VLNEnvWrapper Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| warmup_steps | 200 (H1) | Steps with zero velocity to stabilize |
| max_length | 10,000 | Max low-level steps per episode |
| stuck_threshold | 1,000 | Steps at same position before terminating |
| cmd_indices | 6:9 | Velocity command location in obs buffer |

### VLM Server Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_video_frames | 8 | Frames in VLM context window |
| conv_mode | "llama_3" | LLaVA conversation template |
| default_duration | 0.5 s | Fallback command hold time |

---

## Known Limitations & TODOs

### Current Limitations

1. **Single scene per run:** Each evaluation run uses one Matterport USD.
   To evaluate across scenes, run the benchmark multiple times with
   different `--usd_path` and `--episode_start`/`--episode_end` ranges.

2. **H1 USD path:** Currently points to the Isaac Nucleus `h1_minimal.usd`.
   If your NaVILA low-level policy was trained with a different USD, you
   need to update `_H1_USD_PATH` in `h1_vln.py`.

3. **Joint configuration:** The joint names and default positions must match
   the RSL-RL training configuration exactly. If they don't, the low-level
   policy will produce incorrect actions.

4. **Proprio observation layout:** The velocity command injection indices
   (6:9 or 9:12) depend on the exact observation concatenation order.
   Verify this matches your trained policy.

5. **No height scanner:** The current embodiment does not include a height
   scanner (raycaster). If your policy was trained with height scans, you
   need to add it.

### Multi-env Limitation (num_envs > 1)

**Currently only ``num_envs=1`` is supported at the policy level.**

The simulation layer (task, env wrapper) has per-env data structures that
could support multiple envs, but the VLM policy layer has a fundamental
limitation:

| Layer | Multi-env ready? | Details |
|-------|-----------------|---------|
| **VlnNavTask** (episodes, goals) | ✅ Yes | Per-env episodes, `env.extras["current_goal_pos"]` shape `[N, 3]` |
| **vln_success_term** | ✅ Yes | Per-env goal checking |
| **VLNEnvWrapper** (stuck, stop) | ✅ Yes | Per-env stuck detection, per-env stop flags |
| **VLM Server** | ❌ No | Single instruction + single image history buffer |
| **VLM Client Policy** | ❌ No | Single scheduling state, single instruction |
| **Arena `set_task_description`** | ❌ Design issue | Takes a single string, not per-env |

**Root cause:** Arena's remote policy framework sends ``set_task_description(str)``
as a separate RPC call from ``get_action(obs)``.  For VLN, the instruction
changes every episode and is different per env.  With the current design:
  - The server has no concept of "which env this observation belongs to"
  - Image history from different envs would mix together
  - Instructions from different envs would overwrite each other

**Proposed solutions (require Arena framework discussion):**
  1. Bundle ``task_description`` inside the ``get_action()`` observation payload
  2. Add ``env_id`` to all RPC calls (``get_action``, ``set_task_description``, ``reset``)
  3. Per-env server sessions or multiple server connections

**Future: single-scene multi-robot**
A separate optimization: placing multiple robots in a single shared Matterport
scene (instead of one copy per env) to reduce GPU memory usage.  This would
require changes to IsaacLab's ``{ENV_REGEX_NS}`` asset replication pattern.

### TODOs

- [ ] Add multi-scene episode runner (auto-switch Matterport USD per scene group)
- [ ] Single-scene multi-robot mode (multiple robots in one Matterport scene)
- [ ] Batch VLM inference for multi-env (requires VLM server changes)
- [ ] Add Go2 and G1 embodiments for VLN
- [ ] Support height scanner observation for policies trained with it
- [ ] Add video recording for evaluation visualization
- [ ] Add OracleSuccess and OracleNavigationError metrics
- [ ] Integrate with NaVILA's RslRlVecEnvHistoryWrapper for history-based policies
- [ ] Add unit tests for metrics and episode parsing
- [ ] Support batch evaluation across all dataset episodes via `run_benchmark.py`

---

## Migration Notes from NaVILA-Bench

### What changed from the NaVILA-Bench architecture:

| NaVILA-Bench | IsaacLab Arena VLN | Why |
|---|---|---|
| Raw TCP socket VLM server | ZeroMQ ServerSidePolicy | Uses Arena's remote-policy framework |
| Custom `navila_eval.py` script | `run_vln_benchmark.py` + Arena `policy_runner.py` | Fits Arena's evaluation pipeline |
| `omni.isaac.vlnce` extension | `isaaclab_arena_vln` package | Part of Arena's package structure |
| `H1_MINIMAL_CFG` from isaaclab_assets | `H1VlnSceneCfg` with explicit joint config | Full control over embodiment |
| `RslRlVecEnvHistoryWrapper` | `RslRlVecEnvWrapper` (standard) | Simpler; add history wrapper if needed |
| Gym registration via `gym.register()` | `ArenaEnvBuilder.build_registered()` | Arena's composable environment pattern |
| `MatterportImporterCfg` | `MatterportBackground` (Background class) | Arena's asset abstraction |
| Direct `measures.py` with `MeasureManager` | `MetricBase` + `RecorderManager` (HDF5) | Arena's metrics framework |

### Key code mappings:

```
NaVILA-Bench                              → IsaacLab Arena VLN
───────────────────────────────────────────────────────────────────
config/h1/h1_matterport_base_cfg.py       → embodiments/h1_vln.py
utils/wrappers.py :: VLNEnvWrapper         → policy/vln_env_wrapper.py
utils/wrappers.py :: RslRlVecEnvHistory    → (use standard RslRlVecEnvWrapper)
utils/eval_utils.py :: get_vel_command     → policy/vln_server_side_policy.py :: parse_vlm_output_to_velocity
utils/measures.py                          → metrics/vln_metrics.py
scripts/vlm_server.py                      → policy/vln_server_side_policy.py (integrated)
scripts/navila_eval.py                     → scripts/run_vln_benchmark.py
scripts/demo_planner.py                    → scripts/run_vln_benchmark.py :: run_demo_planner()
```

---

*Document generated on 2026-02-15 for the `feature/vln-benchmark` branch.*
