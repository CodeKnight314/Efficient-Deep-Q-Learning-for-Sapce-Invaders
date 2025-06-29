# GamePlay.AI
## Overview
**GamePlay.AI** is a reinforcement learning framework focused on training **efficient Double Dueling DQN agents** for Atari games with **dense reward structures**, specifically under strict compute budgets. This project investigates how to **maximize performance per unit of compute**—whether GPU time, frames processed, or dollars spent.

Rather than targeting long-horizon planning games like Tetris, GamePlay.AI prioritizes **reflex-based environments** such as Pong, Breakout, and Space Invaders, where fast learning and feedback loops enable meaningful evaluation under tight constraints. To tackle this, optimizatin design choices are used in models and computing constraint are enforced.

The framework includes:
- A modular and lightweight training pipeline compatible with most Atari environments under `ALE-Py`,
- Architecture optimizations (e.g., convolution factorization) for low FLOPs and memory use,
- Frame-dollar efficiency metrics,
- A GUI runner for real-time performance demos and video recording,
- Pretrained weights for games that have been efficiently "solved".

## Installation
1. Clone the repository:
```bash
git clone https://github.com/CodeKnight314/GamePlay.git
cd GamePlay
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation by running help:
```bash
python3 -m src.main --help
```

## Usage

### Training Mode

Train RL agents on Atari games using the following command structure:

```bash
python3 -m src.main game --id <game> --c <config_path> --o <output_path> --mode train [options]
```

**Required arguments:**
- `--c`: Path to config file (e.g., `src/config/pong.yaml`)
- `--o`: Output path for saving model/weights

**Optional arguments:**
- `--id`: Game choice (`pong`, `breakout`, `invaders`) - default: `pong`
- `--seed`: Random seed for reproducibility - default: `1898`
- `--num_envs`: Number of parallel environments - default: `1`
- `--w`: Path to initial weights file (for resume training)
- `--verbose`: Enable rendering and detailed logs

**Example:**
```bash
python3 -m src.main game --id pong --c src/config/pong.yaml --o models/pong_model --mode train --verbose
```

### Test Mode

Evaluate trained agents by running test episodes:

```bash
python3 -m src.main game --id <game> --c <config_path> --o <output_path> --mode test [options]
```

**Required arguments:**
- `--c`: Path to config file (e.g., `src/config/pong.yaml`)
- `--o`: Output path for saving results

**Optional arguments:**
- `--id`: Game choice (`pong`, `breakout`, `invaders`) - default: `pong`
- `--w`: Path to trained weights file (required for meaningful testing)
- `--num_episodes`: Number of test episodes to run - default: `1`
- `--verbose`: Enable rendering during test episodes

**Example:**
```bash
python3 -m src.main game --id pong --c src/config/pong.yaml --o test_results --mode test --w models/trained_pong.pth --num_episodes 10 --verbose
```

### GUI Mode

Launch interactive GUI for human or AI gameplay:

```bash
python3 -m src.main gui --env <game> --c <config_path> --mode <control_mode> [options]
```

**Required arguments:**
- `--c`: Config path for Environment initialization
- `--mode`: Control mode (`AI` or `human`)

**Optional arguments:**
- `--env`: Game environment (`pong`, `breakout`, `invaders`) - default: `pong`
- `--w`: Weights file for AI mode

**Examples:**
```bash
# Human play
python3 -m src.main gui --env pong --c src/config/pong.yaml --mode human

# AI play with trained weights
python3 -m src.main gui --env pong --c src/config/pong.yaml --mode AI --w src/weights/pong_weights.pth
```