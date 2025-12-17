# Stratego RL Environment

A reinforcement learning environment for the board game Stratego, built with Gymnasium and designed for integration with PufferLib.

## Overview

Stratego is a classic two-player strategy board game involving hidden information, combat, and tactical positioning. This implementation provides:
- Full Stratego game logic with standard rules
- Partial observability 
- Gymnasium-compatible RL environment
- Scripted bot opponent for training
- 27-channel spatial observation encoding

## Status

**Work in Progress** 

## Game Rules

- 10x10 board with 2 lakes (impassable terrain)
- 40 pieces per player (flags, bombs, and ranked soldiers)
- Goal: Capture opponent's flag
- Combat reveals pieces and resolves by rank
- Special rules:
  - Spies (rank 1) can capture Marshals (rank 10)
  - Miners (rank 3) can defuse Bombs
  - Scouts (rank 2) can move multiple squares

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Stratego_RL.git
cd Stratego_RL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from stratego_logic import StrategoEnv

env = StrategoEnv(render_mode="human")
obs, info = env.reset()

# Play against scripted bot
for _ in range(100):
    mask = info["action_mask"]
    legal_actions = np.where(mask == 1)[0]
    action = np.random.choice(legal_actions)

    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
```

## Observation Space

Shape: `(27, 10, 10)` - 27 channels representing different piece types and board features

**Your pieces (channels 0-11):**
- 0: Your flag
- 1: Your bombs
- 2-11: Your soldiers (ranks 1-10: Spy, Scout, Miner, Sergeant, Lieutenant, Captain, Major, Colonel, General, Marshal)

**Enemy pieces (channels 12-24):**
- 12: Known enemy flag
- 13: Known enemy bombs
- 14-23: Known enemy soldiers (ranks 1-10)
- 24: Unknown enemy pieces 

**Environment (channels 25-26):**
- 25: Empty cells
- 26: Lakes (impassable)

## Action Space

Discrete(3600) - 100 board positions × 4 directions × 9 distances

Action encoding: `action = position * 36 + direction * 9 + (distance - 1)`
- position: 0-99 (board cell)
- direction: 0=up, 1=down, 2=left, 3=right
- distance: 1-9 squares

Full scout movement implemented: Scouts can move multiple squares as in real Stratego.

## Rewards

**Material Rewards:**
- +/- piece rank value when capturing/losing pieces (spy=1, scout=2, ... marshal=10)

**Strategic Bonuses:**
- +2.0 for spy killing marshal (special rule)
- +1.0 for miner defusing bomb

**Territorial Rewards:**
- +0.1 for moving deeper into enemy territory (past middle rows)

**Terminal Rewards:**
- +100.0 for winning (capturing enemy flag)
- -100.0 for losing (your flag captured)

**Penalties:**
- -0.1 for invalid actions

## Training

Train a PPO agent to beat the scripted bot:

```bash
# Start training
python train.py

# Monitor with TensorBoard
tensorboard --logdir ./logs
```

See [TRAINING.md](TRAINING.md) for detailed instructions, configuration options, and expected results.

## Known Limitations

- Placement phase is randomized (not part of action space)
- Bot opponent has access to full board state (not using masked view)

## Roadmap

- [x] Full scout multi-square movement encoding
- [x] Add reward shaping (material, strategic, territorial)
- [ ] Add training examples with PPO
- [ ] Performance benchmarking
- [ ] Unit tests
- [ ] Fix bot to use masked view (more realistic opponent)
- [ ] Prepare for PufferLib PR

## Contributing

This project is in active development. Feedback and suggestions welcome!

## License

MIT License (to be added)

## Acknowledgments

- Built for integration with [PufferLib](https://github.com/PufferAI/PufferLib)
- Uses [Gymnasium](https://gymnasium.farama.org/) API
