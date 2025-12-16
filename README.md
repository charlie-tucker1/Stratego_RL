# Stratego RL Environment

A reinforcement learning environment for the board game Stratego, built with Gymnasium and designed for integration with PufferLib.

## Overview

Stratego is a classic two-player strategy board game involving hidden information, combat, and tactical positioning. This implementation provides:
- Full Stratego game logic with standard rules
- Partial observability (fog of war)
- Gymnasium-compatible RL environment
- Scripted bot opponent for training
- 27-channel spatial observation encoding

## Status

🚧 **Work in Progress** - Currently in development for eventual PufferLib contribution

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
- 24: Unknown enemy pieces (fog of war)

**Environment (channels 25-26):**
- 25: Empty cells
- 26: Lakes (impassable)

## Action Space

Discrete(400) - 100 board positions × 4 directions (up, down, left, right)

Action encoding: `action = (row * 10 + col) * 4 + direction`

**Note:** Scout multi-square moves are simplified to single-step moves in current version.

## Rewards

- +1.0 for winning (capturing enemy flag)
- -1.0 for losing (your flag captured)
- 0.0 during gameplay
- Small penalty (-0.1) for invalid actions

## Training

Coming soon - training scripts with PPO/other RL algorithms

## Known Limitations

- Scout movement simplified to single squares (not full multi-square moves)
- Sparse rewards (only terminal rewards currently)
- Placement phase is randomized (not part of action space)

## Roadmap

- [ ] Fix scout multi-square movement encoding
- [ ] Add reward shaping (piece captures, reveals, etc.)
- [ ] Add training examples with PPO
- [ ] Performance benchmarking
- [ ] Unit tests
- [ ] Prepare for PufferLib PR

## Contributing

This project is in active development. Feedback and suggestions welcome!

## License

MIT License (to be added)

## Acknowledgments

- Built for integration with [PufferLib](https://github.com/PufferAI/PufferLib)
- Uses [Gymnasium](https://gymnasium.farama.org/) API
