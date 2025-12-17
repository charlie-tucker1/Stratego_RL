# Training Guide

This guide explains how to train a PPO agent to play Stratego.

## Quick Start

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Start training
python train.py
```

Training will run for 1,000,000 steps by default

## Monitoring Training

The script logs metrics to TensorBoard. While training is running, open a new terminal:

```bash
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser to see:
- Win rate vs bot (evaluated every 10k steps)
- Average episode reward
- Episode length
- Loss curves

## Configuration

Edit these variables in `train.py` to adjust training:

```python
TOTAL_TIMESTEPS = 1_000_000  # Total training steps
SAVE_FREQ = 50_000           # Save model every N steps
EVAL_FREQ = 10_000           # Evaluate every N steps
N_EVAL_EPISODES = 10         # Episodes per evaluation

# PPO Hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
```

## Evaluating a Trained Model

After training, evaluate your saved model:

```bash
# Evaluate final model
python train.py --eval models/stratego_ppo_final --episodes 100

# Evaluate checkpoint
python train.py --eval models/stratego_ppo_50000_steps --episodes 100
```


## Troubleshooting

**Out of Memory:**
- Reduce `N_STEPS` or `BATCH_SIZE` in train.py
- Use smaller `features_dim` in `StrategoCNN` (currently 512)

**Training Too Slow:**
- Use GPU if available (automatically detected)
- Reduce `N_EVAL_EPISODES` (fewer evaluation games)
- Increase `EVAL_FREQ` (evaluate less often)

**Agent Not Learning:**
- Check TensorBoard
- Verify reward shaping is working (see train.py logs)
- Try adjusting `LEARNING_RATE`

## Files Generated

```
logs/                    # TensorBoard logs
models/                  # Saved model checkpoints
  stratego_ppo_50000_steps.zip
  stratego_ppo_100000_steps.zip
  ...
  stratego_ppo_final.zip
```

## Custom Reward Shaping

The reward function in `stratego_logic.py` can be tuned:

```python
# Current rewards:
# Material: +/- piece rank (1-10)
# Strategic: +2 spy kills marshal, +1 miner defuses bomb
# Territorial: +0.1 per square into enemy territory
# Terminal: +/-100 for win/loss
```



