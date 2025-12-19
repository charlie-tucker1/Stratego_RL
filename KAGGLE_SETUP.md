# Training on Kaggle (GPU)

Kaggle provides free GPU access with better package management than Colab. Follow these steps:

## Step 1: Create Kaggle Account

1. Go to https://www.kaggle.com
2. Sign up or log in with Google/GitHub

## Step 2: Create New Notebook

1. Click your profile picture (top right) → Your Work
2. Click "New Notebook"
3. You'll see a blank Python notebook

## Step 3: Enable GPU

1. In the notebook, click the 3 dots menu (top right)
2. Settings → Accelerator → **GPU T4 x2**
3. Click "Save"
4. The notebook will restart with GPU enabled

## Step 4: Upload Your Code

**Option A: From GitHub (Recommended)**

Paste this into the first code cell and run it:

```python
# Clone repository
!git clone https://github.com/charlie-tucker1/Stratego_RL.git
%cd Stratego_RL
!ls
```

**Option B: Upload Files Manually**

1. Right panel → Input → Upload
2. Upload `stratego_logic.py` and `train.py`
3. In code cell: `%cd /kaggle/working`

## Step 5: Install Dependencies

Run this in a code cell:

```python
# Install dependencies
!pip install -q gymnasium stable-baselines3 sb3-contrib tensorboard
```

Note: Kaggle already has compatible numpy, so no special handling needed!

## Step 6: Verify Setup

```python
import numpy as np
import torch
from stratego_logic import StrategoEnv
from sb3_contrib import MaskablePPO

print(f"Numpy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("\nAll imports successful!")
```

## Step 7: Start Training

```python
# Run training
!python train.py
```

## Step 8: Download Models

After training completes:

```python
# Zip models
!zip -r stratego_models.zip models/

# Download
from IPython.display import FileLink
FileLink('stratego_models.zip')
```

Click the link to download your trained models.


## Tips:

1. **Save progress**: Kaggle auto-saves outputs, but download checkpoints periodically
2. **Session time**: You get 30 GPU hours/week (resets weekly)
3. **Internet**: Can clone from GitHub and pip install without issues
4. **Monitoring**: Can use TensorBoard in Kaggle notebooks too

## TensorBoard in Kaggle

```python
%load_ext tensorboard
%tensorboard --logdir ./logs
```

## Resume Training

If session times out, upload your checkpoint and run:

```python
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from train import make_env, MetricsCallback
from stable_baselines3.common.callbacks import CheckpointCallback

# Load checkpoint
model = MaskablePPO.load("models/stratego_ppo_1750000_steps.zip")

# Create environment
env = DummyVecEnv([make_env])
model.set_env(env)

# Continue training
model.learn(
    total_timesteps=250_000,  # Remaining steps
    callback=[
        MetricsCallback(eval_freq=10_000),
        CheckpointCallback(save_freq=50_000, save_path="./models", name_prefix="stratego_ppo")
    ],
    progress_bar=True,
    reset_num_timesteps=False
)

# Save final
model.save("models/stratego_ppo_final")
```

## Troubleshooting

**If you get import errors:**
```python
!pip list | grep -E 'numpy|gymnasium|stable'
```

**If training is slow:**
- Verify GPU is enabled (Settings → Accelerator)
- Check: `torch.cuda.is_available()` returns True

**If session disconnects:**
- Checkpoints are saved every 50k steps in `models/`
- Download them periodically
- Resume from latest checkpoint
