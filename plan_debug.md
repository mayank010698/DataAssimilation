# Plan: Debug and Improve 1D ResNet for Lorenz 96

The current 1D ResNet model is failing to learn the dynamics ("disaster" performance). We suspect architecture limitations (receptive field), training data issues, or evaluation instability.

## 1. Baseline Evaluation (Immediate)

- **Goal**: Confirm data quality and establish a performance baseline.
- **Action**: Run the Bootstrap Particle Filter (BPF) on the existing dataset.
    - If BPF fails, the data/config is likely broken.
    - If BPF works, the issue is definitely the `resnet1d` model.
- **Tool**: Run the newly created `run_bpf_l96.py` script.

## 2. Architecture Analysis & Improvement

- **Issue**: Lorenz 96 dynamics at index $i$ depend on $x_{i-2}$. The current kernel size of 3 (centered) only sees $x_{i-1}, x_i, x_{i+1}$. It relies on depth to propagate $x_{i-2}$ info, which is inefficient and hard to learn.
- **Fix**: Increase `kernel_size` to 5 in `ResNet1DVelocityNetwork`. This gives a receptive field of $[-2, +2]$ in the first layer, covering all dependencies.
- **Refinement**: Verify `conditioning_method`. The user used `adaln`. `concat` is often more robust for simple dynamical systems.

## 3. Training Stability

- **Issue**: `window=1` might be too short for noisy data, leading to error accumulation.
- **Fix**: If kernel size fix isn't enough, we will increase context window to `window=3` or `window=5`. (This requires code changes in `RFTransitionDataset`).

## 4. Execution Steps

1.  Run BPF to check data integrity.
2.  Modify `proposals/architectures/resnet1d.py` default `kernel_size` to 5.
3.  Retrain the model (user action required, or we can launch a short run).
4.  Evaluate the new model.