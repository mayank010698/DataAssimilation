Yes, I remember our work on this project well! Let me search through our past conversations to pull together the key details and context for a comprehensive project explanation.I've found a wealth of detail from our conversations. Let me compile a comprehensive project explanation for you.

---

## Project Explanation: Learned Proposal Distributions for Particle Filtering via Flow Matching

### Motivation & Problem Statement

**Core Problem:** In particle filtering for data assimilation, the *bootstrap particle filter (BPF)* uses the transition distribution p(xₜ | xₜ₋₁) as the proposal. This ignores the current observation yₜ, leading to **weight degeneracy** when the likelihood is sharply peaked—most particles end up with negligible weight, producing poor posterior approximations.

**The Optimal Proposal:** The theoretically optimal proposal is p(xₜ | xₜ₋₁, yₜ), which conditions on the current observation. This minimizes weight variance and yields the simplified update:
```
wₜ⁽ⁱ⁾ ∝ wₜ₋₁⁽ⁱ⁾ p(yₜ | xₜ₋₁⁽ⁱ⁾)
```
where weights depend only on the *previous* particle, not the proposed sample. However, sampling from this distribution is intractable for nonlinear systems.

**Your Approach:** Learn an approximate optimal proposal q_φ(xₜ | xₜ₋₁, yₜ) using **rectified flow / flow matching**. This maintains variance-reduction benefits while remaining applicable to complex, nonlinear dynamical systems.

---

### Technical Approach: Rectified Flow Proposals

**Why Rectified Flow (not FlowDAS's stochastic interpolants):**
- **Gaussian source distribution** z ~ N(0, I) enables exact likelihood computation via the continuous normalizing flow (CNF) framework
- FlowDAS uses a learned source distribution, making likelihood computation intractable—but particle filter weight updates *require* evaluating q(xₜ | xₜ₋₁, yₜ)

**The Flow Setup:**
- **Source (s=0):** z ~ N(0, I)
- **Target (s=1):** xₜ ~ p(xₜ | xₜ₋₁) [or p(xₜ | xₜ₋₁, yₜ) with conditioning]
- **Vector field:** v_θ(x, s | xₜ₋₁, yₜ) learned via flow matching loss
- **Log probability:** Computed via change of variables + Hutchinson trace estimator

**Observation Conditioning Methods (for learning p(xₜ | xₜ₋₁, yₜ)):**
1. **FiLM/AdaLN:** Learn scale/shift parameters from y, apply to hidden activations—recommended starting point for low-dim observations
2. **Simple concatenation:** [xₜ₋₁, yₜ, t] as input
3. **Cross-attention:** Most expressive, but overkill for Lorenz63

---

### Key Implementation Details

**Weight Update Formula (general proposal):**
```
w̃ₜ⁽ⁱ⁾ = wₜ₋₁⁽ⁱ⁾ × [p(yₜ | xₜ⁽ⁱ⁾) × p(xₜ⁽ⁱ⁾ | xₜ₋₁⁽ⁱ⁾)] / q(xₜ⁽ⁱ⁾ | xₜ₋₁⁽ⁱ⁾, yₜ)
```

**Normalization Space Handling:**
- Neural networks train on normalized data
- Log probability from RF is computed in normalized space
- Transition distribution p(xₜ | xₜ₋₁) and likelihood p(yₜ | xₜ) are in original space
- Solution: Apply Jacobian correction for the preprocessing transformation, or ensure all terms are computed in the same space

**Dynamics Integration:**
- Particles stored in normalized space (matches data)
- For propagation: normalized → original → integrate Lorenz dynamics → normalized
- Physics (Lorenz equations) only valid in original space

---

### Experimental Setup

**Test Systems:**
- **Lorenz63:** 3D chaotic system, good for initial development
- **Lorenz96:** Higher-dimensional (50-100D), requires more sophisticated architectures

**Architecture Scaling (Lorenz63 → Lorenz96):**
- Lorenz63: Simple MLP with time embedding works well
- Lorenz96: 1D ResNet with circular/periodic convolutions to respect the circular lattice structure; AdaLN for conditioning

**Evaluation:**
- Autoregressive filtering (like FlowDAS)
- Compare RF proposal vs. transition proposal (BPF baseline)
- Metrics: RMSE, ESS (effective sample size), CRPS

---

### Code Structure Overview

| File | Purpose |
|------|---------|
| `data.py` | `DataAssimilationConfig`, `DynamicalSystem` ABC, `Lorenz63`/`Lorenz96`, `DataAssimilationDataset` |
| `bpf.py` / `bpf_improved.py` | `ParticleFilter` ABC, `BootstrapParticleFilter` with arbitrary proposal support |
| `proposals.py` | `Proposal` ABC, `TransitionProposal`, `RFProposalWrapper` |
| `rectified_flow.py` | `VelocityNetwork`, `RFProposal` (PyTorch Lightning module) |
| `rf_dataset.py` | `RFTransitionDataset`, `RFDataModule` for training |
| `train_rf.py` | Training pipeline with callbacks and logging |

---

### Development Stages

1. ✅ **Stage 1:** Learn p(xₜ | xₜ₋₁) without observation conditioning—sanity check that learned transition doesn't degrade performance much vs. known dynamics
2. 🔄 **Stage 2:** Add observation conditioning via FiLM/AdaLN to learn p(xₜ | xₜ₋₁, yₜ)
3. ⏳ **Stage 3:** Scale to Lorenz96 with appropriate architectures
4. ⏳ **Stage 4:** Online adaptation / continual learning (optional)

---

Current Issues & Debugging Status
Rough: I started out with Lorenz 63 experiments using the simple MLP architecture. This seems to work fine for training the flow matching model. When I apply it autoregressively it does super well and gets really low rmse.

Then, for the Lorenz 96 system, I generated datasets with either 10 or 50 dim, and ramped up my model to a 1D resnet. This didn't work, so then to sanity check, I implemented a determistic version of the model. ie. instead of estimated a velocity, output of point estimate of the next state. After fixing a postprocess bug in my Lorenz dataset implementation, that worked great. However, the flow matching model is still a wreck. 

it produces super noisy stuff like this

the residual version is obviously more stable, but still garbage.

So now I'm wondering whats up and whether theres problems in the dataset functionality still, in the architecture, training of the model, inference, what is it.

What Works

Lorenz63 + MLP: Flow matching model trains well, autoregressive evaluation achieves low RMSE ✅
Lorenz96 + Deterministic baseline: Point estimate model (direct next-state prediction) works correctly after fixing a postprocessing bug in the Lorenz96 dataset ✅

What Doesn't Work

Lorenz96 + Flow Matching (1D ResNet): Produces extremely noisy outputs that don't track the true trajectory (see figure). Both standard and residual variants fail.

Observations from the Failure Plot

Generated trajectory (red/orange) shows high-frequency noise completely absent from the true trajectory (blue)
RMSE starts very high (~12), drops around step 50-60, but remains elevated (4-10)
The model seems to capture some coarse structure but adds massive variance