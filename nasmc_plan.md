# Implementing NASMC as a Learned Proposal

> **Purpose.** A concrete implementation plan for adding the Neural Adaptive
> Sequential Monte Carlo (NASMC) proposal-learning method of
> Gu, Ghahramani & Turner (2015, arXiv:1506.03338) to this codebase, so that
> it can be compared head-to-head against the existing rectified-flow (RF)
> proposal on the same datasets and through the same particle-filter
> evaluation harness (`eval.py`, `run.py`).
>
> Scope: this document assumes the reader has read the technical overview
> (`docs/proposal_learning_overview.md` or equivalent) and the NASMC paper.
> It focuses on the *differences* between RF and NASMC, the *decisions* that
> need to be made to fit NASMC into the existing infrastructure, and the
> *new modules* that need to be written.

---

## 1. Summary

NASMC trains a parametric proposal \( q_\phi(x_t \mid x_{t-1}, y_t) \) by
minimising the inclusive KL divergence
\( \mathrm{KL}[p(x_{1:T} \mid y_{1:T}) \,\|\, q_\phi(x_{1:T} \mid y_{1:T})] \)
using samples produced by running SMC with the current proposal. Concretely,
for each training trajectory, we run a full forward SMC sweep, collect
weighted particles \( \{(x_t^{(n)}, \tilde w_t^{(n)})\} \), and take a
gradient step on

$$
\mathcal{L}_\text{NASMC}(\phi)
= -\sum_{t=1}^{T} \sum_{n=1}^{N} \tilde w_t^{(n)}
    \log q_\phi \!\left( x_t^{(n)} \,\middle|\,
      x_{t-1}^{A_{t-1}^{(n)}}, y_t \right),
$$

where particles and weights are treated as stop-gradient values and the
ancestor index \( A_{t-1}^{(n)} \) is the SMC-tracked parent of particle
\( n \) at time \( t \).

This differs from RF in three ways that matter for implementation:

1. **Training loss is global, not local.** A single training example is a
   full trajectory \( (x_{1:T}, y_{1:T}) \), not a one-step triple.
2. **SMC is inside the training loop.** Each gradient step requires a
   forward SMC pass with \( q_\phi \) as proposal.
3. **The proposal density is tractable by construction.** Unlike RF,
   \( \log q_\phi \) does not require integrating an ODE or a Hutchinson
   trace. A Gaussian (or mixture) head gives an analytic
   \( \log q_\phi \) for free.

Everything else stays the same: same datasets, same observation model, same
particle filters, same scaling convention, same velocity/backbone networks,
same `ProposalDistribution` ABC.

---

## 2. What NASMC is (focused recap)

The full paper covers more than we need; the operational content is:

**Objective.** Minimise inclusive KL from true posterior to proposal:
\( \mathrm{KL}[p_\theta(x_{1:T} \mid y_{1:T}) \,\|\, q_\phi(x_{1:T} \mid y_{1:T})] \).
The gradient is an expectation under the posterior of
\( \nabla_\phi \log q_\phi \), which is replaced with its SMC estimate using
the filtering approximation (eq. 1 in the paper):

$$
-\nabla_\phi \mathrm{KL}
\;\approx\;
\sum_t \sum_n \tilde w_t^{(n)} \,
\nabla_\phi \log q_\phi \!\left( x_t^{(n)} \,\middle|\,
  x_{1:t-1}^{A_{t-1}^{(n)}}, y_{1:t} \right).
$$

**Proposal family.** Any parametric density supporting sampling and
log-density. The paper uses LSTM + MDN; we will start with simpler forms.

**Proposal input.** The paper's general form is
\( q_\phi(x_t \mid x_{1:t-1}, y_{1:t}) \), i.e. full history on both sides.
A common special case is \( q_\phi(x_t \mid x_{t-1}, y_t) \), which matches
the existing `ProposalDistribution` ABC exactly.

**Two useful variants from the paper.**
- **`-f-` variant:** parametrise the proposal for the process noise
  \( v_t \) rather than the state \( x_t \), so that the deterministic
  dynamics \( f(x_{t-1}) \) are baked into the mean. Substantially better
  in the paper's benchmark.
- **`-MD-` variant:** mixture density network output instead of a single
  Gaussian, to capture multi-modal posteriors.

**Flexibility of training samples.** The paper notes the particles used for
the gradient do not have to come from running SMC with the current
\( q_\phi \); they can come from the bootstrap filter or from the generative
model. This matters for warm-starting.

---

## 3. How NASMC differs from RF

| Aspect | RF (current) | NASMC (new) |
| --- | --- | --- |
| Training data | One-step triples \((x_{t-1}, x_t, y_t)\) from ground truth | Full trajectories \((x_{1:T}, y_{1:T})\) |
| Loss | Flow-matching MSE, purely local, no SMC | Weighted log-likelihood of SMC-inferred posterior |
| Inner loop at train time | None beyond the minibatch | Full forward SMC with \(N_\text{train}\) particles |
| Sampling at inference | Euler integration of \(v_\theta\) | Direct draw from Gaussian / MDN |
| `log_prob` at inference | Backward integration + trace estimator | Closed-form (Gaussian / MDN log-density) |
| Cost per weight eval | \(O(N_\text{steps} \cdot D)\) plus trace probes | \(O(D)\) |
| Multi-modality | Natural (flows) | Needs MDN head |
| Training instability | Local loss is very stable | SMC collapse is a real failure mode; needs warm-start |

The last row is the most important practical risk. It drives several of the
design decisions below.

---

## 4. Design decisions

These are the decisions that shape the implementation. For each I state the
recommendation and the reason; alternatives are listed explicitly so a
future iteration can revisit them.

### 4.1 Proposal family: start with a diagonal Gaussian, add MDN later

Recommended v1: diagonal-covariance Gaussian head
\( q_\phi(x_t \mid x_{t-1}, y_t) = \mathcal{N}(\mu_\phi, \mathrm{diag}(\sigma_\phi^2)) \)
with \( (\mu_\phi, \log\sigma_\phi) \) produced by an MLP or ResNet1D.

Reasons:
1. Closed-form log-density (no trace estimator, no ODE integration).
2. Reparameterisable sampling (\( x_t = \mu_\phi + \sigma_\phi \odot \varepsilon \)).
3. Matches the RF interface exactly at the `ProposalDistribution` level.
4. Comparable parameter count to RF's velocity head: the existing MLP /
   ResNet1D architectures output a vector of dimension \(D\) for the
   velocity; for a diagonal Gaussian we output \(2D\) (mean and log-std).
5. Full covariance is infeasible at Lorenz-96 \(D=40\) and KS \(D=64\text{–}128\).

Defer: mixture density network (`-MD-` variant) as a config option for v2.
See §12.

### 4.2 Markov one-step conditioning: match the existing ABC

Recommended v1: keep the proposal Markovian in \(x\), i.e.
\( q_\phi(x_t \mid x_{t-1}, y_t) \). Do not introduce an RNN over
\(x_{1:t-1}\) for now.

Reasons:
1. This is exactly what the `ProposalDistribution` ABC supports.
2. No changes needed in `models/bpf.py`, `models/apf.py`, or any filter
   class.
3. The paper's benchmark shows the Markovian `-f-` variant already matches
   or beats EKPF/UPF. The RNN gain is largely from capturing non-Markovian
   *model* structure, which our dynamical systems do not have (they are all
   Markov in \(x\)).
4. The RF baseline is also Markov in \(x\); keeping NASMC Markov preserves
   apples-to-apples comparison.

Defer: RNN-parametrised proposal as a v2 extension. See §12.

### 4.3 Parametrisation: match RF's `predict_delta=True`, offer `-f-` as an option

Recommended v1 default: have the network predict the mean and log-std of
the *delta* \(x_t - x_{t-1}\) in scaled space, i.e.

$$
q_\phi(x_t \mid x_{t-1}, y_t)
= \mathcal{N}\!\left(x_{t-1} + \mu_\phi(x_{t-1}, y_t),\;
    \mathrm{diag}(\sigma_\phi^2(x_{t-1}, y_t)) \right).
$$

Reasons:
1. This is exactly the parametrisation used by RF with `predict_delta=True`,
   so the model capacity requirements are comparable.
2. It avoids the large dynamic range of absolute \(x_t\) when the system is
   chaotic, which is the same argument that motivates `predict_delta` in RF.

Offer as a config flag: the `-f-` variant, which predicts around the
deterministic integrator output \(\tilde f(x_{t-1}) = \mathrm{RK4}(x_{t-1})\)
instead of around \(x_{t-1}\):

$$
q_\phi(x_t \mid x_{t-1}, y_t)
= \mathcal{N}\!\left(\tilde f(x_{t-1}) + \mu_\phi(x_{t-1}, y_t),\;
    \mathrm{diag}(\sigma_\phi^2(x_{t-1}, y_t)) \right).
$$

This needs `system.integrate` to be called inside training and inference.
It is implementable but noisier in practice because it couples training
to the same integrator used for data generation; worth having as a switch
but not the default.

### 4.4 Architecture: reuse `proposals/architectures/`

Recommended: reuse `MLPVelocityNetwork` (Lorenz-63, double-well) and
`ResNet1DVelocityNetwork` (Lorenz-96, KS) as the backbones. Replace their
vector-of-dimension-\(D\) output with a \(2D\) output head: first \(D\)
channels are \(\mu\), next \(D\) are \(\log\sigma\).

Drop the flow-time input \(s\), since Gaussian proposals do not have an
internal integration time. The rest of the input signature
(`x_prev`, `y_full`, `mask`, optional `traj_time_embed`) is reused verbatim.

Concretely, add a factory function
`create_gaussian_head_network(architecture, state_dim, obs_dim, ...)` in
`proposals/architectures/__init__.py` that wraps the existing backbones and
replaces the output projection.

### 4.5 Warm-start: local MLE pretraining, then switch to NASMC

The NASMC objective is fragile early in training: a random \(q_\phi\) gives
SMC runs that collapse to a single ancestor, at which point the weighted
log-likelihood gradient points in arbitrary directions. Several mitigations
from the paper and standard practice:

1. **Local MLE pretraining (recommended).** Before starting NASMC,
   pretrain \(q_\phi\) by maximising the plain log-likelihood of ground-truth
   transitions: \(\max_\phi \mathbb{E}_{(x_{t-1}, x_t, y_t) \sim \text{data}}
   [\log q_\phi(x_t \mid x_{t-1}, y_t)]\). This reuses the existing
   `RFTransitionDataset` and `RFDataModule` as-is. The loss is just the
   Gaussian log-density. This gives a fully trained baseline proposal in
   exactly the same time budget as RF (it is basically the "denoising
   autoencoder" baseline).
2. **Bootstrap-sampled particles early on.** The paper notes that the
   particles used to form the gradient do not have to come from running
   SMC with \(q_\phi\). During a "warm-up" phase of NASMC training we can
   run SMC with the bootstrap proposal
   \(q = p(x_t \mid x_{t-1})\) and use those particles to train \(q_\phi\).
   After the warm-up, switch to using \(q_\phi\) itself as the SMC
   proposal.
3. **Anneal number of particles.** Start with a large \(N_\text{train}\)
   (say 128) so the weighted empirical distribution is richer; reduce if
   speed is an issue.

Recommended schedule:
- Phase 1, "pretrain": 50–100 epochs of local MLE on `RFTransitionDataset`.
  Produces a checkpoint `nasmc_pretrain_<run>.ckpt`.
- Phase 2, "refine": initialise from the pretrained checkpoint and train
  for a further budget with the NASMC objective, running SMC with
  \(q_\phi\) each step. Produces `nasmc_final_<run>.ckpt`.

Both phases should be runnable from the same CLI (`train_nasmc.py`) with a
`--phase {pretrain,refine,both}` flag. In "both" mode the script runs
phase 1 then hands the final weights to phase 2 automatically.

### 4.6 SMC inner loop: detached particles, importance-weighted log-density gradient

The gradient estimator in the paper treats particles and weights as
stop-gradient values. This is both correct and computationally convenient:
no differentiable resampling, no reparameterisation tricks through SMC.

Implementation recipe for phase 2:
1. Sample a minibatch of trajectories \((x_{1:T}^{(b)}, y_{1:T}^{(b)})\).
2. Under `torch.no_grad()`, run a forward SMC sweep for each trajectory
   with the current \(q_\phi\):
   - At each step \(t\), sample \(N\) particles \(\{x_t^{(n)}\}\) from
     \(q_\phi(\cdot \mid x_{t-1}^{A_{t-1}^{(n)}}, y_t)\).
   - Compute log importance weights
     \( \ell_t^{(n)} = \log p(y_t \mid x_t^{(n)})
       + \log p(x_t^{(n)} \mid x_{t-1}^{A_{t-1}^{(n)}})
       - \log q_\phi(x_t^{(n)} \mid x_{t-1}^{A_{t-1}^{(n)}}, y_t) \).
   - Normalise: \( \tilde w_t^{(n)} = \mathrm{softmax}_n(\ell_t) \).
   - Resample when ESS falls below a threshold (say \(N/2\)), and update
     ancestors \(A_t\). Use systematic resampling for low variance.
3. Record \(\{x_t^{(n)}\}, \{A_t^{(n)}\}, \{\tilde w_t^{(n)}\}\).
4. **With gradients on**, recompute
   \( \log q_\phi(x_t^{(n)} \mid x_{t-1}^{A_{t-1}^{(n)}}, y_t) \) using the
   current network, and form

$$
\mathcal{L}
= -\frac{1}{B\,T\,N}\sum_{b,t,n}
  \tilde w_t^{(n),b}\,
  \log q_\phi \!\left( x_t^{(n),b} \,\middle|\,
    x_{t-1}^{A_{t-1}^{(n),b}, b}, y_t^{(b)} \right).
$$

5. Backprop and step the optimiser.

Vectorisation: steps 2 and 4 are most naturally written as loops over
\(t\). Within a time step, all \(N\) particles are processed in parallel.
Across a batch of trajectories, \(B \cdot N\) particles can be stacked.
Plan for `B = 8`, `N = 64`, `T = T_\text{dataset}` at most, but tune by
GPU memory.

Notes and pitfalls:
- When no observation is available at step \(t\) (sparse obs), the
  "proposal" used in the SMC inner loop should fall back to the prior,
  and that step contributes nothing to the loss (the proposal is not being
  trained for unobserved steps). Skip those steps when accumulating
  \( \mathcal{L} \).
- `log q_phi` on a reparameterised sample from \(q_\phi\) itself yields
  zero gradient on average, but this is not what happens here: we recompute
  with fresh parameters against stop-gradient \(x_t^{(n)}\) values, so the
  gradient is the weighted-MLE gradient and is non-trivial.
- The prior log-density \(\log p(x_t \mid x_{t-1})\) is already implemented
  in `models/bpf.py::compute_transition_log_prob`. Reuse it.
- The observation log-likelihood is available via
  `system.observation_log_prob` (or equivalent). Reuse it.

---

## 5. Module layout

New or modified files:

```
proposals/
  nasmc.py                # new: Lightning module(s) for the NASMC proposal
  nasmc_dataset.py        # new: subtrajectory dataset + data module
  train_nasmc.py          # new: CLI entrypoint
  architectures/__init__.py  # modify: add create_gaussian_head_network
  architectures/gaussian_head.py  # new: Gaussian head on top of existing backbones
models/
  proposals.py            # modify: add NASMCProposal wrapper class
  smc_utils.py            # new (optional): shared SMC primitives (resample,
                          # ESS, ancestor tracking), reused by trainer and by
                          # a future differentiable-SMC extension
run.py, eval.py           # modify: register "nasmc" dispatch key
```

No changes required in:
- `data.py`, `generate.py`: dataset format is unchanged.
- `models/bpf.py`, `models/apf.py`, …: the proposal is consumed through
  the existing ABC.
- Any of the existing proposal training code (RF, ShortcutFlow, MeanFlow).
  These remain first-class citizens.

---

## 6. `proposals/nasmc.py` — the Lightning module

Two pieces in one file (or split if preferred).

### 6.1 `GaussianProposal` — a minimal proposal model

Core class. Responsibilities:
- Forward pass: given `(x_prev, y_curr, mask, t_idx?)`, produce
  `(mu, log_sigma)`.
- `sample(x_prev, y_curr, …)`: reparameterised Gaussian draw; returns
  single-particle or batched depending on input shape to match the ABC.
- `log_prob(x_curr, x_prev, y_curr, …)`: analytic Gaussian log-density.
- `sample_and_log_prob(...)`: returns both; trivial here.

Sketch:

```python
class GaussianProposal(pl.LightningModule):
    def __init__(self, state_dim, obs_dim, architecture="mlp",
                 predict_delta=True, use_dynamics_mean=False,
                 ...):
        super().__init__()
        self.save_hyperparameters()
        self.net = create_gaussian_head_network(
            architecture=architecture,
            state_dim=state_dim, obs_dim=obs_dim,
            output_dim=2 * state_dim, ...,
        )
        # use_dynamics_mean implements the -f- variant; defaults to False.

    def _mean_std(self, x_prev, y_curr, mask, t=None):
        out = self.net(x_prev, y_curr, mask, t)           # (B, 2D)
        mu_raw, log_sigma = out.chunk(2, dim=-1)
        if self.hparams.predict_delta:
            mu = x_prev + mu_raw
        elif self.hparams.use_dynamics_mean:
            mu = system_integrate_one_step(x_prev) + mu_raw
        else:
            mu = mu_raw
        log_sigma = torch.clamp(log_sigma, min=-7., max=3.)
        return mu, log_sigma

    def sample(self, x_prev, y_curr, ..., n_samples=1):
        mu, log_sigma = self._mean_std(x_prev, y_curr, ...)
        eps = torch.randn_like(mu)
        return mu + log_sigma.exp() * eps

    def log_prob(self, x_curr, x_prev, y_curr, ...):
        mu, log_sigma = self._mean_std(x_prev, y_curr, ...)
        # Analytic Gaussian log-density (nats).
        return -0.5 * (((x_curr - mu) / log_sigma.exp()) ** 2
                       + 2 * log_sigma
                       + math.log(2 * math.pi)).sum(dim=-1)
```

Use `self.save_hyperparameters()` so `load_from_checkpoint` works, in
keeping with the RF pattern.

### 6.2 `NASMCTrainer` — training module

This class wraps `GaussianProposal` and implements the two training phases.

Responsibilities:
- Hold a `GaussianProposal` and the `DynamicalSystem` (needed for the
  transition log-prob and, optionally, the `-f-` variant).
- `training_step(batch)` dispatches on phase:
  - Phase 1 (pretrain): batch is `(x_prev, x_curr, y_curr)` from
    `RFTransitionDataset`. Loss is `-proposal.log_prob(x_curr, x_prev, y_curr).mean()`.
  - Phase 2 (refine): batch is `(x_1:T, y_1:T, obs_mask)` from
    `NASMCTrajectoryDataset` (see §7). Run the inner SMC loop as in §4.6
    and return the weighted-log-density loss.
- `configure_optimizers`: AdamW + ReduceLROnPlateau, matching RF.
- `on_train_epoch_start` / `on_train_epoch_end`: optional hooks to log ESS,
  fraction of collapsed trajectories, etc.

The SMC inner loop goes in its own method `_smc_forward(x0, y_seq, mask_seq)`
that returns detached `(particles, ancestors, log_weights)`, then a
`_weighted_log_density(particles, ancestors, y_seq, mask_seq)` method that
computes the trainable loss. Keep these private and unit-testable.

---

## 7. `proposals/nasmc_dataset.py` — subtrajectory dataset

Phase 1 can reuse `RFTransitionDataset` directly. Phase 2 needs full
trajectories.

### 7.1 `NASMCTrajectoryDataset`

One item = one full trajectory window. Fields returned per item:
- `trajectory`: `(T, state_dim)` ground-truth states (used only for
  logging / diagnostics; NASMC does not supervise on them during phase 2).
- `observations`: `(T, obs_dim)` observations.
- `obs_mask`: `(T,)` boolean: which time steps have observations.
- `initial_state`: `(state_dim,)` start of the trajectory, used as the
  ground-truth \(x_0\) for SMC initialisation. (Alternatively, initialise
  all \(N\) particles at \(x_0\); or, more realistic, sample from a
  broad prior and let a burn-in do the rest. Default to exact \(x_0\) for
  simplicity.)

The dataset loads from the same `data_scaled.h5`. Construction is a trivial
variant of `RFTransitionDataset` that iterates over trajectories rather
than per-step pairs.

Variable-length trajectories: not a concern here since every trajectory in
a given dataset has the same `len_trajectory`. If that assumption is ever
relaxed, add a length field and a collate that pads.

### 7.2 `NASMCDataModule`

Mirror of `RFDataModule`, but:
- `train_dataloader()` returns the appropriate dataset depending on the
  current phase (pretrain vs refine). One clean option is to expose two
  DataModules and have the trainer switch between them; another is a
  single DataModule with a `phase` attribute set externally.

Either is fine; the CLI in §8 chooses which one to instantiate.

---

## 8. `proposals/train_nasmc.py` — CLI entrypoint

Mirror the argument surface of `train_rf.py` where possible. Additions:

```
--phase {pretrain,refine,both}   # which phase(s) to run
--init_from_ckpt PATH            # optional, for refine-only
--smc_num_particles N            # particles per trajectory during refine
--smc_resample_threshold FRAC    # ESS threshold as fraction of N
--warmup_epochs K                # epochs in refine that still use bootstrap
                                 # proposal to generate particles
--architecture {mlp,resnet1d}
--predict_delta                  # bool
--use_dynamics_mean              # bool, -f- variant
```

Flow of the script:
1. Load dataset config, scalers, system.
2. Build `NASMCTrainer` (either from scratch or from `--init_from_ckpt`).
3. If `phase in {pretrain, both}`: build `RFDataModule`, run
   `trainer.fit(...)`. Save a checkpoint tagged `pretrain`.
4. If `phase in {refine, both}`: build `NASMCDataModule`, set the trainer
   to phase 2, run `trainer.fit(...)`. Save a checkpoint tagged `final`.

All logging goes through `WandbLogger` with the same project as RF runs so
the two sit side by side.

---

## 9. `models/proposals.py` — `NASMCProposal` wrapper

Mirror of `RectifiedFlowProposal`:

```python
class NASMCProposal(ProposalDistribution):
    def __init__(self, checkpoint_path, system, device,
                 obs_mean, obs_std, obs_components=None, ...):
        self.model = GaussianProposal.load_from_checkpoint(checkpoint_path)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.system = system
        self.obs_mean = ...
        self.obs_std = ...

    def sample(self, x_prev, y_curr, dt, t=None, static_params=None):
        x_prev_s = self.system.preprocess(x_prev)
        y_s = (y_curr - self.obs_mean) / self.obs_std if y_curr is not None else None
        with torch.no_grad():
            x_curr_s = self.model.sample(x_prev_s, y_s, t=t)
        return self.system.postprocess(x_curr_s)

    def log_prob(self, x_curr, x_prev, y_curr, dt, t=None, static_params=None):
        x_prev_s = self.system.preprocess(x_prev)
        x_curr_s = self.system.preprocess(x_curr)
        y_s = (y_curr - self.obs_mean) / self.obs_std if y_curr is not None else None
        with torch.no_grad():
            return self.model.log_prob(x_curr_s, x_prev_s, y_s, t=t)
```

Same caveat as the RF wrapper: `log_prob` returns a density in *scaled*
space, and we keep the same treatment (no Jacobian correction) so the
importance weights are comparable to the RF baseline. This is called out
explicitly in §10.

`sample` and `log_prob` accept single-particle or batched inputs. The
Gaussian head is already batched; only the shape plumbing needs care.

Finally, register the wrapper in whatever dispatch table `run.py` uses
(proposal key `"nasmc"`).

---

## 10. Dispatch and eval integration

Expected changes:

- In `run.py` / `eval.py`, wherever there is a mapping from proposal short
  key to wrapper class, add `"nasmc" -> NASMCProposal`.
- The CLI flag that currently picks `--proposal {rf,lrf,transition,...}`
  should accept `nasmc` without other changes. The checkpoint-path flag
  already exists for RF; reuse it.
- No other filter-side changes are required: `models/bpf.py` already
  operates through the ABC, and `models/apf.py`, `enkf.py`, `ensf.py` all
  use the same interface.

---

## 11. Evaluation protocol: NASMC vs RF

To compare methods cleanly, hold all of the following fixed:

1. **Dataset.** Same `datasets/<name>/{config.yaml, data.h5, data_scaled.h5}`
   for both methods. In particular, same `obs_components`, same
   `obs_noise_std`, same `obs_frequency`, same `obs_nonlinearity`.
2. **Backbone architecture.** Same `architecture` (`mlp` or `resnet1d`) and
   same hidden sizes. The only difference is the output head (velocity
   field vs Gaussian mean/log-std).
3. **Training budget.** Same number of gradient steps or, more honestly,
   same wall clock. NASMC phase 1 (pretrain) is comparable to RF in cost;
   NASMC phase 2 adds SMC overhead. Report both with and without phase 2
   so the reader can distinguish "Gaussian baseline trained by MLE" from
   "Gaussian baseline trained by NASMC refinement."
4. **Filter.** Run `eval.py` with `BootstrapParticleFilterUnbatched` (or
   whichever is the standard table entry) for both proposals, same
   `n_particles`, same seeds, same test split.

Metrics to report (already produced by `eval.py`):
- Filtering RMSE vs ground truth.
- ESS trajectory (mean, per-step distribution).
- Log-marginal-likelihood estimate.
- Optional: CRPS for probabilistic evaluation.

Systems to run:
- Double-well (smoke test; very cheap; both methods should work).
- Lorenz-63 (low-dim, chaotic; both methods expected to work).
- Lorenz-96 at \(D=40\) (the main comparison system).
- KS (stretch goal; harder for a diagonal Gaussian because of strong
  spatial correlations).

Ablations worth including in the first results table:
- NASMC phase 1 only (= Gaussian MLE baseline).
- NASMC phase 1 + phase 2 (= full NASMC).
- NASMC with `predict_delta` vs `use_dynamics_mean` (`-f-`).
- RF baseline (existing).
- Bootstrap PF (no learned proposal).

Expectation based on the paper: NASMC phase 2 should improve ESS and LML
over NASMC phase 1 at matched compute, with a smaller RMSE gap. The
question for our setting is whether the multimodality of Lorenz-96 /
KS posteriors under sparse observations is well-captured by a single
diagonal Gaussian. If not, MDN / RNN extensions become necessary (§12).

---

## 12. Open design choices and v2 extensions

The following are deliberately left out of v1 but should be easy to add.

**MDN head (`-MD-` variant).** Replace the Gaussian head with a mixture.
Output `(K, 2D + 1)` per particle: mixing logits plus per-component
\((\mu, \log\sigma)\). `sample` draws a component index, then a Gaussian;
`log_prob` is a log-sum-exp. Everything else (training loop, wrapper) is
unchanged.

**RNN over \(x_{1:t-1}\), \(y_{1:t}\).** Extend the `ProposalDistribution`
ABC with an optional `state` argument (the RNN hidden state), and add a
parallel `reset()` method. Filters that do not know about the state pass
`None`. This is a larger change and best done in a separate PR.

**`-f-` variant switched on.** Already sketched in §4.3 and wired as a
config flag. Benchmark once phase-2 NASMC is stable.

**Differentiable SMC.** The current plan uses stop-gradient particles, as
in the paper. Fully differentiable SMC (with reparameterised resampling or
relaxed resampling) is a research direction that can be layered on top
without changing the wrapper. Not recommended for v1.

**Score-function baselines.** A subtraction-of-mean baseline on the
per-particle log-density term is a standard variance-reduction trick for
the NASMC gradient. Easy to add; worth trying if training is noisy.

**Observation-window conditioning.** The proposal could be given
\(y_{t-k:t+k}\) rather than \(y_t\) alone. This is orthogonal to NASMC vs
RF and should be a shared option across both proposal families if we go
there, to keep comparisons fair.

---

## 13. Risks and mitigations

1. **SMC collapse in phase 2.** If \(q_\phi\) degrades and ESS collapses
   to 1 for every trajectory, the gradient signal vanishes. Mitigation:
   the pretrain phase; warm-up with bootstrap particles; larger
   \(N_\text{train}\); ESS-triggered early stopping that reverts to a
   previous checkpoint.
2. **Diagonal Gaussian too restrictive for high-dim spatial correlations.**
   Lorenz-96 and KS posteriors can have non-trivial spatial correlations.
   If NASMC underperforms RF on these systems, try MDN first, then an
   RNN-parametrised proposal. This is expected and is the primary
   research contribution NASMC is built for anyway.
3. **Scaling / Jacobian caveat.** Both RF and NASMC return log-densities
   in scaled space. This is a known gotcha in the existing codebase and
   the wrapper replicates the same treatment so comparisons remain fair.
   If at some point a rigorous Jacobian correction is added, add it to
   both wrappers simultaneously.
4. **Compute cost of phase 2.** One SMC pass per trajectory per gradient
   step is expensive. Mitigations: small \(N_\text{train}\) (64 is usually
   fine), truncate trajectories to length 256 rather than using the full
   1000 during training, use fewer particles early and ramp up.
5. **Reproducibility with resampling.** Systematic resampling + a fixed
   seed per trajectory per epoch keeps runs reproducible. Budget an
   afternoon for making sure this is deterministic.

---

## 14. Suggested milestones

Each milestone is a self-contained PR-sized unit.

1. **Scaffolding + GaussianProposal forward pass.** Add `gaussian_head.py`,
   `create_gaussian_head_network`, `GaussianProposal` skeleton with
   `sample` / `log_prob`. Unit tests on shape correctness.
2. **Phase 1 (pretrain) end-to-end.** Plug `GaussianProposal` into a
   training loop that reuses `RFDataModule`, train on double-well, verify
   that phase 1 gets to a reasonable validation NLL. No SMC yet.
3. **NASMC wrapper and `eval.py` dispatch.** Add `NASMCProposal`, register
   `"nasmc"` in `run.py` / `eval.py`. Run bootstrap PF with a phase-1
   checkpoint on double-well and Lorenz-63; compare RMSE/ESS to RF and to
   the bootstrap PF.
4. **Subtrajectory dataset.** Add `NASMCTrajectoryDataset` +
   `NASMCDataModule`. Unit tests on shape, masking, train/val/test splits.
5. **SMC inner loop and phase-2 loss.** Add `_smc_forward` and
   `_weighted_log_density` to `NASMCTrainer`. Start with no resampling
   (SIS only) for debugging; then add systematic resampling.
6. **Phase 2 end-to-end on double-well.** Confirm training is stable and
   that phase-2 NASMC improves ESS and/or LML over phase-1 on a small
   system. Fix any bugs surfaced here before moving to Lorenz.
7. **Scaling up to Lorenz-96.** ResNet1D backbone; benchmark versus RF
   with the same architecture and budget.
8. **Ablations and writeup.** Table comparing RF, NASMC-phase1,
   NASMC-phase2, bootstrap PF across systems; brief analysis of where each
   method wins.

Optional follow-ups: MDN head, RNN conditioning, `-f-` variant
benchmarking.