# Plan: Add a Paige-Wood Inference-Network Proposal Alongside the RF Proposal

> **Reference paper.** Paige & Wood (2016), *Inference Networks for Sequential
> Monte Carlo in Graphical Models*. ICML. arXiv and PMLR:
> https://proceedings.mlr.press/v48/paige16.pdf.
>
> **Reference code.** https://github.com/tbrx/compiled-inference (conditional
> MADE / RNADE in PyTorch; the FHMM example is the closest analog to a
> filtering setup).
>
> **Context.** We already have a learned rectified-flow (RF) proposal that
> plugs into the particle filter via the `ProposalDistribution` ABC. This
> document specifies how to add a second learned proposal family
> (Paige-Wood-style "inference network", hereafter **IN**) using the same
> training data, the same `DataModule`, the same PF evaluation harness, and
> the same `ProposalDistribution` contract, so that both proposals can be
> benchmarked head-to-head on the existing datasets (double-well, L63,
> L96, KS, optionally Kolmogorov).

---

## 1. Scope and goal

The goal is to produce an implementation of the Paige-Wood inference-network
proposal that is (a) faithful to the paper's method as specialized to a
state-space model, and (b) drop-in interoperable with every piece of the
existing codebase (`models/bpf.py`, `models/apf.py`, `ensf.py`, `eval.py`,
`run.py`, the `datasets/` directory, the `RFDataModule`, the shared velocity-
net architectures). Success means:

1. `python proposals/train_inference_network.py --data_dir datasets/<name>`
   trains an IN proposal and writes a Lightning checkpoint to the same
   kind of location that `train_rf.py` uses.
2. `eval.py` can be invoked with `--proposal inn --checkpoint <path>` and
   produces RMSE / ESS / log-likelihood numbers on exactly the test
   trajectories the RF proposal is evaluated on.
3. A side-by-side results table on the same datasets can be produced with
   minimal additional plumbing.

Non-goals: reproducing the paper's non-filtering examples (polynomial
regression, Poisson hierarchical model). Those were the paper's main use
cases and are orthogonal to our filtering work.

---

## 2. What the method becomes in a state-space setting

The paper's framework is general (any DAG). Specializing to our model
$x_t \mid x_{t-1} \sim p_\theta$, $y_t \mid x_t \sim g_\theta$ gives a very
clean picture.

### 2.1. Inverse factorization

For a filtering task, the "inverse model" in Paige-Wood's construction is
exactly the per-step optimal proposal,

$$
\tilde p(x_t \mid x_{t-1}, y_t)
\;\propto\; p(x_t \mid x_{t-1})\, g(y_t \mid x_t).
$$

Head-to-head structure: for vector-valued $x_t \in \mathbb{R}^D$, all
components of $x_t$ are coupled in the posterior because they jointly
explain the observation $y_t$ and the transition density. So the CDE must
represent a **joint** density over $x_t$, not a dimensionwise factorization.
(The paper flags this explicitly in Section 3.3 and is why their FHMM uses
a MADE-style joint autoregressive head rather than independent Bernoullis.)

### 2.2. Training objective

The Paige-Wood objective is the expected forward KL under the model's joint,

$$
\mathcal{J}(\eta) \;=\;
\mathbb{E}_{p(x_{t-1}, x_t, y_t)}\!\left[-\log q_\eta(x_t \mid x_{t-1}, y_t)\right]
\;+\;\text{const.}
$$

Two things to notice:

- This is just **negative log-likelihood of $x_t$ under a conditional
  density model**, with $(x_{t-1}, y_t)$ as conditioning. It is supervised
  maximum-likelihood training. No particle filter in the inner loop, no
  score function, no ELBO.
- The expectation is over the *model's* joint, which is exactly what the
  simulator we use to build `data.h5` provides. Every trajectory in
  `data_scaled.h5` gives us a stream of $(x_{t-1}, x_t, y_t)$ triples
  sampled from this joint.

### 2.3. How this differs from RF training

| Aspect | RF proposal | IN proposal (this plan) |
|---|---|---|
| Density family | Continuous normalizing flow (Gaussian base + learned velocity) | Conditional density estimator (e.g., joint diagonal MoG or RNADE with MoG heads) |
| Training loss | Flow-matching MSE of velocity along straight-line interpolant | Negative log-likelihood of $x_t$ given $(x_{t-1}, y_t)$ |
| Sampling cost | Euler integration over $K$ steps of $v_\theta$ | One forward pass through the network, then closed-form sampling from the parametric head |
| `log_prob` cost | Backward ODE integration + divergence estimator (exact $O(D^2)$ or Hutchinson $O(D)$) | Closed form, single forward pass |
| Expressivity | Very flexible (universal as $K \to \infty$) | Limited by density family (e.g., a MoG with $K$ components) |
| Known failure modes | Trace-estimator variance, integration-step sensitivity | Mode collapse, variance collapse in MDN outputs |

This contrast matters for the experimental story: IN should be **much
faster** at inference than RF, but **less expressive**. A head-to-head
comparison on the same metrics (RMSE / CRPS / ESS / wall-clock) is the
natural thing to report.

---

## 3. Contract for dropping IN into our codebase

The `ProposalDistribution` ABC in `models/proposals.py` already defines the
interface any learned proposal must satisfy. Re-read §2 of the tech
overview; this section just lists the IN-specific contract.

**Training-time contract (new Lightning module):**

- Trains from `(x_prev, x_curr, y_curr, time_idx)` batches produced by the
  existing `RFTransitionDataset` / `RFDataModule`. No new dataset class.
- Uses scaled-space inputs (reads `data_scaled.h5`) exactly like RF.
- Saves all hyperparameters via `self.save_hyperparameters()` so
  `load_from_checkpoint` works.
- Exposes `sample(x_prev, y_curr, dt=None, t=None) -> x_curr` and
  `log_prob(x_curr, x_prev, y_curr, dt=None, t=None) -> log q` on the
  Lightning module. `dt` is accepted for compatibility with the ABC but
  unused internally.

**Inference-time contract (new wrapper in `models/proposals.py`):**

- Mirrors `RectifiedFlowProposal` line-for-line. Accepts a checkpoint
  path, `system`, `obs_scalers`, `obs_components`, device, and the optional
  `time_step` flag.
- `sample` and `log_prob` do the same scaled/physical conversions that the
  RF wrapper does (preprocess `x_prev`, `x_curr`; scale `y_curr`; invoke
  underlying module; postprocess sample).
- `log_prob` returns the scaled-space log-density. We keep the same
  convention the RF wrapper uses (no Jacobian correction in either
  wrapper) so the BPF importance weights are apples-to-apples.

**Registration:**

- Add a new short key (`"inn"`) to whatever CLI dispatches proposals in
  `run.py` and `eval.py`. Existing keys like `"rf"`, `"lrf"`, `"transition"`
  are the pattern to follow.

Everything else (the BPF, APF, ensemble filters) stays untouched.

---

## 4. Architectural decisions

This is the most interesting part of the design. There are three
questions: which backbone to use, which CDE head to attach, and how to
handle sparse-in-time observations.

### 4.1. Backbone: reuse RF architectures

`proposals/architectures/` already has:

- `mlp.py::MLPVelocityNetwork` (flat MLP; good for L63, double-well)
- `resnet1d.py::ResNet1DVelocityNetwork` (circular 1D conv + AdaLN; good
  for L96, KS)

Both currently output a velocity vector in $\mathbb{R}^D$. We want a
backbone that instead outputs a **feature vector $h \in \mathbb{R}^H$** which
the CDE head then turns into distribution parameters.

**Plan:** introduce a shared notion of a "feature backbone" by adding an
alternate factory path `create_feature_backbone(…)` in
`proposals/architectures/__init__.py` that wraps the same MLP / ResNet1D
code but removes the final linear projection to $D$ and returns the last
hidden state. To keep blast radius small:

- Don't touch `MLPVelocityNetwork` / `ResNet1DVelocityNetwork` at all.
- Add thin subclasses (`MLPFeatureBackbone`, `ResNet1DFeatureBackbone`) in
  the same files that reuse their `__init__` and `forward` but strip the
  output head.
- The IN module composes `backbone -> CDE head`; the RF module continues
  to compose `backbone -> velocity head` as it does today.

Also critically: **drop the flow-time input `s`** from the backbone when
used for IN. The RF velocity-net takes $(x_s, s, x_{t-1}, y_t, t_\text{traj})$;
IN takes $(x_{t-1}, y_t, t_\text{traj})$ only. Cleanest implementation is to
pass `s=None` through the same backbone constructor and have the backbone's
time-embedding path be optional (it already is if `use_time_step=False`;
just add a similar flag for the flow time).

If stripping `s` is too invasive in practice, an acceptable alternative is
to pass a constant $s = 0$ (or $s = 1$) and accept the wasted parameters
in the time-embedding MLP. Either choice is fine for the comparison; just
document it.

### 4.2. CDE head: four options, pick one as the default

The CDE head takes the backbone feature $h$ and returns parameters of
$q_\eta(x_t \mid h)$. Four reasonable choices, in increasing expressivity:

1. **Factorized diagonal Gaussian (MDN-1).**
   $q(x_t \mid h) = \prod_i \mathcal{N}(x_{t,i}; \mu_i(h), \sigma_i(h)^2)$.
   Parameters: $2D$. Fast, trivial. Closest analog of a learned
   "Gaussian proposal" in classical DA. Known to be weak when the
   posterior is multimodal or strongly correlated across dims.

2. **Factorized mixture of Gaussians per dim (MDN-K).**
   $q(x_t \mid h) = \prod_i \sum_{k=1}^K \alpha_{i,k}
       \mathcal{N}(x_{t,i}; \mu_{i,k}, \sigma_{i,k}^2)$.
   Parameters: $3KD$. Captures per-dim multimodality but no cross-dim
   correlation beyond what the conditioning already encodes.

3. **Joint mixture of diagonal Gaussians (jointMDN-K).** Each component
   is a single diagonal $D$-dim Gaussian, mixture over $K$ components:
   $q(x_t \mid h) = \sum_{k=1}^K \alpha_k
       \mathcal{N}(x_t; \mu_k(h), \mathrm{diag}(\sigma_k(h)^2))$.
   Parameters: $K(2D+1)$. Captures global multimodality and has mild
   inter-dim correlation via the mixture assignment. This is the sweet
   spot for first-pass experiments and matches the spirit of the paper's
   output head while being far simpler than MADE.

4. **Conditional RNADE (faithful Paige-Wood).** Autoregressive model
   over $x_{t,1:D}$ with each $x_{t,i} \mid x_{t,<i}, x_{t-1}, y_t$ a MoG.
   This is exactly what the paper's `learn_smc_proposals.cde` module
   implements, extended to condition on $(x_{t-1}, y_t)$ through
   additional inputs.
   Parameters: roughly $O(D \cdot (H + 3K))$ for $H$ hidden units. This
   is the faithful reproduction of the paper, and the option with the
   most expressivity.

**Recommendation for the default:** **Option 3 (jointMDN-K)** as the
primary configuration, because:

- It's a faithful *continuous-state* analog of the paper's joint density
  model without requiring a conditional RNADE implementation up front.
- It has cheap closed-form sampling and `log_prob`, which is the main
  point of this method vs. RF.
- It covers multimodality, which is exactly where the bootstrap / Gaussian
  proposal is known to fail (per Del Moral & Murray, 2015, which the
  paper cites).

**Implement all four as plug-in CDE heads** in
`proposals/architectures/cde_heads.py` (see §5). Options 1 and 2 are
trivial once 3 is in place; option 4 is a second-pass follow-up.

### 4.3. Conditioning layout

The conditioning inputs are $(x_{t-1}, y_t, t_\text{traj})$, with
$y_t$ possibly masked (sparse observations).

For the **MLP backbone** (L63, double-well):
- Flat-concatenate $[x_{t-1}, y_\text{dense}, \text{mask}, \text{time\_emb}]$,
  identical to how `MLPVelocityNetwork` packs its inputs minus $x_s$ and
  the flow time $s$.

For the **ResNet1D backbone** (L96, KS):
- Treat $x_{t-1}$ as the input "image" along the spatial axis, and use
  AdaLN / FiLM conditioning from $y_t$ and $t_\text{traj}$, exactly as
  `ResNet1DVelocityNetwork` already does. The only change is that the
  output is a pooled feature vector rather than a per-location velocity.

### 4.4. Handling sparse-in-time observations

Current user setup: observations are **spatially dense** (observed across
all state dimensions) but **temporally sparse** (not every step has an
observation). The BPF already handles this: when `y_curr is None` at a
given step, it falls back to the transition prior (see §3 of the tech
overview).

Two options for IN at training time:

(a) **Train on only observed steps** (filter the training pairs to those
    with `obs_mask[t] = True`). Simplest, matches what RF does in
    practice because `RFTransitionDataset` already yields only those
    pairs when `use_observations=True`.

(b) **Train jointly on observed and unobserved steps**, with a binary
    "observation present" token appended to the input. The network then
    learns to condition on $y$ when present and ignore it when not. This
    is useful if we ever want the *same* trained network to handle both
    observed and unobserved steps without falling back to a separate
    transition prior. This is also closer in spirit to the
    `cond_dropout` trick used by RF.

Start with (a). Option (b) is straightforward to add later by generalizing
`RFTransitionDataset` or, more surgically, by enabling a classifier-free-
guidance-style dropout of $y$ during training (just set $y$ to a fixed
"null" value with some probability and let the mask flag it). The latter
does not require a new dataset.

---

## 5. Concrete file plan

Everything below is incremental. Nothing existing changes except the
short list of edits in §5.6.

### 5.1. `proposals/inference_network.py` (new)

A `LightningModule` called `InferenceNetworkProposal`. Sketch:

```python
class InferenceNetworkProposal(pl.LightningModule):
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        obs_indices: list[int] | None,
        architecture: str,            # "mlp" | "resnet1d"
        cde_head: str,                # "gaussian" | "mdn_k" | "joint_mog" | "rnade"
        hidden_dim: int,
        num_layers: int,
        num_mixture_components: int,  # K for MoG heads; ignored for "gaussian"
        use_time_step: bool,
        use_observations: bool,
        learning_rate: float,
        weight_decay: float = 1e-5,
        min_sigma: float = 1e-3,       # clamp for numerical stability
        ...
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = create_feature_backbone(
            architecture=architecture,
            state_dim=state_dim,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_time_step=use_time_step,
            use_observations=use_observations,
            obs_indices=obs_indices,
        )
        self.head = create_cde_head(
            kind=cde_head,
            state_dim=state_dim,
            feature_dim=self.backbone.feature_dim,
            num_mixture_components=num_mixture_components,
            min_sigma=min_sigma,
        )
```

Methods:

- `compute_loss(x_prev, x_curr, y_curr, t) -> loss`: single forward pass
  of backbone, single forward pass of head, returns `-log q(x_curr | h)`
  averaged over the batch.
- `training_step`, `validation_step`: call `compute_loss`, log
  `train_nll` and `val_nll`. `val_loss` (= `val_nll`) is what the
  existing `ModelCheckpoint` monitors.
- `configure_optimizers`: `AdamW(lr, weight_decay)` + `ReduceLROnPlateau`
  on `val_loss`. Identical to RF.
- `sample(x_prev, y_curr=None, dt=None, t=None)`: backbone → head →
  sample once from the parametric family.
- `log_prob(x_curr, x_prev, y_curr=None, dt=None, t=None)`: backbone →
  head → evaluate log-density of `x_curr`.
- `sample_and_log_prob(...)`: optional convenience method, identical in
  semantics to the RF one.

Numerical details (non-negotiable):

- Apply `softplus` + clamp to a floor `min_sigma` on all predicted
  standard deviations. MDN training is notorious for $\sigma \to 0$
  collapse; this single line prevents it.
- Apply `log_softmax` (not raw `softmax` + `log`) on mixture weights to
  avoid underflow.
- Use `torch.logsumexp` over the $K$ component log-densities rather than
  `log(sum(exp(...)))`.

### 5.2. `proposals/architectures/cde_heads.py` (new)

A small file with four classes and a factory:

```python
class GaussianHead(nn.Module): ...           # Option 1: one diag Gaussian
class MDNHead(nn.Module): ...                # Option 2: K MoG per dim
class JointMoGHead(nn.Module): ...           # Option 3 (DEFAULT)
class RNADEHead(nn.Module): ...              # Option 4 (stretch)

def create_cde_head(kind: str, ...):
    ...
```

Each head exposes:

- `forward(features) -> params` returns a dict of distribution parameters.
- `log_prob(x, params) -> (B,)` in nats.
- `sample(params, n=1) -> x` of shape `(B, n, D)` (or `(B, D)` for `n=1`).

The `JointMoGHead` is the most important. Sketch:

```python
class JointMoGHead(nn.Module):
    def __init__(self, feature_dim, state_dim, K, min_sigma=1e-3):
        super().__init__()
        self.K = K; self.D = state_dim; self.min_sigma = min_sigma
        self.proj = nn.Linear(feature_dim, K * (2 * state_dim + 1))

    def forward(self, h):
        raw = self.proj(h)                                  # (B, K*(2D+1))
        raw = raw.view(h.shape[0], self.K, 2 * self.D + 1)
        log_w = F.log_softmax(raw[..., 0], dim=-1)          # (B, K)
        mu    = raw[..., 1:1 + self.D]                      # (B, K, D)
        log_s = raw[..., 1 + self.D:]                       # (B, K, D)
        sigma = F.softplus(log_s).clamp(min=self.min_sigma) # (B, K, D)
        return {"log_w": log_w, "mu": mu, "sigma": sigma}

    def log_prob(self, x, params):
        # x: (B, D). Broadcast over K.
        diff = (x.unsqueeze(1) - params["mu"]) / params["sigma"]  # (B, K, D)
        log_comp = (
            -0.5 * (diff ** 2).sum(-1)
            - params["sigma"].log().sum(-1)
            - 0.5 * self.D * math.log(2 * math.pi)
        )                                                         # (B, K)
        return torch.logsumexp(params["log_w"] + log_comp, dim=-1)

    def sample(self, params, n=1):
        # Gumbel-max on log_w, then reparameterized Gaussian sample.
        ...
```

The `RNADEHead` (option 4) is more involved; treat it as a follow-up.

### 5.3. `proposals/train_inference_network.py` (new)

A thin clone of `train_rf.py` with the same CLI surface:

- `--data_dir`, `--state_dim`, `--obs_dim`, `--use_observations`,
  `--obs_components`, `--architecture`, `--hidden_dim`, `--num_layers`,
  `--use_time_step`, `--learning_rate`, `--batch_size`, `--max_epochs`,
  `--seed`, etc.
- IN-specific flags: `--cde_head`, `--num_mixture_components`,
  `--min_sigma`.
- Builds `RFDataModule` exactly like `train_rf.py` does.
- Builds `InferenceNetworkProposal` with the parsed hparams.
- Same callbacks (`ModelCheckpoint` on `val_loss`, `EarlyStopping`,
  `LearningRateMonitor`, optional periodic checkpoint).
- Same `WandbLogger`.
- Optional trailing `run_proposal_eval(...)` call for autoregressive
  sanity-check evaluation (this works out of the box: `eval_proposal.py`
  only calls `model.sample(...)` and `model.log_prob(...)`, which we
  implement).

### 5.4. `models/proposals.py::InferenceNetworkProposal` wrapper (new)

Parallel to `RectifiedFlowProposal`. Sketch:

```python
class InferenceNetworkProposal(ProposalDistribution):
    def __init__(self, checkpoint_path, system, obs_mean, obs_std,
                 obs_components=None, device="cuda", use_time_step=False):
        from proposals.inference_network import InferenceNetworkProposal as INL
        self.model = INL.load_from_checkpoint(checkpoint_path).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.system = system
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.obs_components = obs_components
        self.use_time_step = use_time_step

    def sample(self, x_prev, y_curr, dt, t=None, static_params=None):
        x_prev_s = self.system.preprocess(x_prev)
        y_s = None if y_curr is None else (y_curr - self.obs_mean) / self.obs_std
        t_arg = t if self.use_time_step else None
        x_curr_s = self.model.sample(x_prev_s, y_s, dt=dt, t=t_arg)
        return self.system.postprocess(x_curr_s)

    def log_prob(self, x_curr, x_prev, y_curr, dt, t=None, static_params=None):
        x_prev_s = self.system.preprocess(x_prev)
        x_curr_s = self.system.preprocess(x_curr)
        y_s = None if y_curr is None else (y_curr - self.obs_mean) / self.obs_std
        t_arg = t if self.use_time_step else None
        return self.model.log_prob(x_curr_s, x_prev_s, y_s, dt=dt, t=t_arg)
```

Two important invariants to preserve:

1. **Input shape flexibility.** The BPF loops per-particle with shape
   `(D,)`. Inside the wrapper, add a `.unsqueeze(0)` and matching
   `.squeeze(0)` (or use `torch.atleast_2d`) so the underlying Lightning
   module always operates on a batch dim. Identical to what the RF
   wrapper does.
2. **Device placement.** `x_prev`, `y_curr` can come in on CPU; move to
   `self.device` before preprocessing.

### 5.5. `models/proposals.py::InferenceNetworkProposal` registration

Add to whatever dispatcher builds the proposal object in `run.py` /
`eval.py`. The existing keys (`"rf"`, `"lrf"`, `"transition"`, etc.) give
the pattern; add `"inn"`.

### 5.6. Minimal edits to existing files

- `proposals/architectures/__init__.py`: add `create_feature_backbone`
  (returns backbone-only variant) alongside the existing
  `create_velocity_network`.
- `proposals/architectures/mlp.py`: add `MLPFeatureBackbone`
  (inherits/copies most of `MLPVelocityNetwork` sans the final
  projection and the `s` input).
- `proposals/architectures/resnet1d.py`: add `ResNet1DFeatureBackbone`
  analogously.
- `models/proposals.py`: add `InferenceNetworkProposal` wrapper class.
- `run.py` and `eval.py`: add the `"inn"` proposal dispatch branch.

No edits to `data.py`, `generate.py`, `proposals/rf_dataset.py`,
`proposals/rectified_flow.py`, or any of the filter classes.

---

## 6. Training objective and loop specifics

This section nails down details that are easy to get wrong.

**Per-batch loss** (for any CDE head):

```
h = backbone(x_prev, y_curr, mask, t_traj)        # (B, feature_dim)
params = head(h)                                  # dict of tensors
nll = -head.log_prob(x_curr, params).mean()       # scalar
loss = nll
```

Everything is in scaled space (the DataModule reads `data_scaled.h5`).

**Optional auxiliary losses (not recommended for v1):**

- Observation-consistency loss analogous to RF's `obs_consistency_weight`:
  take the predicted mean(s) and push the implied observation toward
  $y_t$. Adds complexity for unclear benefit given the main NLL already
  conditions on $y_t$; leave this out unless baseline NLL training
  underperforms.

**Conditioning dropout (cheap, worth including):**

- With probability `cond_dropout` (default $0$, tune up to $0.1$),
  replace $y_t$ with a zero vector and flip the mask off during
  training. This gives us a single network that can be used both for
  observed and unobserved steps (§4.4, option b) without retraining.

**Batch sizes and epochs:** match `train_rf.py`'s defaults for the first
run, then tune. Expect IN to require *fewer* epochs than RF because the
NLL objective has lower variance than flow matching.

**Validation metric:** `val_nll` on the held-out split of
`RFTransitionDataset.val`. Early stopping on `val_nll`.

**Extra metrics worth logging (cheap):**

- `val_mean_l2`: Euclidean distance between the predicted conditional
  *mean* (argmax-weighted mixture mean, or just the mean of the density)
  and the ground-truth $x_t$. This is comparable to RF's one-step
  sample-based validation error.
- `val_mixture_entropy`: per-example entropy of the mixture weights.
  Sanity-check: if it collapses to near $0$ across the validation set,
  the mixture is effectively using one component.
- `val_avg_sigma`: mean predicted $\sigma$. Sanity-check for the
  variance-collapse failure mode.

---

## 7. Proposal ABC implementation details

Both `sample` and `log_prob` must match the BPF's expectations (re-read
§3 of the tech overview).

**`sample(x_prev, y_curr, dt, t=None)`.** Shapes:
- Input `x_prev`: `(D,)` or `(N, D)`.
- Input `y_curr`: `(obs_dim,)` or `(N, obs_dim)` or `None`.
- Output: same batch shape as `x_prev`, with last dim `D`.

Implementation: run backbone once, then draw from the head. For MoG
heads, use the standard two-step sampling: (i) `k ~ Categorical(exp(log_w))`,
(ii) $x \sim \mathcal{N}(\mu_k, \mathrm{diag}(\sigma_k^2))$ using
reparameterization. Gumbel-max on `log_w` is fine and marginally faster.

**`log_prob(x_curr, x_prev, y_curr, dt, t=None)`.** Shapes mirror the
sampling case. Returns a scalar or `(N,)`. Internally evaluate the mixture
log-density via `torch.logsumexp` on per-component log-densities.

**Scaled vs physical space.** Same convention as `RectifiedFlowProposal`:
the wrapper preprocesses/postprocesses states and scales observations, but
the returned log-density is *in scaled space*. The BPF treats both the
transition log-density (physical space) and the proposal log-density
(scaled space) as if they live in the same space; this introduces a
missing Jacobian term for both RF and IN. We keep it that way for
comparability. If we ever want to fix this, fix it in both wrappers at
once.

**Time-step conditioning.** If `use_time_step=True`, pass the normalized
trajectory-time $t$ through to the backbone exactly the way
`RFProposal` does. For IN this is a single extra input channel to the
backbone, unchanged from the velocity-net code path.

---

## 8. Evaluation plan

Goal: produce a side-by-side comparison of RF and IN on the same
datasets, same filters, same metrics.

### 8.1. Datasets

Run on every dataset we currently use for RF. At minimum:

- `double_well_...` (sanity)
- `lorenz63_...`
- `lorenz96_...` (primary interest)
- `kuramoto_sivashinsky_...` (spatial backbone stress test)

For each: train RF (already exists) and IN (new) with identical
`obs_components`, `obs_frequency`, `obs_noise_std`, `process_noise_std`.

### 8.2. Autoregressive proposal-only evaluation

Use `proposals/eval_proposal.py` (already written) to compare:

- RMSE of one-step samples.
- CRPS over the test trajectories.
- Forecast spread vs. ensemble spread.

This checks the learned conditional in isolation, independent of the
particle filter. IN should at least match RF here on the metrics where
the two density families are comparable; where it underperforms,
that points to expressivity limits of the head.

### 8.3. Particle-filter evaluation (the main result)

Use `eval.py` with the BPF and with each proposal. Report:

- **Filtering RMSE** (state reconstruction) vs. number of particles $N$
  at $N \in \{32, 64, 128, 256, 512\}$.
- **Filtering CRPS** (probabilistic reconstruction) at the same $N$.
- **Effective sample size** trace over time, at fixed $N$.
- **Log marginal likelihood** estimate.
- **Wall-clock time per filter step**, broken down into (i) proposal
  `sample` cost, (ii) proposal `log_prob` cost, (iii) rest of the filter.
  IN should win here substantially; numbers around an order of magnitude
  are plausible given RF's Euler + divergence integration.

If we also run APF or EnSF with the proposals, those should work out of
the box via the same ABC.

### 8.4. Small sanity experiments

- Replace the learned $q$ with $q = p(x_t \mid x_{t-1})$ (the BPF
  fallback). This is the proposal-free baseline. Both RF and IN should
  beat it, especially at highly informative observations.
- Compare against the *optimal* Gaussian proposal computed analytically
  for the linear-Gaussian system (`data.py::LinearGaussian`). IN with a
  single-Gaussian head should approach this as the network trains.

---

## 9. Ablations

A few ablations that will help isolate where IN wins and loses:

1. **CDE head sweep:** `gaussian` vs `mdn_k` vs `joint_mog` vs `rnade`.
   Fix everything else. This tells us how much expressivity matters.
2. **Mixture components** $K \in \{1, 4, 8, 16\}$ for `joint_mog`.
3. **With vs. without $y_t$ conditioning:** training a
   `use_observations=False` variant of IN should roughly recover the
   learned one-step prior $\hat p(x_t \mid x_{t-1})$ and serve as a
   direct analog of the "transition" proposal.
4. **Backbone capacity:** `hidden_dim` halved and doubled vs. default.
   At some point IN should saturate; where that happens vs. where RF
   saturates is informative.
5. **Process noise sweep:** as process noise shrinks, the optimal
   proposal becomes peakier; IN with small $K$ should start to struggle
   relative to RF here.
6. **Obs noise sweep:** the converse case (`obs_noise_std` small) is
   where an observation-informed proposal gives the largest wins over
   the bootstrap baseline.

---

## 10. Risks, pitfalls, open questions

- **Variance collapse in MDNs.** The most common MDN failure mode. The
  `min_sigma` floor plus `softplus` parameterization plus `logsumexp`
  numerics should prevent it. If it still happens, add a small entropy
  bonus on the mixture weights or switch to a log-variance
  parameterization.

- **Mode collapse.** If $K$ is much larger than the effective number of
  modes in the posterior, only one component carries weight. Log the
  mixture entropy during training to catch this.

- **Scale / Jacobian.** The proposal's `log_prob` is returned in scaled
  space; the transition's is in physical space. Same caveat as RF. Flag
  this in the README.

- **Expressivity on high-dim spatial systems.** For L96 (D = 40) and KS
  (D = 64), a `joint_mog` with $K = 8$ has only 8 components worth of
  multimodality for the whole joint. If the filtering posterior has
  meaningful spatial multimodality, this is not enough. The `rnade`
  head (option 4 in §4.2) is the principled fix, since it can represent
  arbitrarily complex joints via autoregression.

- **Distribution shift at test time.** Training samples have
  $x_{t-1} \sim p(x_{t-1})$ (the model's forward marginal). At test time
  $x_{t-1}$ is drawn from the *filtering posterior*, which can be
  heavily off-support if observations are informative. This is the same
  issue RF has and the paper does not address it explicitly; it is
  standard for amortized inference. Worth flagging in the paper writeup
  but not a blocker.

- **Gradient pathologies from very small $\sigma$.** If the target $x_t$
  is almost deterministic given $(x_{t-1}, y_t)$ (low process and obs
  noise), the NLL gradient can blow up as $\sigma \to 0$. `min_sigma`
  bounds this but also biases the model; set it to something like
  $10^{-3}$ in scaled space (where training data has unit variance)
  and tune from there.

- **Autoregressive ordering for RNADE.** The paper notes ordering is
  arbitrary. For spatial systems a natural default is the spatial
  ordering along the grid; for non-spatial systems any fixed ordering
  works. Randomized orderings (à la Uria et al.) are a stretch goal.

- **Observation sparsity (spatial).** If we ever move from "dense in
  space" observations to "sparse in space" (e.g., partial grid
  coverage), the mask handling in the backbone has to support it. This
  is already handled in `MLPVelocityNetwork` and should be mirrored in
  the feature backbone.

---

## 11. Phased implementation milestones

Phase-gated so each phase produces a runnable artifact.

**Phase 1 — skeleton (half a day).**
- Add `MLPFeatureBackbone` in `mlp.py` and `create_feature_backbone`
  factory.
- Add `GaussianHead` and `JointMoGHead` in
  `proposals/architectures/cde_heads.py`.
- Add `InferenceNetworkProposal` Lightning module with `training_step`
  and `validation_step` only.
- Train on `double_well_...` to convergence. Success metric: `val_nll`
  converges and the predicted mean on held-out data has reasonable RMSE.

**Phase 2 — sampling and log-prob (half a day).**
- Implement `sample` and `log_prob` on the Lightning module.
- Wire up `eval_proposal.py` for autoregressive evaluation; should work
  with zero extra code changes because of the ABC-compatible method
  signatures.
- Run on `double_well` and `lorenz63`. Compare one-step RMSE/CRPS to RF.

**Phase 3 — PF integration (half a day).**
- Add `InferenceNetworkProposal` wrapper in `models/proposals.py`.
- Register `"inn"` key in `run.py` / `eval.py`.
- Run BPF on `double_well` with IN proposal. Confirm RMSE and ESS are
  reasonable and that wall-clock is faster than RF.

**Phase 4 — spatial systems (one day).**
- Add `ResNet1DFeatureBackbone`.
- Train IN on L96 and KS with the same configs RF was trained on.
- Run PF evaluation. Expect this to expose head-expressivity
  limitations; that's the interesting experimental signal.

**Phase 5 — ablations and RNADE head (one to two days).**
- Run the ablation sweep from §9.
- Optionally implement the `RNADEHead` (option 4 in §4.2) and re-run on
  L96 / KS.

**Phase 6 — writeup (half a day).**
- Produce a results table (RF vs. IN) across datasets with RMSE, CRPS,
  ESS, wall-clock per PF step.
- Document the `log_prob`-in-scaled-space caveat and any surprises.

Total: roughly 4 to 5 days of work from someone who already knows the
codebase. Most of the risk is in Phase 5 (RNADE) if we go that far; the
first three phases are nearly mechanical.

---

## Appendix A. Correspondence between Paige-Wood notation and ours

| Paper notation | Our notation |
|---|---|
| $x$ (latents) | $x_{1:T}$ (state trajectory) |
| $y$ (observed) | $y_{1:T}$ (observation trajectory) |
| $\tilde p(x_i \mid \widetilde{PA}(x_i))$ | $p(x_t \mid x_{t-1}, y_t)$ |
| $q(x \mid \phi(\eta, y))$ | $q_\eta(x_t \mid x_{t-1}, y_t)$ |
| Joint density at head-to-head node | `JointMoGHead` or `RNADEHead` output |
| $\mathcal{J}(\eta)$ objective | Mean NLL over `(x_prev, x_curr, y_curr)` triples |

## Appendix B. Why this is simpler than NASMC (Gu et al. 2015)

NASMC (the next-logical learned-proposal baseline) also trains a
$q_\eta(x_t \mid x_{t-1}, y_t)$ but minimizes a different objective that
requires *running the particle filter* during training and propagating
gradients through the resampled weights. It is more principled in
principle (the objective matches the proposal's actual use) but
substantially more expensive and harder to tune. Paige-Wood sidesteps
this by targeting the forward KL under the model's joint, which can be
optimized offline with standard NLL training.

The practical upshot: **we can ship a credible Paige-Wood implementation
without any changes to the filtering code, and reuse the exact training
dataset used for RF.** This is the main reason this method is an
attractive first learned-proposal baseline to compare RF against.