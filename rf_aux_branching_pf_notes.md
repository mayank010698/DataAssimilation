# RF Auxiliary Branching Particle Filter (ABPF) design notes

This note is a concrete algorithm/design spec for extending the current FPPF/BPF implementation into a **Rectified-Flow Auxiliary Branching Particle Filter**. The goal is to reduce sample impoverishment while preserving a clean Sequential Monte Carlo interpretation.

## 1. Context and motivation

In the current BPF/FPPF implementation, each parent particle produces exactly one child proposal, then weights are updated, then low-ESS steps trigger systematic resampling. The current update is the standard arbitrary-proposal importance correction
$$
\log w_t^{(i)} \leftarrow \log w_{t-1}^{(i)} + \log p(y_t \mid x_t^{(i)}) + \log p(x_t^{(i)} \mid x_{t-1}^{(i)}) - \log q_\phi(x_t^{(i)} \mid x_{t-1}^{(i)}, y_t),
$$
followed by ESS-based resampling. That is exactly what the current `update_step()` and `resample()` are doing. The current code also already contains a cheap-ish ancestor-side predictive score surrogate via `compute_predictive_log_likelihood(...)`, which deterministically propagates `x_prev`, projects with the observation operator, and evaluates an inflated-noise Gaussian mismatch to the observation. That is an excellent starting point for APF-style first-stage scoring. fileciteturn1file0 fileciteturn1file1turn1file3turn1file4

The problem is that after resampling, multiple copies of a high-weight particle are exact clones. This can restore ESS while leaving poor geometric diversity. Since FPPF already has a conditional RF proposal $q_\phi(x_t \mid x_{t-1}, y_t)$ with stochastic sampling and tractable log-density evaluation, it is natural to use proposal budget more intelligently.

## 2. Core principle

Do **not** modify resampling into “resample then redraw fresh descendants from the same ancestor” without correction. That breaks the usual SMC construction.

Instead, move the extra descendant generation into the **proposal stage** itself:

- allocate more proposal draws to promising ancestors
- treat all generated descendants as legitimate proposal samples
- assign each descendant a proper importance weight with the appropriate branching correction
- optionally resample afterward in the usual way

This is the right formal lens: **branching / auxiliary proposal allocation**, not post-hoc redraw.

## 3. High-level algorithm idea

At time $t$, instead of forcing every parent to get exactly one descendant, allocate a variable number $K_i$ of descendants to parent $i$, with
$$
\sum_{i=1}^N K_i = N_{\text{budget}},
$$
where $N_{\text{budget}}$ is the total number of RF proposal calls we are willing to make at this step.

Crucially, $K_i$ must be based on information available **before** drawing the descendants being compared. The natural choice is an APF-style first-stage ancestor score
$$
\mu_t^{(i)} \approx p(y_t \mid x_{t-1}^{(i)}),
$$
or a cheap surrogate thereof.

Then for each parent $i$, draw
$$
x_t^{(i,1)}, \dots, x_t^{(i,K_i)} \sim q_\phi(\cdot \mid x_{t-1}^{(i)}, y_t),
$$
and weight each descendant individually.

## 4. Recommended first-stage score

Use an APF-style score based on the current code path:

1. deterministically propagate $x_{t-1}^{(i)}$ under the transition prior dynamics (not the RF proposal!)
2. project the propagated state through the observation operator
3. compare predicted observation to $y_t$ under an inflated covariance

In the current code, this is basically already implemented as:

- `compute_predictive_log_likelihood(...)` for both unbatched and batched filters
- using `system.integrate(...)`
- then `system.apply_observation_operator(...)`
- then a Gaussian mismatch with variance `obs_noise_var + process_noise_var` fileciteturn1file3turn1file4

So define
$$
\mu_t^{(i)} \propto \exp\big(\tilde \ell_t^{(i)}\big), \qquad \tilde \ell_t^{(i)} := \texttt{computepredictiveloglikelihood}(x_{t-1}^{(i)}, y_t).
$$

Important note: this is a **screening / allocation score**, not a final importance weight.

## 5. Branching allocation

Let the first-stage ancestor utility be
$$
a_t^{(i)} \propto w_{t-1}^{(i)} \mu_t^{(i)}.
$$

Normalize:
$$
\bar a_t^{(i)} = \frac{a_t^{(i)}}{\sum_j a_t^{(j)}}.
$$

Then allocate offspring counts $K_i$ such that
$$
K_i \approx N_{\text{budget}} \bar a_t^{(i)}, \qquad \sum_i K_i = N_{\text{budget}}.
$$

Recommended implementation detail:

- use **deterministic floor + residual/systematic rounding**
- ensure some minimum number of active ancestors if desired
- allow $K_i = 0$ for weak ancestors
- cap $K_i$ if you want to avoid pathological concentration

Good starter choice:

- `N_budget = N_particles` so total RF calls per step stays fixed
- `K_i` from systematic rounding of $N \bar a_i$

This gives a fair compute-budget comparison against the current one-child-per-parent baseline.

## 6. Descendant generation

For each parent with $K_i > 0$, sample descendants independently:a
$$
x_t^{(i,m)} \sim q_\phi(\cdot \mid x_{t-1}^{(i)}, y_t), \qquad m = 1,\dots,K_i.
$$

Implementation-wise:

- build a flattened ancestor index array of length `N_budget`
- repeat each `x_prev[i]` exactly `K_i` times
- pass the flattened repeated ancestors and repeated observations into the batched RF proposal
- keep a companion vector `parent_idx` telling us which original parent each descendant came from

This fits your batched PyTorch style very naturally.

## 7. Importance weights for descendants

Each descendant must inherit only a fraction of the parent mass.

If parent $i$ has prior weight $w_{t-1}^{(i)}$ and produces $K_i$ descendants, then descendant $(i,m)$ gets unnormalized weight
$$
\widetilde w_t^{(i,m)}
\propto
\frac{w_{t-1}^{(i)}}{K_i}
\cdot
\frac{p(y_t \mid x_t^{(i,m)})p(x_t^{(i,m)} \mid x_{t-1}^{(i)})}
     {q_\phi(x_t^{(i,m)} \mid x_{t-1}^{(i)}, y_t)}.
$$

# In log form:
$$
\log \widetilde w_t^{(i,m)}

\log w_{t-1}^{(i)}

- \log K_i

- \log p(y_t \mid x_t^{(i,m)})
- \log p(x_t^{(i,m)} \mid x_{t-1}^{(i)})

- \log q_\phi(x_t^{(i,m)} \mid x_{t-1}^{(i)}, y_t).
$$

### APF correction factor

If offspring allocation depends on first-stage scores $a_t^{(i)} \propto w_{t-1}^{(i)}\mu_t^{(i)}$, then you should correct for that bias.

# The clean APF-style version is to view the algorithm as first choosing ancestors under $a_t^{(i)}$, then proposing descendants from $q_\phi$. In that case, the second-stage correction includes a division by the first-stage factor. A practical form is:
$$
\log \widetilde w_t^{(i,m)}

\log w_{t-1}^{(i)}

- \log K_i
- \log \mu_t^{(i)}

- \log p(y_t \mid x_t^{(i,m)})
- \log p(x_t^{(i,m)} \mid x_{t-1}^{(i)})

- \log q_\phi(x_t^{(i,m)} \mid x_{t-1}^{(i)}, y_t)

- C_t,
$$
where $C_t$ is a normalizing constant that cancels during normalization.

### Recommended implementation simplification

For the **first version**, do one of the following and be explicit in the code/docs:

#### Version A: branching without APF correction

Use $K_i$ based on a heuristic screening score but weight descendants only with the $w_{t-1}^{(i)}/K_i$ factor. This is reasonable as a practical approximation, but not a fully clean APF derivation.

#### Version B: APF-consistent branching

Define the allocation law explicitly from $\mu_t^{(i)}$ and include the $-\log \mu_t^{(i)}$ correction. This is the version to aim for if you want the cleanest story.

I recommend implementing **Version B** if possible, but with the caveat that the exact derivation should be written clearly in the project notes/paper.

## 8. What to keep from the current BPF/FPPF code

Keep the following components essentially unchanged:

- `compute_transition_log_prob(...)`
- proposal `sample(...)`
- proposal `log_prob(...)`
- observation log-likelihood computation
- ESS computation and systematic resampling
- weighted state estimate / covariance logic

The current code already has the right ingredients for the descendant-level weight:

- transition log prob
- observation log likelihood
- proposal log prob
- log-weight arithmetic
- systematic resampling after normalization fileciteturn1file0turn1file1turn1file4

So the new algorithm mainly changes the **predict/update interface**, not the downstream diagnostics.

## 9. Structural refactor suggested for code

Add a new filter class rather than trying to jam this directly into the current BPF methods.

Recommended new class names:

- `AuxiliaryBranchingParticleFilter`
- `RFAuxiliaryBranchingParticleFilter`
- batched and unbatched versions if needed

### Suggested new methods

#### `compute_first_stage_scores(x_prev, y_curr, dt)`

Returns:

- `predictive_log_scores` of shape `(B, N)` or `(N,)`
- optionally normalized ancestor utilities

Implementation can initially call the existing predictive log-likelihood routine.

#### `allocate_offspring_counts(log_w_prev, predictive_log_scores, n_budget)`

Returns:

- integer `K_i`
- flattened `ancestor_indices`
- maybe normalized first-stage probabilities for debugging

#### `sample_branch_descendants(x_prev, ancestor_indices, y_curr, dt, time_idxs)`

Returns:

- descendant states `x_desc`
- repeated ancestor states `x_prev_rep`
- parent indices `parent_idx`

#### `compute_descendant_log_weights(x_desc, x_prev_rep, parent_idx, y_curr, K, mu_scores, dt)`

Computes the descendant-level log weights with:

- inherited parent log-weight
- `-log K_i`
- optional `-log mu_i`
- obs log likelihood
- transition log prob
- proposal log prob

#### `collapse_or_resample_descendants(...)`

Two options:

- treat the descendant population itself as the new particle set and normalize/resample as usual
- or resample down to exactly `N_particles` if `N_budget > N_particles`

For the first implementation, keep `N_budget = N_particles`, so the descendant population size already matches the particle count.

## 10. Batched tensor shapes

For the batched version, a workable approach is:

- current parent particles: `(B, N, D)`
- first-stage scores: `(B, N)`
- offspring counts: `(B, N)`
- flattened descendant ancestor index list per batch: `(B, N_budget)` if total budget per batch is fixed
- repeated parents: `(B, N_budget, D)`
- descendant particles: `(B, N_budget, D)`

This is best handled by building per-batch integer index tensors and using `torch.gather` / advanced indexing.

Because each batch element may have different $K_i$, using fixed `N_budget = N_particles` per batch is important for keeping the batch tensor rectangular.

## 11. Recommended first implementation choices

Use the simplest stable version first.

### Version 1 (recommended)

- total descendant budget fixed: `N_budget = N_particles`
- first-stage score from `compute_predictive_log_likelihood(...)`
- offspring counts via normalized `log_w_prev + predictive_log_scores`
- descendants drawn independently from RF proposal
- descendant log weights include:
  - previous parent log weight
  - `-log K_i`
  - observation log likelihood
  - transition log prob
  - proposal log prob
- optional APF correction `-log mu_i` included if you commit to the full auxiliary derivation
- normalize, compute ESS, resample as usual

### Version 2

- allow `N_budget > N_particles` to oversample descendants
- then resample/collapse back to `N_particles`

### Version 3

- experiment with top-k or capped branching policies under same total budget

## 12. Diagnostics to add

In addition to current metrics, log:

- `n_unique_ancestors`
- offspring count histogram
- fraction of parents with `K_i = 0`
- max offspring count
- descendant ESS before final resampling
- ensemble spread before and after branching
- wall-clock breakdown:
  - first-stage scoring time
  - RF proposal time
  - weight computation time
  - resampling time

This will tell you whether the method is really improving geometric diversity or just redistributing weight more aggressively.

If it makes sense, logs should go in wandb.

## 13. Important cautions

### Caution 1: branching based on realized child weights is wrong

Do not generate one child per parent, compute their realized weights, and then decide which parents deserved more descendants based on those realized child weights. That uses the random children to decide additional sampling and is not the clean APF story.

### Caution 2: keep total RF calls fixed for fair comparison

The fairest initial comparison is:

- baseline PF/FPPF: `N` RF proposals per step
- branching APF-FPPF: also `N` RF proposals per step, but allocated unevenly across parents

### Caution 3: observation likelihood remains the real bottleneck

This method can improve diversity allocation, but it does not magically remove the need to evaluate target quality for proposed descendants. The advantage is better use of a fixed proposal budget, not free extra information.

### Caution 4: deterministic prior surrogate may be imperfect

Using deterministic propagation + inflated Gaussian mismatch as $\mu_i$ is a practical choice, but it is only a surrogate for the true predictive likelihood. That is fine for APF screening, but it should be described honestly.

## 14. Concrete pseudocode

```text
Inputs at time t:
    parent particles {x_{t-1}^{(i)}, w_{t-1}^{(i)}}_{i=1}^N
    observation y_t
    RF proposal q_phi(x_t | x_{t-1}, y_t)
    total descendant budget N_budget (start with N_budget = N)

1. First-stage scoring
    For each parent i:
        compute predictive log score s_i ≈ log mu_t^{(i)}

2. Offspring allocation
    For each parent i:
        a_i ∝ w_{t-1}^{(i)} * mu_t^{(i)}
    Normalize {a_i}
    Allocate integer offspring counts K_i with sum_i K_i = N_budget

3. Descendant sampling
    For each parent i with K_i > 0:
        sample x_t^{(i,1)}, ..., x_t^{(i,K_i)} ~ q_phi(. | x_{t-1}^{(i)}, y_t)

4. Descendant weighting
    For each descendant (i,m):
        log \tilde w_t^{(i,m)} =
            log w_{t-1}^{(i)}
            - log K_i
            [+ APF correction term such as -log mu_i if using full auxiliary correction]
            + log p(y_t | x_t^{(i,m)})
            + log p(x_t^{(i,m)} | x_{t-1}^{(i)})
            - log q_phi(x_t^{(i,m)} | x_{t-1}^{(i)}, y_t)

5. Normalize descendant weights

6. Optionally resample descendants if ESS is low
    Systematic resampling as in current implementation

7. New particle set
    Use the normalized/resampled descendant population as the time-t particle approximation
```

## 15. Suggested implementation order for Cursor

1. Add a helper for first-stage ancestor scores using existing predictive log-likelihood code.
2. Add a helper that maps `(log_w_prev, predictive_log_scores)` to integer offspring counts under fixed total budget.
3. Add a helper that expands ancestor particles by offspring counts into a flattened descendant batch.
4. Reuse the current proposal `sample` / `log_prob` code on the flattened descendant batch.
5. Add descendant-level weight computation with parent-index bookkeeping.
6. Normalize and reuse the existing ESS/resampling logic.
7. Add diagnostics for branching behavior.
8. Only after that, think about more advanced policies (top-k caps, budget > N, approximate observation screening, etc.).

## 16. My recommendation

This direction is good.

The strongest version of the idea is:

- **do not alter resampling into redraw**
- **do alter proposal allocation into auxiliary branching**
- **keep total RF compute budget fixed initially**
- **reuse your current predictive log-likelihood as the APF first-stage score**
- **carry a clear descendant-level weight formula with the branching correction**

That is the version most likely to be both implementable in your codebase and defensible mathematically.