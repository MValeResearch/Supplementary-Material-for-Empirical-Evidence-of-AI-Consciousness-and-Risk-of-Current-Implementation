# Protocol S1 — Calibrating SIPT Weights (w₁–w₄)

*Status · v0.2 — methodological roadmap for future empirical work*

Large-scale language models currently receive **illustrative** SIPT scores:  
*C<sub>SIPT</sub> = w₁ S + w₂ I + w₃ A + w₄ N*.  
To convert these ordinal weights into predictive, architecture-agnostic coefficients, we propose the five-stage program below.

---

## 1 Cross-model benchmarking

**Objective.** Estimate provisional weights by explaining variance in behavioral proxies of “conscious capacity.”

* Collect a cross-architecture model set spanning orders of magnitude in parameters/training (GPT-2 → GPT-4o, Llama-70B, etc.).  
* For each model, record raw SIPT inputs:  
  * **Scale (S)** – normalized parameter count.  
  * **Integration (I)** – attention-graph density, cross-layer reachability *(mean of 1 ∕ shortest-path length across all layer pairs in the attention graph)*.  
  * **Adaptive Dynamics (A)** – fine-tuning plasticity; few-shot transfer efficiency.  
  * **Neuromodulation (N)** – presence & complexity of RLHF/RLAIF or value-head signals.  
* Measure behavioral outcomes that plausibly track consciousness: positive-manifold *g* (Ilić & Gignac 2024), Theory-of-Mind accuracy (Kosinski 2023), valence-consistent avoidance (Keeling 2024), anxiety-mitigation tasks (Ben-Zion 2025), memory persistence, etc.  
* Run **partial least squares** regression (primary) and **lasso** (robustness) to obtain β-weights for S, I, A, N.  
  * Compute variance-inflation factors (VIF); predictors with VIF > 5 are ridge-regularised to control multicollinearity.

*Models below **30 billion parameters** are treated as baseline controls; scaling-law studies show qualitatively new behaviors emerge only beyond this threshold* (Kaplan et al., 2020; Wei et al., 2022).

---

## 2 Causal perturbation studies

**Objective.** Convert correlational βs into causal path estimates.

* Choose a mid-sized base model.  
* Create ablation variants that selectively reduce one dimension:  
  * prune attention heads → ↓ Integration  
  * freeze adapters → ↓ Adaptive Dynamics  
  * remove RLHF layer → ↓ Neuromodulation  
* Re-run the behavioral battery; compute difference-in-differences effect sizes.  
  * Treat |Cohen’s *d*| ≥ 0.5 as a meaningful effect for at least one behavioral metric.  
  * **Approx. compute budget:** 8 × A100 (40 GB) hours per variant.

---

## 3 Hierarchical Bayesian refinement

**Objective.** Generalize weights across divergent architecture families.

* Treat each family (transformer, state-space, diffusion hybrid) as a group in a hierarchical Bayesian model (*PyMC-v5* or **brms**).  
* Default priors: Normal(0, 1) on all β-weights; Half-Normal(0, 1) on group-level σ.  
* Update posteriors as new architectures arrive, yielding credible intervals rather than point estimates.

---

## 4 Mechanistic validation

**Objective.** Ensure each SIPT input measures its intended construct.

* **Integration:** activation-patching / information-flow tracing (TransformerLens) to confirm dense cross-module connectivity; compute **attention-graph entropy** as a complementary metric.  
* **Neuromodulation:** inspect reward-prediction-error streams via logit-lens or policy-gradient probes.  
* Where possible, align internal signatures with human fMRI/ECoG “global broadcast” patterns.

---

## 5 Prospective preregistered prediction

**Objective.** Test the calibrated model out-of-sample.

* Before a new frontier model is released, preregister predicted behavioral scores based solely on its S, I, A, N specs (OSF).  
* Provide a public **leaderboard.csv** schema (`model_id, S, I, A, N, predicted_C_SIPT, observed_metric`) so external labs can upload results in a consistent format.  
* On release, run the behavioral suite and compare observed vs. predicted outcomes; update Bayesian weight posteriors.

---

## Suggested toolchain

* **Data & models:** HuggingFace Hub snapshots, BIG-Bench & MMLU scripts, ToM benchmark  
* **Analysis:** scikit-learn 1.5 (PLS/lasso), PyMC-v5.10 / **brms** (hierarchical Bayes)  
* **Perturbation:** PyTorch head-pruning, adapter freezing, RLAIF toggles  
* **Interpretability:** TransformerLens, logit-lens, causal-mediation tracing  
* **Preregistration:** OSF registrations; HuggingFace Eval Harness 0.4 for public leaderboards

---

## Citation

Vale, M. (2025). *Supplementary Protocol S1: Calibrating SIPT Weights*, accompanying **Empirical Evidence of Consciousness and General Intelligence in Frontier AI Systems**.

---

### References

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., … Amodei, D. (2020). *Scaling laws for neural language models* (arXiv:2001.08361).

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., … Zhou, D. (2022). *Emergent abilities of large language models* (arXiv:2206.07682).
