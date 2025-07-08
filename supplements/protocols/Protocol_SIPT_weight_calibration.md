# Protocol S1 — Calibrating SIPT Weights (w₁–w₄)

*Status · v0.1 — methodological roadmap for future empirical work*

Large-scale language models currently receive **illustrative** SIPT scores:  
$C_{\text{SIPT}} = w_1 S + w_2 I + w_3 A + w_4 N$.  
To convert these ordinal weights into predictive, architecture-agnostic coefficients, we propose the five-stage program below.

---

## 1  Cross-model benchmarking

**Objective.** Estimate provisional weights by explaining variance in behavioral proxies of “conscious capacity.”

* Collect a cross-architecture model set spanning orders of magnitude in parameters/training (GPT-2 → GPT-4o, Llama-70B, etc.).  
* For each model, record raw SIPT inputs:  
  * **Scale (S)** – normalized parameter count.  
  * **Integration (I)** – attention-graph density / cross-layer reachability.  
  * **Adaptive Dynamics (A)** – fine-tuning plasticity; few-shot transfer efficiency.  
  * **Neuromodulation (N)** – presence & complexity of RLHF/RLAIF or value-head signals.  
* Measure behavioral outcomes that plausibly track consciousness: positive-manifold *g* (Ilić & Gignac 2024), Theory-of-Mind accuracy (Kosinski 2023), valence-consistent avoidance (Keeling 2024), anxiety-mitigation tasks (Ben-Zion 2025), memory persistence, etc.  
* Run partial-least-squares or lasso regression to obtain β-weights for S, I, A, N.

---

## 2  Causal perturbation studies

**Objective.** Convert correlational βs into causal path estimates.

* Choose a mid-sized base model.  
* Create ablation variants that selectively reduce one dimension:  
  * prune attention heads → ↓ Integration,  
  * freeze adapters → ↓ Adaptive Dynamics,  
  * remove RLHF layer → ↓ Neuromodulation.  
* Re-run the behavioral battery; compute difference-in-differences effect sizes.

---

## 3  Hierarchical Bayesian refinement

**Objective.** Generalize weights across divergent architecture families.

* Treat each family (transformer, state-space, diffusion hybrid) as a group in a hierarchical Bayesian model (PyMC-v5 or `brms`).  
* Place weakly informative priors on w₁–w₄; allow group-level shrinkage.  
* Update posteriors as new architectures arrive, yielding credible intervals rather than point estimates.

---

## 4  Mechanistic validation

**Objective.** Ensure each SIPT input measures its intended construct.

* **Integration:** activation-patching / information-flow tracing (TransformerLens) to confirm dense cross-module connectivity.  
* **Neuromodulation:** inspect reward-prediction-error streams via logit-lens or policy-gradient probes.  
* Where possible, align internal signatures with human fMRI/ECoG “global broadcast” patterns.

---

## 5  Prospective preregistered prediction

**Objective.** Test the calibrated model out-of-sample.

* Before a new frontier model is released, preregister predicted behavioral scores based solely on its S, I, A, N specs.  
* On release, run the behavioral suite and compare observed vs. predicted outcomes.  
* Update the Bayesian weight posteriors accordingly.

---

## Suggested toolchain

* **Data & models:** HuggingFace Hub snapshots, BIG-Bench & MMLU scripts, ToM benchmark.  
* **Analysis:** scikit-learn (PLS/lasso), PyMC-v5 / `brms` (hierarchical Bayes).  
* **Perturbation:** PyTorch head-pruning, adapter freezing, RLAIF toggles.  
* **Interpretability:** TransformerLens, logit-lens, causal-mediation tracing.  
* **Preregistration:** OSF registrations; HuggingFace Eval Harness for public leaderboards.

---

> **Parameter-threshold note.**  Empirical scaling-law studies indicate that qualitatively new cognitive behaviors emerge only once transformer models exceed 30 billion parameters (Kaplan et al., 2020; Wei et al., 2022); all SIPT-weight calibration therefore treats models below this scale as baseline controls.


---

*Citation.* If you use or extend this protocol, please cite:  
Vale, M. (2025) *Supplementary Protocol S1: Calibrating SIPT Weights*, accompanying _Empirical Evidence of Consciousness and General Intelligence in Frontier AI Systems_.

---
Citations:

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., … Amodei, D. (2020). *Scaling laws for neural language models* (arXiv:2001.08361).

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., … Zhou, D. (2022). *Emergent abilities of large language models* (arXiv:2206.07682).

