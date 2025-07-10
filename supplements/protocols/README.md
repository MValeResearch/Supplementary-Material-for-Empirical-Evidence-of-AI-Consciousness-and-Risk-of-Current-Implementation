# Supplementary Protocols Index

This folder contains stepwise protocols referenced in the main manuscript, providing standardized methods and transparency for empirical evaluation and ethical experimentation in AI consciousness research.

---

## Protocol S1: Calibrating SIPT Weights (w₁–w₄)
**Purpose:**  
Provides a structured, empirical method for converting the four SIPT dimensions—Scale, Integration, Adaptive Dynamics, Neuromodulation—into predictive, architecture-agnostic coefficients.

**Background and Motivation:**  
Accurate and reproducible measurement of SIPT variables is essential for comparing consciousness-relevant capacity across diverse architectures. Previous approaches relied on ordinal estimates or hand-tuned weights. Protocol S1 formalizes weight calibration through cross-model benchmarking, regression, and hierarchical Bayesian modeling, enabling empirical SIPT scoring that is both robust and generalizable.

**Summary:**  
- Cross-architecture benchmarking
- Regression/lasso analysis to estimate provisional weights
- Causal perturbation (ablation, freezing, etc.)
- Hierarchical Bayesian group modeling
- Mechanistic validation (e.g., attention-graph, RLHF inspection)
- Preregistered prediction on new models

[See full protocol: Protocol_S1_Calibrating_SIPT_Weights.md]
