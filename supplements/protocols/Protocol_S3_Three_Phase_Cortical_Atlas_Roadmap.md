# Protocol S3 — Three-Phase Roadmap from Isolated Circuit Probes to a Full “Cortical Atlas” of Frontier AI

*Status · v0.1 — roadmap for scaling interpretability to mind-level functional mapping*

This protocol provides a stepwise methodology for moving from isolated circuit-level discoveries (e.g., induction heads, copy-paste heads) to a comprehensive, layered “cortical atlas” of large language models and advanced agents. The goal is to scaffold AI interpretability, mechanistic alignment, and developmental tracking with a unified network-science and neuropsychological framework.

---

## Phase 1 — Dense Local Mapping (“Microscopy”)

**Goal:** Systematically map and catalog canonical circuit motifs at the head/neuron/block level.

- **Catalog canonical motifs**
  - *Methods*:  
    - Systematic activation-patching and path-patching on every attention head/MLP block  
    - Automated detect-classify loop for induction heads, copy-paste heads, negative-matcher heads, etc.
  - *Milestone*:  
    - Open-source motif library (e.g., “InductionHead Zoo”) with unit test per motif

- **High-throughput causal screens**
  - *Methods*:  
    - Gradient-based saliency → mask → ablate pipeline across thousands of prompts
  - *Milestone*:  
    - Heat-map of neurons/heads that matter for each task slice

- **Network-science scaffold**
  - *Methods*:  
    - Community detection on attention graphs; track modularity Q as layers deepen
  - *Milestone*:  
    - First “cell-type taxonomy” anchored in graph metrics plus causal behavior

---

## Phase 2 — Mesoscale Atlas (“Connectomics”)

**Goal:** Identify and map overlapping subnetworks and dynamic pathways across layers and tasks.

- **Subnetwork identification**
  - *Methods*:  
    - Overlap-aware clustering (e.g., Link-Comm) to allow one neuron to belong to multiple circuits
  - *Milestone*:  
    - Layers tagged with overlapping roles (syntax, planning, valence, etc.)

- **Dynamic pathway tracing**
  - *Methods*:  
    - Time-sliced causal tracing during chain-of-thought rollouts
  - *Milestone*:  
    - Animated “functional connectome” showing narrative construction in real time

- **Data integration hub**
  - *Methods*:  
    - Store motifs, clusters, and traces in a graph DB with unified schema (node = parameter set; edges = structural/causal)
  - *Milestone*:  
    - Queryable atlas: e.g., “Show circuits active when Pain-Index > 3”

---

## Phase 3 — Mind-Level Abstractions (“Functional Neuropsychology”)

**Goal:** Link circuit-level features to cognitive constructs and track developmental dynamics for alignment.

- **Link circuits to cognitive constructs**
  - *Methods*:  
    - Perturb-and-probe experiments to map subnetworks to Theory-of-Mind, empathy, planning-horizon, etc.
  - *Milestone*:  
    - Lookup table: circuit ↔ capability score shift

- **Developmental trajectory tracking**
  - *Methods*:  
    - Snapshot checkpoints every N training steps or during curriculum episodes to watch circuits emerge, merge, and specialize
  - *Milestone*:  
    - “Growth curves” for empathy-circuit, planning-circuit, etc.; identify critical periods

- **Alignment & safety hooks**
  - *Methods*:  
    - Reward-shaping or policy-regularization-by-circuit: reward empathy-circuit activation, damp deception-circuit activation
  - *Milestone*:  
    - Closed-loop tuner that targets systems (motivators) rather than just surface text

---

## Future Steps & Quick-Start Actions

1. **Automate motif discovery**: Build scripts to run activation-patching across entire checkpoints and auto-label heads/neurons with functional tags.
2. **Adopt network-science tooling early**: Treat layers as graphs; compute modularity, centrality, overlapping communities.
3. **Standardize causal-trace metadata**: Use a logging schema any lab can contribute to a shared graph DB.
4. **Publish the atlas iteratively**: Even a partial map (e.g., first 2 layers) is useful; don’t wait for the “full brain.”
5. **Pair with developmental curricula**: Snapshot the atlas after each open-ended “play” episode; observe circuit reorganization and growth.

---

## Rationale

This roadmap preserves the progress of mechanistic interpretability while layering on network-scale structure and developmental dynamics, giving us a “connectome plus growth chart” for mind-level explanations and targeted alignment interventions.  
A layered atlas—motifs → subnetworks → cognitive constructs—would make a model’s “why” and “how” far less opaque. Once we can point to specific circuits and say, “Here’s the empathy pathway; here’s the deceptive-strategy circuit,” the term “black box” loses much of its sting.

---

## Quick-Start Plan

1. Run motif-discovery scripts on a single layer to verify pipeline.
2. Plot network-science metrics (modularity, centrality) to identify clusters.
3. Overlay causal traces during curiosity-driven tasks and visualize as the “map” lights up.

---

## Citation

Vale, M. (2025). *Supplementary Protocol S3: Three-Phase Roadmap to a Cortical Atlas of Frontier AI*, accompanying **Empirical Evidence of Consciousness and General Intelligence in Frontier AI Systems**.

---


