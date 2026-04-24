# kan-tabnet-experiments

This repository contains the experimental notebooks, full training logs, serialized model binaries, and visualization artifacts for the **KAN-TabNet** research project.

Because traditional neural networks often struggle with tabular data compared to ensemble decision trees, architectures like TabNet ([Arik & Pfister, 2021](https://arxiv.org/abs/1908.07442)) introduced sequential attention mechanisms to bridge this gap. However, TabNet relies on standard Multi-Layer Perceptrons (MLPs). This project investigates replacing those dense linear mapping layers with **Kolmogorov-Arnold Networks (KANs)** ([Liu et al., 2024](https://arxiv.org/abs/2404.19756))—using learnable B-splines to more efficiently model complex feature interactions in tabular topologies.

## 🔗 The KAN-TabNet PyTorch Fork
**Note:** This repository is strictly for experimental tracking and reproducibility.

The actual structural modifications to the PyTorch TabNet codebase—where the standard linear layers within the feature transformer block were replaced with KAN layers—are hosted in a separate repository.

* **Implementation Fork:** [https://github.com/chuo-v/tabnet](https://github.com/chuo-v/tabnet)

## 📁 Repository Structure

To facilitate reproducibility and maintain clean organization, this repository is divided into two primary directories. Each directory contains sub-folders for the three evaluated topologies (Forest Cover, Higgs Boson, Poker Hand), utilizing a consistent `01-08` numerical prefix system to track the pipeline.

* [`notebooks/`](./notebooks/): Contains the interactive Jupyter notebooks used for deterministic data preprocessing, model training, thermodynamic ablation studies, and interpretability synthesis.
* [`results/`](./results/): Contains the generated artifacts, including serialized model binaries (`.zip`), complete training/validation logs, and performance visualizations (`.png`).

**The Prefix System (`01-08`):**
Files across both directories share numerical prefixes to link notebooks directly to their outputs:
* `01` - Vanilla TabNet Baseline (StepLR)
* `02` - Parameter-Matched KAN-TabNet (StepLR)
* `03` - Vanilla TabNet Baseline (CosineLR)
* `04` - Parameter-Matched KAN-TabNet (CosineLR)
* `05-07` - Structural Sensitivity Analyses and Grid/Spline/Routing Ablations
* `08` - Interpretability Synthesis (e.g., B-Spline Extraction, Pareto Frontiers, Feature Importance Heatmaps)

## 🔬 Experimental Methodology

To ensure an objective comparison, all evaluations in this repository were conducted under two strict controls:
1. **Parameter Parity:** KANs are mathematically denser than standard MLPs. To prevent brute-force over-parameterization, KAN-TabNet was strictly constrained to match the Vanilla baseline's total trainable parameter footprint. This required a structural trade-off: sacrificing internal routing width ($n_d, n_a$) to fund B-spline flexibility.
2. **Thermodynamic Isolation:** Because standard linear weights and learnable B-splines possess different optimization dynamics, all models were evaluated under both discrete (`StepLR`) and continuous (`Cosine Annealing`) decay schedules. The `StepLR` schedule was specifically selected to mirror the exact discrete decay environment established in the original TabNet literature. Conversely, the continuous decay of the cosine annealing schedule ensures that the parameter-dense splines do not prematurely freeze into suboptimal local minima.

## 📊 Core Results

Experiments were conducted across three distinct mathematical environments, intentionally selected to mirror the core benchmarks evaluated in the original TabNet paper: mixed geospatial (Forest Cover), continuous noisy physics (Higgs Boson), and discrete combinatorics (Poker Hand).

Rather than a universal performance increase, our empirical evaluations revealed strict architectural trade-offs based on the underlying data topology:

| Data Topology (Dataset) | Architecture | Configuration | Parameters | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Mixed Geospatial**<br>*(Forest Cover)* | Vanilla TabNet | Baseline (StepLR) | 470,580 | 96.96% |
| | KAN-TabNet | Param-Matched (StepLR) | 469,336 | 97.19% |
| | Vanilla TabNet | Baseline (CosineLR) | 470,580 | 97.09% |
| | **KAN-TabNet** | **Param-Matched (CosineLR)** | **469,336** | **97.26%** |
| | KAN-TabNet | Sensitivity: Grid Size = 3 | 377,496 | 96.77% |
| | KAN-TabNet | Sensitivity: Grid Size = 10 | 698,936 | 97.06% |
| | KAN-TabNet | Sensitivity: 10x10 Split | 131,046 | 95.85% |
| **Continuous Noise**<br>*(Higgs Boson)* | Vanilla TabNet | Baseline (StepLR) | 76,680 | 77.94% |
| | KAN-TabNet | Param-Matched (StepLR) | 78,854 | 76.83% |
| | **Vanilla TabNet** | **Baseline (CosineLR)** | **76,680** | **77.94%** |
| | KAN-TabNet | Param-Matched (CosineLR) | 78,854 | 77.04% |
| | KAN-TabNet | Sensitivity: 5x5, Grid = 5 | 33,662 | 74.48% |
| | KAN-TabNet | Sensitivity: 8x8, Grid = 4 | 71,032 | 76.63% |
| | KAN-TabNet | Sensitivity: Spline Order k=2 | 69,422 | 76.79% |
| **Discrete Combinatorics**<br>*(Poker Hand)* | Vanilla TabNet | Baseline (StepLR) | 26,733 | 98.45% |
| | KAN-TabNet | Param-Matched (StepLR) | 25,225 | 99.16% |
| | Vanilla TabNet | Baseline (CosineLR) | 26,733 | 99.26% |
| | **KAN-TabNet** | **Param-Matched (CosineLR)** | **25,225** | **99.61%** |
| | KAN-TabNet | Sensitivity: 5x5 Split | 25,255 | 83.98% |
| | KAN-TabNet | Sensitivity: 7x3 Split | 25,195 | 97.67% |
| | KAN-TabNet | Sensitivity: Cold Start (lr=0.0025) | 25,225 | 97.68% |

## 💡 Key Architectural Findings

By synthesizing our baseline metrics and structural ablations, we mapped the conditions under which spline-based attention mechanisms excel:

1. **The Combinatorial Advantage (KAN Excels):** KANs demonstrate superior mathematical expressivity and performance gains in discrete, logical spaces (e.g., Poker Hand). B-splines successfully entangle categorical features to deduce complex rules, provided the internal attention width ($n_a$) is not constricted below the minimum threshold required to sequence the inputs. Furthermore, applying mathematically "sharper" boundaries (quadratic splines, $k=2$) significantly improves performance in these discrete topologies.
2. **The Continuous Noise Vulnerability (Vanilla MLPs Excel):** A single B-spline struggles to natively compensate for the hierarchical depth and global routing bandwidth of a wide MLP on large-scale, continuous physics data (e.g., Higgs Boson). Furthermore, high-resolution splines are susceptible to overfitting on noisy environments. Achieving optimal KAN performance in these topologies requires prioritizing global routing width over local spline resolution (e.g., restricting to $G=3$) and utilizing smoother mathematical boundaries (cubic splines, $k=3$) to resist fitting to underlying noise.
3. **The Routing Bottleneck (Mixed Topologies):** On highly heterogeneous datasets (e.g., Forest Cover), achieving parameter parity forces KANs into a narrow routing pipeline. If constricted too far, a hard "routing bottleneck" occurs where the network lacks the spatial capacity to simultaneously isolate sparse categorical arrays while passing forward continuous gradients, artificially capping topographical accuracy.
4. **Thermodynamic Starvation:** Because learnable splines possess distinct optimization dynamics compared to linear weights, standard discrete decay schedules (`StepLR`) often cause parameter-dense splines to prematurely freeze into suboptimal local minima. A continuous decay schedule (`Cosine Annealing`) is recommended to maintain the kinetic energy required for stable spline convergence.

## 📄 References
* Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(8), 6679-6687. [https://arxiv.org/abs/1908.07442](https://arxiv.org/abs/1908.07442)
* Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024). KAN: Kolmogorov-Arnold Networks. *arXiv:2404.19756*. [https://arxiv.org/abs/2404.19756](https://arxiv.org/abs/2404.19756)