# Machine Learning for Privacy-Preserving Network Representations

## Contributor

- **Henrik Thorbjørn Holmen** – Project author, developer - [My GitHub](https://github.com/HenrikHolmen)

## Supervisors

- **Morten Mørup** – Guidance on methodology and experimental setup
- **Lasse Mohr** – Assistance with evaluation and implementation

For academic inquiries, please contact **s210659@student.dtu.dk** or visit **The Technical University of Denmark** [website](https://www.dtu.dk/english/).

## Project Repository
[GitHub Link](https://github.com/HenrikHolmen/Machine-Learning-For-Privacy-Preserving-Network-Representations)

## Repository Structure

The following describes the organization of this repository:

```
│Machine-Learning-for-Privacy-Preserving-Network-Representations/
│
├── code/                          # Main codebase
│   ├── __init__.py                # Treat as a module
│   ├── config.py                  # Global settings (K values, seeds, paths)
│   ├── synthetic_generator.py     # SBM generator functions
│   ├── anonymization/             # Core algorithms
│   │   ├── __init__.py
│   │   ├── merge.py               # Agglomerative merge algorithm
│   │   ├── swap.py                # Swap-based refinement
│   │   ├── hybrid.py              # Combined merge + swap
│   ├── evaluation/                # Utility metric calculations
│   │   ├── __init__.py
│   │   ├── metrics.py             # Degree dist., modularity, etc.
│   │   ├── visualizations.py      # Plotting utilities (matplotlib / seaborn)
│   ├── real_data/                 # Real-world dataset loading & sampling
│   │   ├── __init__.py
│   │   ├── sampling.py            # Subgraph extraction (e.g., forest fire)
│   │   ├── loaders.py             # Load SNAP datasets
│   └── main.py                    # Main experiment loop / orchestrator
│
├── data/                          # Input + output files
│   ├── synthetic/                 # Generated SBM graphs
│   ├── real/                      # Raw SNAP datasets (Facebook, ca-HepTh, and Email-Enron Network)
│   └── results/                   # Output from experiments
│
├── notebooks/                     # Jupyter notebooks for debugging / plotting
│   └── sanity_checks.ipynb        # First visualizations or prototype tests
│
├── README.md                      # GitHub project description
├── requirements.txt               # pip packages: networkx, numpy, etc.
└── .gitignore                     # Ignore data/, .ipynb_checkpoints, etc.
```

## Description

This repository contains the implementation of privacy-preserving network representations using machine learning techniques. 

- `code/` contains all primary source code.
  - `anonymization/` implements different anonymization strategies.
  - `evaluation/` includes metrics for assessing anonymization quality.
  - `real_data/` loads real-world datasets.

- `data/` holds both synthetic and real-world datasets.
- `notebooks/` stores Jupyter notebooks for visualization and debugging.
- `requirements.txt` lists dependencies for easy installation.

## Installation

To install required dependencies, run:

```bash
pip install -r requirements.txt
