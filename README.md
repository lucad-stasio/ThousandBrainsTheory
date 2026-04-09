# ThousandBrainsTheory
The Thousand Brains Theory by Jeff Hawkins is one that has fundamental consequences about how the human mind works, and touches on what it means to be a conscious entity. This repository shows the code used to compare a CNN to a TBT-Inspired Neural Net in the MNIST Character Recognition challenge. 

# Thousand Brains Theory (TBT) in PyTorch: A Multi-Agent Consensus Architecture

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Cognitive Science](https://img.shields.io/badge/Cognitive_Science-Neural_Architecture-blue?style=for-the-badge)

## Overview
This repository translates the biological principles of Jeff Hawkins' **Thousand Brains Theory (TBT)** into a functional computational framework. It tests the hypothesis that a decentralized, multi-agent consensus network can achieve standard baseline accuracy using a fraction of the computational weight of a traditional monolithic architecture (CNN), while demonstrating unique failure states under structural occlusion.

Modern machine learning relies heavily on hierarchical processing (CNNs), which pass data through a single, massive funnel. The human neocortex operates differently, relying on thousands of independent "cortical columns" that view fractions of sensory input, build 3D reference frames, and vote on a final consensus.

This project implements a biologically-constrained neural network to test these principles in a 2D environment.

## The Biological Model: NMDA Coincidence Detection
A naive multi-agent system suffers from the "Binding Problem"—treating spatial coordinates as disconnected sensory noise. To solve this, this architecture implements a computational analog of **NMDA Coincidence Detection** found in pyramidal neurons. 

Inputs are separated into two distinct streams:
1. **Proximal Dendrites:** Process the raw sensory pixels (The "What").
2. **Basal Dendrites:** Process the spatial coordinates (The "Where").

By mathematically gating the sensory stream with the spatial context stream, the independent agents build spatially-bound local hypotheses. A Confidence-Weighted consensus mechanism (analogous to Lateral Inhibition) allows highly confident agents to mathematically silence agents looking at pure noise.

## Architecture Comparison
We benchmarked the Spatially Bound TBT network against a standard Convolutional Neural Network (CNN) monolith.

* **The CNN Monolith:** A standard global 2D grid processor utilizing Convolutional and Max Pooling layers.
* **The TBT Multi-Agent Network:** The visual environment is divided into 16 discrete sectors. 16 independent neural networks act as "mobile sensors," scanning their designated sectors and outputting local hypotheses based on specific `(X,Y)` coordinates before reaching a global consensus.

## Experimental Results

### 1. Computational Efficiency
The multi-agent network achieved near-parity baseline accuracy using less than a third of the active parameters.
* **CNN Monolith:** 206,922 parameters
* **TBT Multi-Agent:** 63,952 parameters (~89% accuracy)

### 2. The Occlusion Protocol (Damage Resilience)
When 25% of the visual space was violently occluded:
* The CNN salvaged a **77%** accuracy (due to Max Pooling salvaging global 2D texture statistics).
* The TBT network degraded gracefully to **58%**. 

**Analysis:** The occlusion test reveals the fundamental limitation of testing a 3D sensorimotor theory in a 2D environment. While the CNN excels at 2D statistical pattern matching, the TBT agents attempt to build physical 3D reference frames. When a 2D object is occluded, the physically bound agents are completely blinded, whereas the statistical monolith simply pools the surviving texture. 

## Repository Structure

```text
├── models/
│   ├── cnn_monolith.py       # Baseline hierarchical CNN architecture
│   └── tbt_multi_agent.py    # Decentralized architecture with coincidence detection
├── notebooks/
│   ├── 01_Architecture_Setup.ipynb
│   ├── 02_Training_and_Evaluation.ipynb
│   └── 03_Occlusion_Testing.ipynb
├── utils/
│   ├── data_slicer.py        # Environment zoning and coordinate mapping
│   └── visualizer.py         # Matplotlib learning curve generation
├── requirements.txt
└── README.md
```

Getting Started
Clone the repository:
```text
Bash
git clone [https://github.com/yourusername/TBT-PyTorch-Architecture.git](https://github.com/yourusername/TBT-PyTorch-Architecture.git)
cd TBT-PyTorch-Architecture
```

Install the required dependencies:
```text
Bash
pip install -r requirements.txt
```
Run the primary evaluation notebook:
```text
Navigate to /notebooks/02_Training_and_Evaluation.ipynb to view the training loops, or run the complete pipeline locally.
```

Contact & Research
Luca D'stasio Undergraduate Researcher | Cognitive Science & Philosophy
University of Michigan
