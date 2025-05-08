# Feedback Control for Credit Assignment in RNNs

This repository contains the code accompanying the NeurIPS 2024 paper:

**Feedback control guides credit assignment in recurrent neural networks**  
Klara Kaleb, Barbara Feulner, Juan A. Gallego, Claudia Clopath  
[NeurIPS 2024 Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/09236f27bad623511341362f26ffcabb-Abstract-Conference.html)

---

## 🚀 Quick Start

The main steps for running a full experiment are as follows:

1. **Initial training on random reaches**  
   Run `setup_OFC_network.py` to train the model on a synthetic random reach dataset.  
   → [Code](setup_OFC_network.py)

2. **Adaptation to visuomotor rotation (VR) perturbation**  
   Run `adaptation_learning.py` to simulate fast adaptation using a feedback-driven plasticity rule.  
   → [Code](adaptation_learning.py)

---

## 📦 Requirements

The original experiments were run using:

- Python 3.7.4  
- PyTorch 1.8.1
