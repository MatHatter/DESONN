# DDESONN — Dynamic Ensemble Self-Organizing Neural Network

DDESONN is a custom R-based neural network framework for dynamic architecture design, flexible layer-wise training, and self-organizing behavior. It aims to be a modular, lightweight alternative to traditional deep learning libraries, with a focus on predictive modeling such as binary classification (e.g., heart failure prediction).

> ⚠️ **Note**: This codebase is an early-stage project and currently unpolished. Structure and naming will be cleaned up in future updates. Contributions and feedback are welcome.

---

## 🔧 How It Works

The system is composed of two main scripts:
- `DDESONN.R`: Defines the core neural network structure and training logic.
- `TestDESONN.R`: Loads sample data, initializes the network, and runs training/testing.

Key features:
- Customizable layer sizes, activations, and weight initializations
- Manual backpropagation and update logic
- Binary output using thresholded ReLU
- Per-sample error tracking
- No external machine learning packages required
