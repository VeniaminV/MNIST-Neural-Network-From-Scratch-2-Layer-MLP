# MNIST Neural Network — From-Scratch 2-Layer MLP

A fully hand-rolled feedforward neural network trained on the MNIST handwritten digit dataset — **no PyTorch, no TensorFlow, no ML frameworks**. Just Python, NumPy, and math.

Built as a CS 445 (Machine Learning) programming assignment at Portland State University.

---

## Overview

This project implements a **2-layer fully connected neural network** (input → hidden → output) from scratch using only NumPy. It trains on 60,000 MNIST images and classifies handwritten digits (0–9) with competitive accuracy.

Everything is implemented manually:
- Forward pass with sigmoid activations
- Backpropagation
- Stochastic gradient descent with momentum
- Accuracy tracking and confusion matrix evaluation

---

## Architecture

```
Input Layer       Hidden Layer       Output Layer
(784 neurons)  →  (n neurons)    →   (10 neurons)
  pixels          sigmoid             sigmoid
                  activation          activation
```

- **Input**: 784 features (28×28 pixel values, normalized to [0, 1])
- **Hidden**: configurable size (default: 100 neurons)
- **Output**: 10 neurons — one per digit class (0–9)
- **Activation**: Sigmoid at both layers
- **Targets**: 0.9 for correct class, 0.1 for all others (avoids zero-gradient saturation)

---

## ⚙️ How It Works

### Forward Pass
```
hidden = sigmoid(W1 · x)
output = sigmoid(W2 · hidden)
```

### Backpropagation
```
error_output = (target - output) * sigmoid'(output)
error_hidden = (W2ᵀ · error_output) * sigmoid'(hidden)
```

### Weight Update (SGD + Momentum)
```
ΔW = lr * (error ⊗ activation) + momentum * ΔW_prev
W  = W + ΔW
```

---

## Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Data
Download the MNIST CSV files and place them in the same directory as the script:
- [`mnist_train.csv`](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) — 60,000 training samples
- [`mnist_test.csv`](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) — 10,000 test samples

Each CSV row: `label, pixel_0, pixel_1, ..., pixel_783`

### Run
```bash
python mnist_nn.py
```

Outputs:
- Per-epoch train/test accuracy printed to console
- `accuracy.png` — accuracy curves over training
- Confusion matrix printed at the end

---

## Experiments

The code supports three experiments by toggling a few lines:

### Experiment 1 — Hidden Layer Size
Change `hidden_size` to compare network capacity:
```python
hidden_size = 20   # underfitting
hidden_size = 50
hidden_size = 100  # default
```

### Experiment 2 — Learning Rate & Momentum
Adjust in the `train_network` call:
```python
lr=0.1, momentum=0.9   # default
lr=0.01                # slower convergence
lr=0.5                 # risk of instability
```

### Experiment 3 — Reduced Training Data
Use `balanced_subset()` to train on a fraction of data (equal samples per class):
```python
# 50% of training data
X_train, y_train = balanced_subset(X_train, y_train, 0.5)

# 25% of training data
X_train, y_train = balanced_subset(X_train, y_train, 0.25)
```

---

## 📊 Results

| Hidden Size | Train Accuracy | Test Accuracy |
|-------------|---------------|---------------|
| 20          | ~95%          | ~94%          |
| 50          | ~97%          | ~96%          |
| 100         | ~98%          | ~97%          |

*Results approximate — vary slightly between runs due to random weight initialization.*

---

## File Structure

```
mnist-nn/
├── mnist_nn.py          # Main script — network definition and training
├── mnist_train.csv      # Training data (not included, see Setup)
├── mnist_test.csv       # Test data (not included, see Setup)
├── accuracy.png         # Generated after training
└── README.md
```

---

## Key Implementation Details

| Component | Detail |
|-----------|--------|
| Weight init | `uniform(-0.05, 0.05)` — small random values |
| Activation | Sigmoid `1 / (1 + e^-x)` |
| Loss signal | `(target - output) * sigmoid'(output)` |
| Optimizer | SGD + momentum (no adaptive rates) |
| Training mode | Online (one sample at a time) |
| Evaluation | After every epoch on both train and test sets |

---

## Concepts Demonstrated

- Feedforward neural network design
- Backpropagation and gradient flow
- Stochastic gradient descent with momentum
- Sigmoid activation and vanishing gradients
- Target encoding to avoid output saturation
- Generalization vs. overfitting analysis
- Confusion matrix interpretation

---

## Built With

- **Python 3**
- **NumPy** — matrix operations and weight updates
- **Pandas** — CSV loading
- **Matplotlib** — accuracy curve plotting
- **scikit-learn** — confusion matrix

---

## Author

**Veniamin V.** — Portland State University, CS 445 Machine Learning  
[github.com/VeniaminV](https://github.com/VeniaminV)

---

*Built without ML frameworks to understand the fundamentals from the ground up.*
