# Deep Learning: A Research Compendium
### *From First Principles to Modern Practice*

**Author:** Kushal Kharel  
**Status:** ![Active](https://img.shields.io/badge/status-actively%20maintained-brightgreen) — continuously expanded as a single source of truth for deep learning theory and implementation.

---

> *"The purpose of this repository is not to be a tutorial, but a rigorous reference — one that bridges the mathematical foundations of machine learning with working, reproducible code. Each notebook is self-contained and builds directly on the last."*

---

## Overview

This compendium is a structured, research-oriented deep dive into the theoretical and computational foundations of deep learning. Each notebook is written from first principles — deriving the mathematics, implementing from scratch, validating against established libraries, and building intuition through visualization.

The collection is organized as a **progressive curriculum**. Notebooks are numbered to reflect the recommended reading order. Concepts introduced early (e.g., gradient descent, activation saturation) are deliberately referenced and extended in later entries.

---

## Repository Structure

```
deep-learning-compendium/
│
├── 1_Linear_Regression.ipynb
├── 2_Logistic_Regression.ipynb
├── 3_Gradient_Descent.ipynb
├── 4_Activation_Functions.ipynb
├── 5_Weight_Initialization.ipynb
├── 6_Optimization_and_Learning.ipynb
├── 7_Learning_Rate_Schedules.ipynb
│
└── README.md
```

> **Navigation note:** Follow the numbered order on a first pass. The concepts chain tightly — notebook 5 (Weight Initialization) explicitly requires notebook 4 (Activation Functions), and notebook 7 (Learning Rate Schedules) synthesizes all prior material.

---

## Curriculum Map

| # | Notebook | Core Concepts | Key Implementations |
|---|----------|---------------|---------------------|
| 1 | [Linear Regression](#1-linear-regression) | OLS, MSE, gradient descent vs. closed form | `LinearRegression` from scratch, SciPy optimization, multivariate extension |
| 2 | [Logistic Regression](#2-logistic-regression-from-first-principles) | MLE, sigmoid, gradient ascent, calibration | `LogisticRegressionGA` class, ROC-AUC, ECE, softmax extension |
| 3 | [Gradient Descent](#3-gradient-descent) | Batch, SGD, Mini-Batch; vectorized updates | Numerical differentiation, tangent line approximation, all three GD variants |
| 4 | [Activation Functions](#4-activation-functions) | Step → Sigmoid → Tanh → ReLU and beyond | Vanishing gradient demonstration, weight update zig-zag simulation |
| 5 | [Weight Initialization](#5-weight-initialization) | Variance propagation, symmetry breaking | Zero, Random, Xavier/Glorot, He/Kaiming — all verified analytically and empirically |
| 6 | [Optimization & Learning](#6-the-geometry-of-optimization) | Loss manifolds, optimizer taxonomy | Batch GD, SGD, Mini-Batch GD, Adam — loss surface visualization in 3D |
| 7 | [Learning Rate Schedules](#7-learning-rate-schedules--optimizer-dynamics) | LR schedule theory, Rosenbrock benchmark | Constant, Step Decay, Exponential, Cosine Annealing, Warmup + Cosine |

---

## Notebook Descriptions

---

### 1. Linear Regression

**File:** `1_Linear_Regression.ipynb`

**Abstract:** This notebook establishes the computational and statistical bedrock of supervised learning. We begin with simple linear regression on the `tips` dataset, then transition to the gradient descent perspective that motivates all subsequent optimization work.

**Topics Covered:**

- What regression is and when to use it
- The OLS closed-form solution: $\theta = (X^TX)^{-1}X^Ty$ and its $\mathcal{O}(n^3)$ computational cost
- The Mean Squared Error cost function and its derivation:
$$J(\theta) = \frac{1}{n} \sum_{i=1}^n (y_i - \theta x_i)^2$$
- Residual analysis and homoscedasticity checking
- Why gradient descent is preferred over matrix inversion for large-scale problems
- Extension to the multivariate case and the effect of additional features on $R^2$
- SciPy-based numerical optimization as a bridge to neural network training

**Dependencies:** `numpy`, `pandas`, `seaborn`, `plotly`, `scikit-learn`, `scipy`

---

### 2. Logistic Regression: From First Principles

**File:** `2_Logistic_Regression.ipynb`

**Abstract:** Despite its name, logistic regression is a probabilistic classification model. This notebook derives the full maximum likelihood estimation framework, implements gradient ascent from scratch, and rigorously validates the implementation against scikit-learn using log-likelihood equivalence and probability agreement tests.

**Topics Covered:**

- From linear output to probability: the sigmoid (logistic) function
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- Log-likelihood derivation and the gradient ascent update rule:
$$\theta := \theta + \alpha \sum_{i=1}^n (y_i - h_\theta(x)) x_j$$
- Numerically stable implementation (clipping, log-sum trick)
- The importance of standardization for gradient-based methods
- Evaluation beyond accuracy: ROC-AUC, Brier Score, calibration curves
- Expected Calibration Error (ECE): measuring probability reliability
- Comparison against scikit-learn's L-BFGS solver — parameter agreement test
- Why exact parameter convergence is not guaranteed for separable data (MLE non-existence)
- Extension to **Softmax Regression** for K-class classification

**Dependencies:** `numpy`, `matplotlib`, `scikit-learn`

---

### 3. Gradient Descent

**File:** `3_Gradient_Descent.ipynb`

**Abstract:** Gradient descent is the engine of all deep learning. This notebook builds the algorithm from first intuition (the "foggy mountain" analogy) through rigorous derivation, then implements and compares all three primary variants.

**Topics Covered:**

- The optimization objective: $\min_{\theta \in \mathbb{R}^d} J(\theta)$
- Brute-force minimization and its three fundamental flaws
- Numerical differentiation: $f'(x) \approx \frac{f(x+h) - f(x)}{h}$
- Tangent line approximation and error analysis
- The vectorized gradient update for linear regression:
$$\theta := \theta - \frac{\alpha}{n} X^T(X\theta - Y)$$
- **Batch Gradient Descent:** stable, exact, computationally expensive
- **Stochastic Gradient Descent (SGD):** one sample per update, noisy but fast
- **Mini-Batch Gradient Descent:** the practical middle ground used in all modern frameworks
- Learning rate sensitivity and the step-size dilemma
- Convergence criteria and early stopping

**Dependencies:** `numpy`, `sympy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`

---

### 4. Activation Functions

**File:** `4_Activation_Functions.ipynb`

**Abstract:** Activation functions are what separate neural networks from linear models. This notebook traces the historical arc from the Heaviside step function to ReLU, deriving the mathematical motivation for each transition and demonstrating the vanishing gradient problem empirically.

**Topics Covered:**

- Why nonlinearity is essential: composition of linear functions is linear
- **Step Function:** the McCulloch-Pitts neuron, zero-gradient problem
- **Sigmoid:** differentiability, probability interpretation, vanishing gradients for large $|x|$
- The "always positive" gradient problem and its effect on weight updates:
$$\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a} \cdot \sigma'(z) \cdot x_i$$
- **Tanh:** zero-centered outputs, stronger gradient signal (max gradient = 1 vs. 0.25)
- Zig-zag gradient dynamics simulation: why zero-centered inputs enable diagonal weight updates
- The vanishing gradient problem in deep networks — empirical demonstration
- **ReLU (Rectified Linear Unit):** $f(x) = \max(0, x)$ — constant gradient for $x > 0$, modern gold standard
- Dying ReLU problem and its variants (Leaky ReLU, ELU, GELU)

**Dependencies:** `numpy`, `torch`, `scipy`, `matplotlib`, `seaborn`

---

### 5. Weight Initialization

**File:** `5_Weight_Initialization.ipynb`

> *Prerequisite: Notebook 4 (Activation Functions)*

**Abstract:** Weight initialization is the silent determinant of whether a deep network trains at all. This notebook develops the variance propagation framework from scratch and rigorously derives Xavier and He initialization, showing why each is matched to a specific activation function.

**Topics Covered:**

- The symmetry trap: why zero initialization causes identical gradient updates for all neurons in a layer
- Variance propagation through a layer: $\text{Var}(y) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$
- The Goldilocks dilemma: constant-scale random initialization collapses (small $\sigma$) or explodes (large $\sigma$) with depth
- **Xavier (Glorot) Initialization** — for symmetric activations (Tanh, Sigmoid):
$$w \sim \mathcal{U}\!\left[-\frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}\right]$$
- Why Xavier drove the field from Sigmoid to Tanh in the early 2010s
- **He (Kaiming) Initialization** — for ReLU, which kills 50% of variance:
$$w \sim \mathcal{N}\!\left(0, \frac{2}{n_{in}}\right)$$
- Empirical validation: variance stability across 50 layers with each scheme
- Gain factors in PyTorch and how to apply them correctly

**Dependencies:** `torch`, `numpy`, `matplotlib`

---

### 6. The Geometry of Optimization

**File:** `6_Optimization_and_Learning.ipynb`

**Abstract:** This notebook elevates the optimization discussion from scalar loss curves to the full geometric picture — loss manifolds, parameter-space trajectories, and the spectrum from Batch GD to Adam. Each optimizer is visualized on a 3D regression surface, and its parameters are recovered back to real-world units.

**Topics Covered:**

- The loss manifold as a convex quadratic surface for linear regression with MSE
- The Normal Equation as ground truth: $w = (X^TX)^{-1}X^Ty$
- Standardization and the centered universe: why bias collapses to zero in standardized space, and how to recover real-world coefficients
- **Batch Gradient Descent:** full convergence proof path, 3D surface visualization
- **Stochastic Gradient Descent:** erratic trajectory around minimum as a feature, not a bug — implicit regularization via noise
- **Mini-Batch Gradient Descent:** noise smoothing, generalization benefit from residual variance
- Learning rate decay: balancing exploration with convergence precision
- **Adam Optimizer:** moment estimation, bias correction, adaptive per-parameter step sizes
- Side-by-side MSE comparison across all optimizers

**Dependencies:** `numpy`, `matplotlib`

---

### 7. Learning Rate Schedules & Optimizer Dynamics

**File:** `7_Learning_Rate_Schedules.ipynb`

**Abstract:** A research-level comparative study of six canonical learning rate schedules evaluated on the Rosenbrock benchmark — a high-condition-number optimization landscape that exposes the failure modes invisible on simple convex problems. Implemented in TensorFlow with XLA JIT compilation.

**Topics Covered:**

- The Rosenbrock function: $f(x,y) = (1-x)^2 + 100(y-x^2)^2$, condition number $\kappa \approx 2700$
- Why a high condition number breaks naive gradient descent (optimal step in one direction overshoots the other by $\sim$2700×)
- Gradient field analysis and the banana valley problem
- **Constant LR:** convergence floor theorem, noise floor $\alpha L \sigma^2$
- **Step Decay:** $\alpha(t) = \alpha_0 \cdot \gamma^{\lfloor t/s \rfloor}$ — VGG/ResNet protocol, abrupt discontinuity hazard
- **Exponential Decay:** $\alpha(t) = \alpha_0 \cdot e^{-\lambda t/T}$ — linear convergence for strongly convex problems
- **Cosine Annealing** (Loshchilov & Hutter, 2016): smooth endpoints, maximum LR change at midpoint, superior to exponential for the first ~40% of training
- **Linear Warmup + Cosine Decay:** the canonical Transformer schedule (GPT, LLaMA, BERT, ViT) — why Adam's second-moment initialization makes warmup essential
- Full empirical benchmark: convergence speed, final loss, trajectory visualization

**Dependencies:** `numpy`, `matplotlib`, `tensorflow`, `tqdm`

---

## Dependencies

All notebooks are self-contained. The full environment can be installed with:

```bash
pip install numpy pandas matplotlib seaborn plotly scipy scikit-learn torch tensorflow tqdm
```

For GPU-accelerated experiments in notebook 7, a CUDA-compatible TensorFlow installation is recommended but not required.

---

## Design Philosophy

This collection adheres to three principles:

1. **Derivation before implementation.** Every algorithm is derived mathematically before a line of code is written. Equations are not decorative — they determine the implementation.

2. **Validation over trust.** Each custom implementation is benchmarked against an established reference (scikit-learn, PyTorch, TensorFlow). Numerical agreement is verified explicitly.

3. **Visualization as proof.** Abstract claims (vanishing gradients, zig-zag convergence, variance collapse) are demonstrated empirically through code and plots, not just asserted.

---

## Roadmap

This is a living repository. Planned additions follow the same numbered convention and build directly on the existing foundations:

---

## Citation

If you find this compendium useful in your work or teaching, please cite:

```bibtex
@misc{kharel_dl_compendium,
  author       = {Kushal Kharel},
  title        = {Deep Learning: A Research Compendium},
  year         = {2026},
  howpublished = {\url{https://github.com/your-username/deep-learning-compendium}},
  note         = {A progressive, first-principles treatment of deep learning theory and implementation}
}
```

---

## License

This repository is released for educational and research use. Please attribute the author if you adapt or reproduce any material.

---

<p align="center">
  <i>Built with rigor. Maintained with care.</i><br>
  <b>Kushal Kharel</b>
</p>
