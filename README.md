# MNIST Neural Networks from Scratch (NumPy)

Handwritten-digit classification on MNIST using neural networks built with **only NumPy**: forward pass, cross-entropy loss, backpropagation, and gradient descent all written by hand, with no deep learning framework.

Three models are built up step by step, each improving on the last:

| # | Model | Optimizer | Test accuracy |
|---|-------|-----------|---------------|
| 1 | Single layer (softmax regression, no hidden layer) | Gradient descent | **90.95%** |
| 2 | One hidden layer (128 units, ReLU) | Gradient descent | **92.45%** |
| 3 | One hidden layer (128 units, ReLU) | Gradient descent + momentum (β = 0.9) | **94.77%** |

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/): 60,000 training and 10,000 test images of 28×28 grayscale digits.

- Images are flattened to 784-dim vectors and scaled to [0, 1].
- Data is stored column-wise: `X` has shape `(784, m)`.
- Labels are one-hot encoded: `Y` has shape `(10, m)`.
- The notebook downloads the dataset automatically from the CVDF mirror, so no manual download is needed.

## Models

### 1. Single-layer network (no hidden layer)

```
x (784) → Z = Wx + b → softmax → ŷ (10)
```

- **Parameters:** `W (10, 784)`, `b (10, 1)`
- **Loss:** cross-entropy
- **Hyperparameters:** learning rate 0.1, 1000 iterations, full-batch (60,000)
- **Gradients:** `dZ = A − Y`, `dW = (1/m) · dZ · Xᵀ`, `db = (1/m) · Σ dZ`

### 2. One hidden layer (no momentum)

```
x (784) → Z1 = W1x + b1 → ReLU → Z2 = W2·A1 + b2 → softmax → ŷ (10)
```

- **Parameters:** `W1 (128, 784)`, `b1 (128, 1)`, `W2 (10, 128)`, `b2 (10, 1)`
- **Activations:** ReLU (hidden), softmax (output)
- **Hyperparameters:** learning rate 0.01, 1000 iterations, batch size 60,000 (full-batch)

### 3. One hidden layer with momentum

Same architecture and hyperparameters as model 2, with a momentum term added to the parameter updates:

```
v = β·v + (1 − β)·grad
θ = θ − lr·v          (β = 0.9)
```

Momentum smooths the updates and accelerates convergence, which accounts for the jump from 92.45% to 94.77% over the same 1000 iterations.

## Results

| Model | Test accuracy |
|-------|---------------|
| Single layer | 90.95% |
| Hidden layer, no momentum | 92.45% |
| Hidden layer + momentum (β = 0.9) | 94.77% |

Each notebook also plots the loss curve over iterations and shows a grid of random test images with their predicted labels.

## Repository contents

| File | Description |
|------|-------------|
| `ANN.ipynb` | Single-layer network: training, loss curve, test accuracy, sample predictions |
| `ANN_.ipynb` | Earlier version of the single-layer notebook (without per-iteration accuracy tracking) |
| `ANN.md` | Experiment notes covering all three models, with plots |

## Running it

```bash
pip install numpy matplotlib jupyter
jupyter notebook ANN.ipynb
```

Run all cells top to bottom. Training uses full-batch gradient descent on 60,000 samples for 1000 iterations, which takes a few minutes on a laptop CPU.

## Ideas for next steps

- Mini-batch / stochastic gradient descent
- Additional hidden layers and other optimizers (RMSprop, Adam)
- Learning-rate scheduling and weight-initialization tweaks (He init for ReLU)
- Regularization (L2, dropout) and a confusion matrix to see which digits get mixed up
