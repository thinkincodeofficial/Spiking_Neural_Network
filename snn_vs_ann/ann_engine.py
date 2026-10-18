"""
Pure Functional Artificial Neural Network (ANN) Engine.
Implements multi-layer feedforward classification with Backpropagation and Softmax Cross-Entropy.
"""

import numpy as np


def init_ann(n_in, n_hidden, n_out, seed=42):
    """Initializes Xavier uniform weights and zero biases for 2-layer classifier."""
    rng = np.random.default_rng(seed)
    lim1 = np.sqrt(6.0 / (n_in + n_hidden))
    lim2 = np.sqrt(6.0 / (n_hidden + n_out))

    w1 = rng.uniform(-lim1, lim1, size=(n_in, n_hidden)).astype(np.float64)
    b1 = np.zeros(n_hidden, dtype=np.float64)
    w2 = rng.uniform(-lim2, lim2, size=(n_hidden, n_out)).astype(np.float64)
    b2 = np.zeros(n_out, dtype=np.float64)

    return w1, b1, w2, b2


def relu(z):
    """Element-wise Rectified Linear Unit activation."""
    return np.maximum(0.0, z)


def relu_derivative(z):
    """Derivative of ReLU function."""
    return (z > 0.0).astype(np.float64)


def softmax(z):
    """Numerically stable softmax probability distribution across classes."""
    # z shape: (batch, n_out) or (n_out,)
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def cross_entropy_loss(probs, targets):
    """Computes mean cross-entropy loss over batch."""
    eps = 1e-12
    return -np.mean(np.sum(targets * np.log(probs + eps), axis=-1))


def ann_forward(w1, b1, w2, b2, x):
    """
    Feedforward pass through 2-layer network:
    z1 = X @ W1 + b1, a1 = ReLU(z1)
    z2 = a1 @ W2 + b2, probs = Softmax(z2)
    """
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    probs = softmax(z2)
    cache = (x, z1, a1, z2, probs)
    return probs, cache


def ann_backward(w2, cache, targets):
    """
    Reverse-mode automatic differentiation computing exact analytical gradients
    for Cross-Entropy loss with Softmax output.
    """
    x, z1, a1, _, probs = cache
    batch_size = x.shape[0] if x.ndim > 1 else 1

    # Softmax + Cross-Entropy derivative: (probs - targets)
    dz2 = (probs - targets) / batch_size
    dw2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)

    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * relu_derivative(z1)
    dw1 = np.dot(x.T, dz1)
    db1 = np.sum(dz1, axis=0)

    return dw1, db1, dw2, db2


def ann_train_step(w1, b1, w2, b2, x_batch, y_batch, lr=0.05):
    """Executes single gradient descent step on a batch of training examples."""
    probs, cache = ann_forward(w1, b1, w2, b2, x_batch)
    loss = cross_entropy_loss(probs, y_batch)
    dw1, db1, dw2, db2 = ann_backward(w2, cache, y_batch)

    w1_next = w1 - lr * dw1
    b1_next = b1 - lr * db1
    w2_next = w2 - lr * dw2
    b2_next = b2 - lr * db2

    return w1_next, b1_next, w2_next, b2_next, loss


def train_ann(w1, b1, w2, b2, x_train, y_train, epochs=40, lr=0.08, batch_size=16):
    """Trains the feedforward network over multiple epochs using mini-batch SGD."""
    num_samples = x_train.shape[0]
    w1_curr, b1_curr, w2_curr, b2_curr = w1.copy(), b1.copy(), w2.copy(), b2.copy()
    loss_history = []

    for _ in range(epochs):
        indices = np.random.permutation(num_samples)
        epoch_losses = []
        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            w1_curr, b1_curr, w2_curr, b2_curr, loss = ann_train_step(
                w1_curr, b1_curr, w2_curr, b2_curr,
                x_train[batch_idx], y_train[batch_idx], lr=lr
            )
            epoch_losses.append(loss)
        loss_history.append(float(np.mean(epoch_losses)))

    return w1_curr, b1_curr, w2_curr, b2_curr, loss_history


def ann_predict(w1, b1, w2, b2, x):
    """Predicts class labels (argmax) and class probabilities."""
    probs, _ = ann_forward(w1, b1, w2, b2, x)
    predictions = np.argmax(probs, axis=-1)
    return predictions, probs
