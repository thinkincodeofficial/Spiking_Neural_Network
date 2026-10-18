"""
Dataset Generators for Comparative ANN vs SNN Experiments:
1. Static Digit Classification (8x8 handwritten digits).
2. Temporal Motion Classification (Moving dot Left-to-Right vs Right-to-Left).
Purely functional implementation using NumPy.
"""

import numpy as np


def _get_digit_templates():
    """Returns basic 8x8 bitmap templates for digits 0, 1, 2, 3."""
    t0 = np.array([
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0]
    ], dtype=np.float64)

    t1 = np.array([
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0]
    ], dtype=np.float64)

    t2 = np.array([
        [0, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0]
    ], dtype=np.float64)

    t3 = np.array([
        [0, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0]
    ], dtype=np.float64)

    return [t0, t1, t2, t3]


def get_static_digits_dataset(n_samples_per_digit=50, noise_std=0.15, train_split=0.8, seed=42):
    """
    Generates a synthetic 8x8 handwritten digits dataset (classes 0, 1, 2, 3).
    Returns normalized training and testing sets with flattened feature vectors (64-dim)
    and one-hot encoded targets.
    """
    rng = np.random.default_rng(seed)
    templates = _get_digit_templates()
    num_classes = len(templates)

    images = []
    labels = []

    for digit_idx, base_template in enumerate(templates):
        for _ in range(n_samples_per_digit):
            # Apply random shift (-1, 0, +1 pixels)
            shift_y = rng.choice([-1, 0, 1])
            shift_x = rng.choice([-1, 0, 1])
            shifted = np.roll(np.roll(base_template, shift_y, axis=0), shift_x, axis=1)

            # Add zero-mean Gaussian noise & random scaling
            scale = rng.uniform(0.75, 1.0)
            noise = rng.normal(0.0, noise_std, size=(8, 8))
            sample = np.clip(shifted * scale + noise, 0.0, 1.0)

            images.append(sample.flatten())
            labels.append(digit_idx)

    images = np.array(images, dtype=np.float64)
    labels = np.array(labels, dtype=np.int64)

    # One-hot encode targets
    num_samples = len(labels)
    targets_one_hot = np.zeros((num_samples, num_classes), dtype=np.float64)
    targets_one_hot[np.arange(num_samples), labels] = 1.0

    # Shuffle and split into train and test
    indices = rng.permutation(num_samples)
    split_idx = int(num_samples * train_split)

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return {
        "x_train": images[train_idx],
        "y_train": targets_one_hot[train_idx],
        "labels_train": labels[train_idx],
        "x_test": images[test_idx],
        "y_test": targets_one_hot[test_idx],
        "labels_test": labels[test_idx],
        "num_classes": num_classes,
        "input_dim": 64
    }


def get_temporal_moving_dot_dataset(n_samples=120, n_pixels=8, timesteps=16, noise_prob=0.015, train_split=0.8, seed=42):
    """
    Generates temporal moving dot motion sequences:
    Class 0: Moving Left-to-Right (pixel 0 -> 1 -> ... -> 7)
    Class 1: Moving Right-to-Left (pixel 7 -> 6 -> ... -> 0)

    Returns:
        snn_spike_trains: shape (N, timesteps, n_pixels) with binary spike events
        ann_sequences: shape (N, timesteps * n_pixels) flattened frame representation
        labels: shape (N,) integer class indices (0: L->R, 1: R->L)
        targets_one_hot: shape (N, 2)
    """
    rng = np.random.default_rng(seed)
    num_classes = 2
    samples_per_class = n_samples // 2

    snn_spike_trains = np.zeros((n_samples, timesteps, n_pixels), dtype=np.float64)
    labels = np.zeros(n_samples, dtype=np.int64)

    idx = 0
    # Class 0: Left -> Right
    for _ in range(samples_per_class):
        start_t = rng.integers(0, 3)
        dt_step = rng.integers(1, 3)  # variable speed
        for p in range(n_pixels):
            t = start_t + p * dt_step
            if t < timesteps:
                snn_spike_trains[idx, t, p] = 1.0
        # Add background stochastic noise spikes
        noise = (rng.uniform(0.0, 1.0, size=(timesteps, n_pixels)) < noise_prob).astype(np.float64)
        snn_spike_trains[idx] = np.clip(snn_spike_trains[idx] + noise, 0.0, 1.0)
        labels[idx] = 0
        idx += 1

    # Class 1: Right -> Left
    for _ in range(samples_per_class):
        start_t = rng.integers(0, 3)
        dt_step = rng.integers(1, 3)  # variable speed
        for p in range(n_pixels):
            t = start_t + p * dt_step
            ch = (n_pixels - 1) - p  # reversed channel
            if t < timesteps:
                snn_spike_trains[idx, t, ch] = 1.0
        # Add background stochastic noise spikes
        noise = (rng.uniform(0.0, 1.0, size=(timesteps, n_pixels)) < noise_prob).astype(np.float64)
        snn_spike_trains[idx] = np.clip(snn_spike_trains[idx] + noise, 0.0, 1.0)
        labels[idx] = 1
        idx += 1

    # Flatten temporal dimension for standard feedforward ANN baseline
    ann_sequences = snn_spike_trains.reshape(n_samples, timesteps * n_pixels)

    targets_one_hot = np.zeros((n_samples, num_classes), dtype=np.float64)
    targets_one_hot[np.arange(n_samples), labels] = 1.0

    # Shuffle and split
    indices = rng.permutation(n_samples)
    split_idx = int(n_samples * train_split)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return {
        "snn_train": snn_spike_trains[train_idx],
        "snn_test": snn_spike_trains[test_idx],
        "ann_train": ann_sequences[train_idx],
        "ann_test": ann_sequences[test_idx],
        "y_train": targets_one_hot[train_idx],
        "y_test": targets_one_hot[test_idx],
        "labels_train": labels[train_idx],
        "labels_test": labels[test_idx],
        "timesteps": timesteps,
        "n_pixels": n_pixels,
        "num_classes": num_classes
    }
