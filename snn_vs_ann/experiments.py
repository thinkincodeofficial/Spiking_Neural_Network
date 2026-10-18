"""
Comparative Experiments Runner:
- Demo: Biological STDP Hebbian Plasticity Pairing.
- Experiment 1: Static Image Classification (8x8 Digits).
- Experiment 2: Temporal Motion Classification (Moving Dot Direction).
"""

import time
import numpy as np

from .ann_engine import init_ann, train_ann, ann_predict
from .snn_engine import (
    init_snn_weights, rate_encode, snn_forward,
    train_snn, simulate_stdp_pairing
)


def run_stdp_experiment():
    """
    Demonstrates biological Spike-Timing-Dependent Plasticity (STDP):
    Pre-spike before Post-spike (Delta t > 0) -> Long-Term Potentiation (LTP).
    Post-spike before Pre-spike (Delta t < 0) -> Long-Term Depression (LTD).
    """
    delta_t_vals, delta_w_vals = simulate_stdp_pairing()
    return {
        "delta_t": delta_t_vals,
        "delta_w": delta_w_vals
    }


def run_static_classification_experiment(dataset, epochs_ann=35, epochs_snn=20):
    """
    Experiment 1: Static Image Classification on 8x8 Digits.
    Compares ANN vs SNN on identical inputs and architecture budget.
    """
    x_train, y_train = dataset["x_train"], dataset["y_train"]
    x_test, y_test = dataset["x_test"], dataset["y_test"]
    labels_test = dataset["labels_test"]
    num_classes = dataset["num_classes"]
    input_dim = dataset["input_dim"]
    n_hidden = 24

    # 1. Train Standard Feedforward ANN
    t0_ann = time.perf_counter()
    w1, b1, w2, b2 = init_ann(input_dim, n_hidden, num_classes, seed=42)
    w1, b1, w2, b2, ann_loss = train_ann(
        w1, b1, w2, b2, x_train, y_train, epochs=epochs_ann, lr=0.08
    )
    t_ann = time.perf_counter() - t0_ann
    ann_preds, ann_probs = ann_predict(w1, b1, w2, b2, x_test)
    ann_acc = float(np.mean(ann_preds == labels_test)) * 100.0

    # 2. Train Spiking Neural Network (SNN) with Rate Encoding
    timesteps_snn = 16
    spikes_train = rate_encode(x_train, timesteps=timesteps_snn, max_rate=0.75, seed=42)
    spikes_test = rate_encode(x_test, timesteps=timesteps_snn, max_rate=0.75, seed=43)

    t0_snn = time.perf_counter()
    w_snn = init_snn_weights(input_dim, num_classes, seed=42)
    w_snn, snn_loss = train_snn(
        w_snn, spikes_train, y_train, epochs=epochs_snn, lr=0.03
    )
    t_snn = time.perf_counter() - t0_snn

    # Evaluate SNN on Test Set
    snn_counts, s_hist, v_hist = snn_forward(w_snn, spikes_test)
    snn_preds = np.argmax(snn_counts, axis=-1)
    snn_acc = float(np.mean(snn_preds == labels_test)) * 100.0

    # Sparsity and Operations
    total_elements = s_hist.size
    active_spikes = int(np.sum(s_hist))
    snn_sparsity = 100.0 - (active_spikes / total_elements * 100.0)

    # Operations per inference sample
    ann_macs_per_sample = input_dim * n_hidden + n_hidden * num_classes
    snn_input_spikes_per_sample = np.mean(np.sum(spikes_test, axis=(0, 2)))
    snn_acs_per_sample = snn_input_spikes_per_sample * num_classes

    # Total parameters
    ann_params = (input_dim * n_hidden + n_hidden) + (n_hidden * num_classes + num_classes)
    snn_params = input_dim * num_classes

    return {
        "ann_acc": ann_acc,
        "snn_acc": snn_acc,
        "ann_train_time": t_ann,
        "snn_train_time": t_snn,
        "ann_params": ann_params,
        "snn_params": snn_params,
        "ann_macs_sample": ann_macs_per_sample,
        "snn_acs_sample": snn_acs_per_sample,
        "snn_sparsity": snn_sparsity,
        "sample_digit": x_test[0].reshape(8, 8),
        "sample_ann_probs": ann_probs[0],
        "sample_snn_spikes": s_hist[:, 0, :],
        "sample_snn_v": v_hist[:, 0, :],
        "sample_snn_input_spikes": spikes_test[:, 0, :],
        "sample_true_label": labels_test[0]
    }


def run_temporal_classification_experiment(dataset, epochs_ann=30, epochs_snn=25):
    """
    Experiment 2: Temporal Motion Classification (Moving Dot Direction).
    Class 0: Left -> Right, Class 1: Right -> Left.
    """
    snn_train, snn_test = dataset["snn_train"], dataset["snn_test"]
    ann_train, ann_test = dataset["ann_train"], dataset["ann_test"]
    y_train, y_test = dataset["y_train"], dataset["y_test"]
    labels_test = dataset["labels_test"]
    timesteps = dataset["timesteps"]
    n_pixels = dataset["n_pixels"]
    num_classes = dataset["num_classes"]

    # 1. ANN Baseline (processes flattened temporal frame representations)
    ann_input_dim = timesteps * n_pixels
    n_hidden_ann = 12
    w1, b1, w2, b2 = init_ann(ann_input_dim, n_hidden_ann, num_classes, seed=42)
    w1, b1, w2, b2, _ = train_ann(
        w1, b1, w2, b2, ann_train, y_train, epochs=epochs_ann, lr=0.08
    )
    ann_preds, _ = ann_predict(w1, b1, w2, b2, ann_test)
    ann_acc = float(np.mean(ann_preds == labels_test)) * 100.0

    # 2. SNN Model (processes event stream dynamically through time)
    # Re-order shape for SNN forward: (timesteps, num_samples, n_pixels)
    snn_train_t = np.transpose(snn_train, (1, 0, 2))
    snn_test_t = np.transpose(snn_test, (1, 0, 2))

    w_snn = init_snn_weights(n_pixels, num_classes, seed=42)
    w_snn, _ = train_snn(
        w_snn, snn_train_t, y_train, epochs=epochs_snn, lr=0.04
    )

    snn_counts, s_hist, v_hist = snn_forward(w_snn, snn_test_t, r=18.0)
    snn_preds = np.argmax(snn_counts, axis=-1)
    snn_acc = float(np.mean(snn_preds == labels_test)) * 100.0

    # Compute Latency to First Spike (Decision time)
    # s_hist shape: (timesteps, test_samples, num_classes)
    latencies = []
    for sample_idx in range(len(labels_test)):
        sample_spikes = s_hist[:, sample_idx, :]
        spike_times = np.where(np.sum(sample_spikes, axis=1) > 0)[0]
        if len(spike_times) > 0:
            latencies.append(spike_times[0] + 1)
        else:
            latencies.append(timesteps)

    snn_median_latency = float(np.median(latencies))
    ann_latency = float(timesteps)  # ANN must wait for full sequence buffer

    # Compute Operations
    ann_macs_sample = ann_input_dim * n_hidden_ann + n_hidden_ann * num_classes
    snn_active_spikes_sample = np.mean(np.sum(snn_test, axis=(1, 2)))
    snn_acs_sample = snn_active_spikes_sample * num_classes

    # Extract sample traces for Left->Right (idx 0) and Right->Left (idx 1)
    lr_idx = np.where(labels_test == 0)[0][0]
    rl_idx = np.where(labels_test == 1)[0][0]

    return {
        "ann_acc": ann_acc,
        "snn_acc": snn_acc,
        "ann_latency": ann_latency,
        "snn_latency": snn_median_latency,
        "ann_macs_sample": ann_macs_sample,
        "snn_acs_sample": snn_acs_sample,
        "lr_spikes": snn_test_t[:, lr_idx, :],
        "rl_spikes": snn_test_t[:, rl_idx, :],
        "lr_v_history": v_hist[:, lr_idx, :],
        "rl_v_history": v_hist[:, rl_idx, :],
        "learned_snn_weights": w_snn
    }
