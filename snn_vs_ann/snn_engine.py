"""
Biological Spiking Neural Network (SNN) Engine.
Pure functional implementation in NumPy covering:
1. Leaky Integrate-and-Fire (LIF) Euler dynamics.
2. Rate-based Poisson spike encoding for static inputs.
3. Supervised classification via BPTT with Fast-Sigmoid Surrogate Gradients.
4. Spike-Timing-Dependent Plasticity (STDP) Hebbian learning and pairing protocol.
"""

import numpy as np


def init_snn_weights(n_in, n_out, seed=42):
    """Initializes synaptic weights for a spiking layer using uniform distribution."""
    rng = np.random.default_rng(seed)
    limit = np.sqrt(2.0 / (n_in + n_out))
    return rng.uniform(0.1, limit, size=(n_in, n_out)).astype(np.float64)


def rate_encode(images, timesteps=16, max_rate=0.75, seed=42):
    """
    Converts static continuous pixel values in [0, 1] to Poisson binary spike trains.
    Output shape: (timesteps, batch_size, n_in)
    """
    rng = np.random.default_rng(seed)
    batch_size, n_in = images.shape
    prob = np.clip(images * max_rate, 0.0, 1.0)
    rand_vals = rng.uniform(0.0, 1.0, size=(timesteps, batch_size, n_in))
    return (rand_vals < prob[np.newaxis, :, :]).astype(np.float64)


def lif_step(v_membrane, i_syn, v_rest=-70.0, v_reset=-70.0, v_thresh=-55.0, tau_m=20.0, dt=1.0, r=10.0):
    """
    Simulates one discrete Euler integration time step for Leaky Integrate-and-Fire (LIF):
    tau_m * (dv / dt) = -(v - v_rest) + R * I(t)
    """
    dv = (dt / tau_m) * (-(v_membrane - v_rest) + r * i_syn)
    v_tentative = v_membrane + dv
    spikes = (v_tentative >= v_thresh).astype(np.float64)
    v_next = np.where(spikes > 0.0, v_reset, v_tentative)
    return v_next, spikes


def surrogate_derivative(v, v_thresh=-55.0, gamma=0.3):
    """Fast-sigmoid surrogate gradient for non-differentiable Heaviside spike function."""
    return 1.0 / (1.0 + gamma * np.abs(v - v_thresh)) ** 2


def snn_forward(weights, spike_train, v_rest=-70.0, v_reset=-70.0, v_thresh=-55.0, tau_m=20.0, dt=1.0, r=10.0):
    """
    Forward simulation of single-layer LIF network over an entire temporal spike train.
    Input spike_train shape: (timesteps, batch_size, n_in) or (timesteps, n_in)
    """
    if spike_train.ndim == 2:
        # Convert (T, n_in) to (T, 1, n_in)
        spike_train = spike_train[:, np.newaxis, :]
        squeeze_batch = True
    else:
        squeeze_batch = False

    timesteps, batch_size, _ = spike_train.shape
    n_out = weights.shape[1]

    v_history = np.zeros((timesteps, batch_size, n_out), dtype=np.float64)
    s_history = np.zeros((timesteps, batch_size, n_out), dtype=np.float64)

    v_curr = np.full((batch_size, n_out), v_rest, dtype=np.float64)

    for t in range(timesteps):
        i_syn = np.dot(spike_train[t], weights)  # (batch, n_out)
        v_curr, spikes = lif_step(v_curr, i_syn, v_rest, v_reset, v_thresh, tau_m, dt, r)
        v_history[t] = v_curr
        s_history[t] = spikes

    total_spike_counts = np.sum(s_history, axis=0)  # (batch, n_out)

    if squeeze_batch:
        return total_spike_counts[0], s_history[:, 0, :], v_history[:, 0, :]
    return total_spike_counts, s_history, v_history


def snn_train_step(weights, spike_train, targets_one_hot, lr=0.01,
                   v_rest=-70.0, v_reset=-70.0, v_thresh=-55.0, tau_m=20.0, dt=1.0, r=10.0):
    """
    Supervised Backpropagation Through Time (BPTT) with surrogate gradients for LIF layer.
    Optimizes Softmax Cross-Entropy loss on the accumulated spike rate.
    """
    timesteps, batch_size, _ = spike_train.shape
    total_spikes, s_history, v_history = snn_forward(
        weights, spike_train, v_rest, v_reset, v_thresh, tau_m, dt, r
    )

    # Softmax probabilities over spike counts
    probs = np.exp(total_spikes - np.max(total_spikes, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)

    loss = -np.mean(np.sum(targets_one_hot * np.log(probs + 1e-12), axis=-1))

    # Gradient of loss w.r.t total accumulated spikes
    d_total_spikes = (probs - targets_one_hot) / batch_size  # (batch, n_out)

    # Accumulate gradient w.r.t weights across time using surrogate derivative
    dw = np.zeros_like(weights)
    leak_factor = 1.0 - (dt / tau_m)

    for t in range(timesteps - 1, -1, -1):
        # Surrogate derivative at threshold
        surr_grad = surrogate_derivative(v_history[t], v_thresh)
        d_v = d_total_spikes * surr_grad * (r * dt / tau_m)
        dw += np.dot(spike_train[t].T, d_v)

    # Parameter update
    weights_next = np.clip(weights - lr * dw, 0.0, 3.0)
    return weights_next, loss


def train_snn(weights, spike_train_dataset, y_train, epochs=25, lr=0.02, batch_size=16):
    """Trains the SNN via surrogate gradient BPTT across multiple epochs."""
    timesteps, num_samples, n_in = spike_train_dataset.shape
    w_curr = weights.copy()
    loss_history = []

    for _ in range(epochs):
        indices = np.random.permutation(num_samples)
        epoch_losses = []
        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            batch_spikes = spike_train_dataset[:, batch_idx, :]
            batch_targets = y_train[batch_idx]
            w_curr, loss = snn_train_step(w_curr, batch_spikes, batch_targets, lr=lr)
            epoch_losses.append(loss)
        loss_history.append(float(np.mean(epoch_losses)))

    return w_curr, loss_history


def stdp_step(weights, s_pre, s_post, trace_pre, trace_post,
              tau_plus=20.0, tau_minus=20.0, a_plus=0.035, a_minus=0.030,
              w_min=0.0, w_max=1.0, dt=1.0):
    """
    Local Spike-Timing-Dependent Plasticity (STDP) update with exponential traces:
    LTP: pre spikes before post (Delta t > 0) -> potentiate
    LTD: pre spikes after post (Delta t < 0) -> depress
    """
    trace_pre_next = trace_pre * np.exp(-dt / tau_plus) + s_pre
    trace_post_next = trace_post * np.exp(-dt / tau_minus) + s_post

    dw_ltp = a_plus * np.outer(s_post, trace_pre_next)
    dw_ltd = -a_minus * np.outer(trace_post_next, s_pre)

    weights_next = np.clip(weights + dw_ltp + dw_ltd, w_min, w_max)
    return weights_next, trace_pre_next, trace_post_next


def simulate_stdp_pairing(delta_t_range=None, tau_plus=20.0, tau_minus=20.0, a_plus=0.04, a_minus=0.035):
    """
    Simulates canonical neuroscience STDP pairing protocol across different time lags Delta t:
    Delta t = t_post - t_pre.
    Returns delta_t array and corresponding delta_w values demonstrating biological curve.
    """
    if delta_t_range is None:
        delta_t_range = np.linspace(-60, 60, 121)

    delta_w = np.zeros_like(delta_t_range, dtype=np.float64)

    for i, dt_val in enumerate(delta_t_range):
        if dt_val > 0:
            # Pre before Post: LTP
            delta_w[i] = a_plus * np.exp(-dt_val / tau_plus)
        elif dt_val < 0:
            # Post before Pre: LTD
            delta_w[i] = -a_minus * np.exp(dt_val / tau_minus)
        else:
            delta_w[i] = 0.0

    return delta_t_range, delta_w
