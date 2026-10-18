"""
Visualization Suite for Comparative Experiments:
1. STDP Pairing Demonstration (LTP & LTD biological curve).
2. Experiment 1: Static Digit Classification (ANN activations vs SNN membrane waveforms).
3. Experiment 2: Temporal Motion Classification (Moving dot raster & decision latency).
4. Master Comparison Dashboard.
"""

import numpy as np
import matplotlib.pyplot as plt


# Dark aesthetic theme configuration
THEME_BG = "#0f172a"
PANEL_BG = "#1e293b"
GRID_COLOR = "#334155"
TEXT_COLOR = "#f8fafc"
ACCENT_BLUE = "#38bdf8"
ACCENT_GREEN = "#4ade80"
ACCENT_ORANGE = "#fb923c"
ACCENT_RED = "#f43f5e"
ACCENT_PURPLE = "#c084fc"


def plot_stdp_pairing_curve(stdp_data, save_path="stdp_demonstration.png"):
    """Visualizes the biological STDP curve: Delta t vs Delta w."""
    delta_t = stdp_data["delta_t"]
    delta_w = stdp_data["delta_w"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor(THEME_BG)
    ax.set_facecolor(PANEL_BG)

    # Plot LTP (positive dt) and LTD (negative dt)
    ltp_mask = delta_t >= 0
    ltd_mask = delta_t <= 0

    ax.plot(delta_t[ltp_mask], delta_w[ltp_mask], color=ACCENT_GREEN, linewidth=2.5, label=r"Long-Term Potentiation (LTP: $\Delta t > 0$)")
    ax.fill_between(delta_t[ltp_mask], delta_w[ltp_mask], color=ACCENT_GREEN, alpha=0.2)

    ax.plot(delta_t[ltd_mask], delta_w[ltd_mask], color=ACCENT_RED, linewidth=2.5, label=r"Long-Term Depression (LTD: $\Delta t < 0$)")
    ax.fill_between(delta_t[ltd_mask], delta_w[ltd_mask], color=ACCENT_RED, alpha=0.2)

    ax.axhline(0, color=TEXT_COLOR, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(0, color=TEXT_COLOR, linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_title("Spike-Timing-Dependent Plasticity (STDP) Learning Rule", color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(r"Spike Timing Difference: $\Delta t = t_{post} - t_{pre}$ (ms)", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel(r"Synaptic Weight Change: $\Delta w$", color=TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    ax.annotate("Pre fires BEFORE Post\n-> Synapse Strengthens (LTP)", xy=(15, 0.02), xytext=(22, 0.03),
                arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.5), color=TEXT_COLOR, fontsize=9)
    ax.annotate("Pre fires AFTER Post\n-> Synapse Weakens (LTD)", xy=(-15, -0.018), xytext=(-55, -0.03),
                arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.5), color=TEXT_COLOR, fontsize=9)

    ax.legend(loc="upper right", facecolor=THEME_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return save_path


def plot_experiment_1_static(exp1_data, save_path="experiment_1_static.png"):
    """Visualizes Experiment 1: Static 8x8 digit -> ANN outputs vs SNN spike/membrane waveforms."""
    sample_digit = exp1_data["sample_digit"]
    ann_probs = exp1_data["sample_ann_probs"]
    snn_v = exp1_data["sample_snn_v"]
    snn_spikes = exp1_data["sample_snn_spikes"]
    snn_in_spikes = exp1_data["sample_snn_input_spikes"]
    true_label = exp1_data["sample_true_label"]

    timesteps, num_classes = snn_v.shape
    time_axis = np.arange(timesteps)

    fig = plt.figure(figsize=(15, 8), dpi=150)
    fig.patch.set_facecolor(THEME_BG)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28, left=0.06, right=0.96, top=0.90, bottom=0.10)

    # 1. Input 8x8 Digit
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.set_facecolor(PANEL_BG)
    cax = ax_img.imshow(sample_digit, cmap="viridis", interpolation="nearest")
    ax_img.set_title(f"Input Digit (True Label: {true_label})", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_img.tick_params(colors=TEXT_COLOR)
    fig.colorbar(cax, ax=ax_img, fraction=0.046, pad=0.04).ax.tick_params(colors=TEXT_COLOR)

    # 2. ANN Softmax Probabilities
    ax_ann = fig.add_subplot(gs[0, 1:])
    ax_ann.set_facecolor(PANEL_BG)
    bars = ax_ann.bar(range(num_classes), ann_probs, color=ACCENT_ORANGE, width=0.45)
    ax_ann.set_title(r"Standard ANN Output: Dense Softmax Probabilities $\sigma(z)$", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_ann.set_xlabel("Digit Class", color=TEXT_COLOR, fontsize=10)
    ax_ann.set_ylabel("Probability", color=TEXT_COLOR, fontsize=10)
    ax_ann.set_xticks(range(num_classes))
    ax_ann.set_ylim(0, 1.1)
    ax_ann.tick_params(colors=TEXT_COLOR)
    ax_ann.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)
    for b in bars:
        ax_ann.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f"{b.get_height():.2f}",
                    ha="center", color=TEXT_COLOR, fontsize=9, fontweight="bold")

    # 3. SNN Rate-Encoded Input Raster
    ax_raster = fig.add_subplot(gs[1, 0])
    ax_raster.set_facecolor(PANEL_BG)
    active_pixels = np.where(sample_digit.flatten() > 0.3)[0][:12]  # select top 12 active pixels
    for idx, p in enumerate(active_pixels):
        spike_times = time_axis[snn_in_spikes[:, p] > 0.5]
        ax_raster.scatter(spike_times, np.full_like(spike_times, idx), color=ACCENT_BLUE, marker="|", s=40, linewidths=1.8)
    ax_raster.set_title("SNN Poisson Input Spikes", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_raster.set_xlabel("Time step (ms)", color=TEXT_COLOR, fontsize=10)
    ax_raster.set_ylabel("Pixel Afferents", color=TEXT_COLOR, fontsize=10)
    ax_raster.tick_params(colors=TEXT_COLOR)
    ax_raster.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    # 4. SNN Postsynaptic LIF Membrane Potentials & Spikes
    ax_v = fig.add_subplot(gs[1, 1:])
    ax_v.set_facecolor(PANEL_BG)
    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_PURPLE]
    for c in range(num_classes):
        is_target = (c == true_label)
        lw = 2.4 if is_target else 1.2
        alpha = 1.0 if is_target else 0.55
        label = f"Class {c} {'(Target)' if is_target else ''}"
        ax_v.plot(time_axis, snn_v[:, c], color=colors[c % len(colors)], linewidth=lw, alpha=alpha, label=label)

        # Plot output spike events
        out_spikes = time_axis[snn_spikes[:, c] > 0.5]
        if len(out_spikes) > 0 and is_target:
            ax_v.scatter(out_spikes, np.full_like(out_spikes, -52.0), color=ACCENT_GREEN, marker="^", s=70, label="Output Spike")

    ax_v.axhline(-55.0, color=ACCENT_RED, linestyle="--", linewidth=1.2, label="Threshold $v_{thresh}=-55$ mV")
    ax_v.set_title("SNN Membrane Potential Evolution $v(t)$ Over Time Steps", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_v.set_xlabel("Time step (ms)", color=TEXT_COLOR, fontsize=10)
    ax_v.set_ylabel("Membrane Potential (mV)", color=TEXT_COLOR, fontsize=10)
    ax_v.tick_params(colors=TEXT_COLOR)
    ax_v.legend(loc="upper left", facecolor=THEME_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax_v.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    fig.suptitle("Experiment 1: Static Image Classification (ANN Dense Softmax vs. SNN LIF Dynamics)",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.97)

    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return save_path


def plot_experiment_2_temporal(exp2_data, save_path="experiment_2_temporal.png"):
    """Visualizes Experiment 2: Moving dot motion classification (L->R vs R->L) and latency."""
    lr_spikes = exp2_data["lr_spikes"]  # (T, 8)
    rl_spikes = exp2_data["rl_spikes"]  # (T, 8)
    lr_v = exp2_data["lr_v_history"]    # (T, 2)
    rl_v = exp2_data["rl_v_history"]    # (T, 2)
    ann_latency = exp2_data["ann_latency"]
    snn_latency = exp2_data["snn_latency"]

    timesteps, n_pixels = lr_spikes.shape
    time_axis = np.arange(timesteps)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), dpi=150)
    fig.patch.set_facecolor(THEME_BG)

    # 1. Left -> Right Spike Raster
    ax_lr_r = axes[0, 0]
    ax_lr_r.set_facecolor(PANEL_BG)
    for p in range(n_pixels):
        spk = time_axis[lr_spikes[:, p] > 0.5]
        ax_lr_r.scatter(spk, np.full_like(spk, p), color=ACCENT_BLUE, marker="|", s=70, linewidths=2.5)
    ax_lr_r.set_title(r"Moving Dot: Left -> Right ($Class\ 0$)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_lr_r.set_ylabel("Pixel Sensor Index", color=TEXT_COLOR, fontsize=10)
    ax_lr_r.tick_params(colors=TEXT_COLOR)
    ax_lr_r.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    # 2. Left -> Right SNN Membrane Potential
    ax_lr_v = axes[0, 1]
    ax_lr_v.set_facecolor(PANEL_BG)
    ax_lr_v.plot(time_axis, lr_v[:, 0], color=ACCENT_BLUE, linewidth=2.2, label="Neuron 0 (Detector L->R)")
    ax_lr_v.plot(time_axis, lr_v[:, 1], color=ACCENT_ORANGE, linewidth=1.5, alpha=0.6, label="Neuron 1 (Detector R->L)")
    ax_lr_v.axhline(-55.0, color=ACCENT_RED, linestyle="--", linewidth=1.2, label="Threshold -55 mV")
    ax_lr_v.set_title("SNN Membrane Potential (L->R Input)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_lr_v.tick_params(colors=TEXT_COLOR)
    ax_lr_v.legend(loc="upper left", facecolor=THEME_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax_lr_v.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    # 3. Right -> Left Spike Raster
    ax_rl_r = axes[1, 0]
    ax_rl_r.set_facecolor(PANEL_BG)
    for p in range(n_pixels):
        spk = time_axis[rl_spikes[:, p] > 0.5]
        ax_rl_r.scatter(spk, np.full_like(spk, p), color=ACCENT_ORANGE, marker="|", s=70, linewidths=2.5)
    ax_rl_r.set_title(r"Moving Dot: Right -> Left ($Class\ 1$)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_rl_r.set_xlabel("Time step (ms)", color=TEXT_COLOR, fontsize=10)
    ax_rl_r.set_ylabel("Pixel Sensor Index", color=TEXT_COLOR, fontsize=10)
    ax_rl_r.tick_params(colors=TEXT_COLOR)
    ax_rl_r.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    # 4. Right -> Left SNN Membrane Potential
    ax_rl_v = axes[1, 1]
    ax_rl_v.set_facecolor(PANEL_BG)
    ax_rl_v.plot(time_axis, rl_v[:, 0], color=ACCENT_BLUE, linewidth=1.5, alpha=0.6, label="Neuron 0 (Detector L->R)")
    ax_rl_v.plot(time_axis, rl_v[:, 1], color=ACCENT_ORANGE, linewidth=2.2, label="Neuron 1 (Detector R->L)")
    ax_rl_v.axhline(-55.0, color=ACCENT_RED, linestyle="--", linewidth=1.2, label="Threshold -55 mV")
    ax_rl_v.set_title("SNN Membrane Potential (R->L Input)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_rl_v.set_xlabel("Time step (ms)", color=TEXT_COLOR, fontsize=10)
    ax_rl_v.tick_params(colors=TEXT_COLOR)
    ax_rl_v.legend(loc="upper left", facecolor=THEME_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax_rl_v.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    fig.suptitle(f"Experiment 2: Temporal Motion Classification (SNN Decision Latency: {snn_latency:.0f} ms vs. ANN: {ann_latency:.0f} ms)",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return save_path


def plot_master_comparison_dashboard(exp1_data, exp2_data, save_path="final_comparison_dashboard.png"):
    """Renders the final 4-panel synthesis comparison dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=150)
    fig.patch.set_facecolor(THEME_BG)

    # 1. Accuracy Comparison
    ax_acc = axes[0, 0]
    ax_acc.set_facecolor(PANEL_BG)
    cats = ["Exp 1: Static Digits", "Exp 2: Temporal Motion"]
    x = np.arange(len(cats))
    w = 0.35
    ax_acc.bar(x - w / 2, [exp1_data["ann_acc"], exp2_data["ann_acc"]], width=w, color=ACCENT_ORANGE, label="Standard ANN")
    ax_acc.bar(x + w / 2, [exp1_data["snn_acc"], exp2_data["snn_acc"]], width=w, color=ACCENT_GREEN, label="Spiking SNN")
    ax_acc.set_title("Classification Accuracy (%)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(cats, color=TEXT_COLOR)
    ax_acc.set_ylim(0, 110)
    ax_acc.tick_params(colors=TEXT_COLOR)
    ax_acc.legend(facecolor=THEME_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax_acc.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    # 2. Inference Synaptic Operations
    ax_ops = axes[0, 1]
    ax_ops.set_facecolor(PANEL_BG)
    ann_ops = [exp1_data["ann_macs_sample"], exp2_data["ann_macs_sample"]]
    snn_ops = [exp1_data["snn_acs_sample"], exp2_data["snn_acs_sample"]]
    ax_ops.bar(x - w / 2, ann_ops, width=w, color=ACCENT_ORANGE, label="ANN MACs (Mult-Add)")
    ax_ops.bar(x + w / 2, snn_ops, width=w, color=ACCENT_GREEN, label="SNN ACs (Add-Only)")
    ax_ops.set_yscale("log")
    ax_ops.set_title("Synaptic Operations per Sample (Log Scale)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_ops.set_xticks(x)
    ax_ops.set_xticklabels(cats, color=TEXT_COLOR)
    ax_ops.tick_params(colors=TEXT_COLOR)
    ax_ops.legend(facecolor=THEME_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax_ops.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    # 3. Decision Latency
    ax_lat = axes[1, 0]
    ax_lat.set_facecolor(PANEL_BG)
    ax_lat.bar(["ANN (Buffer)", "SNN (First-Spike)"], [exp2_data["ann_latency"], exp2_data["snn_latency"]],
               color=[ACCENT_ORANGE, ACCENT_GREEN], width=0.45)
    ax_lat.set_title("Temporal Decision Latency (ms)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_lat.set_ylabel("Latency (ms)", color=TEXT_COLOR, fontsize=10)
    ax_lat.tick_params(colors=TEXT_COLOR)
    ax_lat.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    # 4. Silicon Energy Estimates
    ax_eng = axes[1, 1]
    ax_eng.set_facecolor(PANEL_BG)
    e_ann = [exp1_data["ann_macs_sample"] * 4.6, exp2_data["ann_macs_sample"] * 4.6]
    e_snn = [exp1_data["snn_acs_sample"] * 0.9, exp2_data["snn_acs_sample"] * 0.9]
    ax_eng.bar(x - w / 2, e_ann, width=w, color=ACCENT_ORANGE, label="ANN Energy (pJ)")
    ax_eng.bar(x + w / 2, e_snn, width=w, color=ACCENT_GREEN, label="SNN Energy (pJ)")
    ax_eng.set_yscale("log")
    ax_eng.set_title("Estimated Silicon Energy (pJ, Log Scale)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_eng.set_xticks(x)
    ax_eng.set_xticklabels(cats, color=TEXT_COLOR)
    ax_eng.tick_params(colors=TEXT_COLOR)
    ax_eng.legend(facecolor=THEME_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax_eng.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    fig.suptitle("Final Synthesis: Are SNNs Actually Better Than ANNs?",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return save_path
