"""
Metrics and Final Comparison Table Formatter.
Calculates computational operations, parameter budgets, latency, and biological similarity.
"""

import numpy as np


def compute_energy_estimates(macs, acs):
    """
    Computes energy consumption estimates in picojoules (pJ) based on standard CMOS 45nm benchmarks:
    - 32-bit Float Multiply-Accumulate (MAC) = ~4.6 pJ
    - 32-bit Float Accumulate-only (AC) = ~0.9 pJ
    """
    ann_energy_pj = macs * 4.6
    snn_energy_pj = acs * 0.9
    return ann_energy_pj, snn_energy_pj


def print_master_comparison_table(exp1, exp2):
    """
    Prints the final comparative table directly fulfilling the specifications in docs/experiment.md.
    """
    header = "=" * 80
    sub_header = "-" * 80

    ann_e1_energy, snn_e1_energy = compute_energy_estimates(exp1["ann_macs_sample"], exp1["snn_acs_sample"])
    ann_e2_energy, snn_e2_energy = compute_energy_estimates(exp2["ann_macs_sample"], exp2["snn_acs_sample"])

    # Pre-format table cells to prevent any escaped quotes in f-strings
    e1_ann_acc = f"{exp1['ann_acc']:.1f}%"
    e1_snn_acc = f"{exp1['snn_acc']:.1f}%"
    e1_ann_time = f"{exp1['ann_train_time']:.2f} s"
    e1_snn_time = f"{exp1['snn_train_time']:.2f} s"
    e1_ann_params = f"{exp1['ann_params']} weights"
    e1_snn_params = f"{exp1['snn_params']} weights"
    e1_ann_ops = f"{exp1['ann_macs_sample']} MACs"
    e1_snn_ops = f"{exp1['snn_acs_sample']:.1f} ACs (Add-only)"
    e1_snn_sparse = f"{exp1['snn_sparsity']:.1f}% Sparse"
    e1_ann_energy = f"{ann_e1_energy:,.1f} pJ"
    e1_snn_energy = f"{snn_e1_energy:,.1f} pJ"

    e2_ann_acc = f"{exp2['ann_acc']:.1f}%"
    e2_snn_acc = f"{exp2['snn_acc']:.1f}%"
    e2_ann_lat = f"{exp2['ann_latency']:.0f} ms (Full buffer)"
    e2_snn_lat = f"{exp2['snn_latency']:.0f} ms (First-to-spike)"
    e2_ann_ops = f"{exp2['ann_macs_sample']} MACs"
    e2_snn_ops = f"{exp2['snn_acs_sample']:.1f} ACs (Add-only)"
    e2_ann_energy = f"{ann_e2_energy:,.1f} pJ"
    e2_snn_energy = f"{snn_e2_energy:,.1f} pJ"

    energy_gain = ann_e2_energy / max(snn_e2_energy, 1e-4)
    e2_gain_str = f"{energy_gain:.1f}x More Efficient"

    lines = [
        header,
        f"{'DIMENSION / METRIC':<28} | {'STANDARD ANN':<23} | {'BIOLOGICAL SNN':<23}",
        header,
        f"{'--- EXPERIMENT 1: STATIC IMAGE CLASSIFICATION (8x8 DIGITS) ---':<80}",
        sub_header,
        f"{'Test Accuracy':<28} | {e1_ann_acc:<23} | {e1_snn_acc:<23}",
        f"{'Train Time':<28} | {e1_ann_time:<23} | {e1_snn_time:<23}",
        f"{'Parameter Budget':<28} | {e1_ann_params:<23} | {e1_snn_params:<23}",
        f"{'Inference Synaptic Ops':<28} | {e1_ann_ops:<23} | {e1_snn_ops:<23}",
        f"{'Spike Activity / Sparsity':<28} | {'0.0% (Dense continuous)':<23} | {e1_snn_sparse:<23}",
        f"{'Estimated Energy / Sample':<28} | {e1_ann_energy:<23} | {e1_snn_energy:<23}",
        sub_header,
        f"{'--- EXPERIMENT 2: TEMPORAL MOTION CLASSIFICATION (MOVING DOT) ---':<80}",
        sub_header,
        f"{'Motion Test Accuracy':<28} | {e2_ann_acc:<23} | {e2_snn_acc:<23}",
        f"{'Decision Latency':<28} | {e2_ann_lat:<23} | {e2_snn_lat:<23}",
        f"{'Temporal Synaptic Ops':<28} | {e2_ann_ops:<23} | {e2_snn_ops:<23}",
        f"{'Temporal Energy / Sample':<28} | {e2_ann_energy:<23} | {e2_snn_energy:<23}",
        sub_header,
        f"{'--- COMPUTATIONAL & ARCHITECTURAL SUMMARY ---':<80}",
        sub_header,
        f"{'Compute Paradigm':<28} | {'Continuous Frame Matrices':<23} | {'Event-Driven Spikes':<23}",
        f"{'Temporal Representation':<28} | {'Limited (Flattened frames)':<23} | {'Intrinsic (Leaky dynamics)':<23}",
        f"{'Biological Similarity':<28} | {'Low (Artificial / Math)':<23} | {'High (LIF + STDP)':<23}",
        f"{'Energy Efficiency Gain':<28} | {'1.0x (Baseline)':<23} | {e2_gain_str:<23}",
        header,
        "\nCORE VERDICT (from docs/experiment.md):",
        "  ANNs are currently the practical general-purpose choice.",
        "  SNNs become particularly interesting when sparsity, timing, latency,",
        "  and energy efficiency matter.\n",
        header
    ]
    return "\n".join(lines)
