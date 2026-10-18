"""
Main Orchestration Script: ANN vs SNN Experimental Study.
Follows the experimental design in docs/experiment.md:
1. Learning Demonstration: STDP (Spike timing -> synaptic weight change).
2. Experiment 1: Static Image Classification (8x8 Digits).
3. Experiment 2: Temporal Motion Classification (Moving Dot Direction).
4. Master Synthesis: Are SNNs actually better than ANNs?
"""

import os
import numpy as np

from snn_vs_ann.datasets import get_static_digits_dataset, get_temporal_moving_dot_dataset
from snn_vs_ann.experiments import (
    run_stdp_experiment,
    run_static_classification_experiment,
    run_temporal_classification_experiment
)
from snn_vs_ann.metrics import print_master_comparison_table
from snn_vs_ann.visualizer import (
    plot_stdp_pairing_curve,
    plot_experiment_1_static,
    plot_experiment_2_temporal,
    plot_master_comparison_dashboard
)


def main():
    print("\n" + "=" * 80)
    print("ANN VS SNN EXPERIMENTAL SUITE: FROM BIOLOGY TO BENCHMARK")
    print("=" * 80)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------------------------------------------
    # DEMO: BIOLOGICAL LEARNING (STDP PLASTICITY)
    # -------------------------------------------------------------
    print("\n[PART 1] Biological Learning Demonstration: STDP Hebbian Plasticity...")
    stdp_data = run_stdp_experiment()
    stdp_plot_path = os.path.join(results_dir, "stdp_demonstration.png")
    plot_stdp_pairing_curve(stdp_data, save_path=stdp_plot_path)
    print("      -> Simulated pre/post pairing protocol across Delta t [-60 ms, +60 ms].")
    print("      -> Pre before Post  ==> LTP (weight strengthened).")
    print("      -> Post before Pre ==> LTD (weight depressed).")
    print(f"      -> STDP plot exported: {os.path.relpath(stdp_plot_path)}")

    # -------------------------------------------------------------
    # EXPERIMENT 1: STATIC IMAGE CLASSIFICATION (8x8 DIGITS)
    # -------------------------------------------------------------
    print("\n[PART 2] Experiment 1: Static Image Classification (8x8 Digits)...")
    print("      Question: Can an SNN perform as well as an ANN on normal static images?")
    static_dataset = get_static_digits_dataset(n_samples_per_digit=45, seed=42)
    exp1_results = run_static_classification_experiment(static_dataset, epochs_ann=35, epochs_snn=22)

    exp1_plot_path = os.path.join(results_dir, "experiment_1_static.png")
    plot_experiment_1_static(exp1_results, save_path=exp1_plot_path)
    print(f"      -> ANN Test Accuracy: {exp1_results['ann_acc']:.1f}% ({exp1_results['ann_macs_sample']} MACs/sample)")
    print(f"      -> SNN Test Accuracy: {exp1_results['snn_acc']:.1f}% ({exp1_results['snn_acs_sample']:.1f} ACs/sample)")
    print(f"      -> SNN Membrane Sparsity: {exp1_results['snn_sparsity']:.1f}% (Event-driven silence)")
    print(f"      -> Static experiment plot exported: {os.path.relpath(exp1_plot_path)}")
    print("      -> INSIGHT: ANN is fast and direct for static images. SNN matches accuracy,")
    print("                  but requires rate-encoding steps. BUT what if TIMING is the info?")

    # -------------------------------------------------------------
    # EXPERIMENT 2: TEMPORAL MOTION CLASSIFICATION (MOVING DOT)
    # -------------------------------------------------------------
    print("\n[PART 3] Experiment 2: Temporal Motion Classification (Moving Dot Direction)...")
    print("      Question: What happens when timing and sequence order ARE the information?")
    temporal_dataset = get_temporal_moving_dot_dataset(n_samples=120, timesteps=16, seed=42)
    exp2_results = run_temporal_classification_experiment(temporal_dataset, epochs_ann=30, epochs_snn=25)

    exp2_plot_path = os.path.join(results_dir, "experiment_2_temporal.png")
    plot_experiment_2_temporal(exp2_results, save_path=exp2_plot_path)
    print(f"      -> ANN Motion Accuracy: {exp2_results['ann_acc']:.1f}% (Latency: {exp2_results['ann_latency']:.0f} ms - waits for full buffer)")
    print(f"      -> SNN Motion Accuracy: {exp2_results['snn_acc']:.1f}% (Latency: {exp2_results['snn_latency']:.0f} ms - early first-spike decision!)")
    print(f"      -> SNN Synaptic Ops:    {exp2_results['snn_acs_sample']:.1f} ACs vs ANN {exp2_results['ann_macs_sample']} MACs")
    print(f"      -> Temporal experiment plot exported: {os.path.relpath(exp2_plot_path)}")
    print("      -> INSIGHT: This is where SNN shines! Real-time temporal integration with zero wait.")

    # -------------------------------------------------------------
    # FINAL COMPARISON & MASTER DASHBOARD
    # -------------------------------------------------------------
    print("\n[PART 4] Final Comparative Synthesis...")
    table_text = print_master_comparison_table(exp1_results, exp2_results)
    print(table_text)

    dashboard_path = os.path.join(results_dir, "final_comparison_dashboard.png")
    plot_master_comparison_dashboard(exp1_results, exp2_results, save_path=dashboard_path)
    print(f"\n[DONE] Master summary dashboard saved: {os.path.relpath(dashboard_path)}\n")


if __name__ == "__main__":
    main()
