# Can a Biological Brain Beat a Neural Network?

A from-scratch comparison of a **spiking neural network (SNN)** and a standard **artificial neural network (ANN)**. It's written in plain Python and NumPy, using **only functions**: no classes and no deep learning framework.

Both networks get the same problems. We compare accuracy, response time, counted operations and estimated energy.

## What's inside

- **Leaky Integrate-and-Fire (LIF) neurons:** a "leaky bucket" model of a biological neuron (`lif_step` in `snn_vs_ann/snn_engine.py`).
- **Rate encoding:** turns static pixel values into Poisson spike trains (`rate_encode`).
- **STDP (Spike-Timing-Dependent Plasticity):** a local, biology-inspired learning rule with no backpropagation (`stdp_step`).
- **A two-layer ANN baseline:** ReLU, softmax and backpropagation (`snn_vs_ann/ann_engine.py`).
- **Operation counting and energy estimates:** multiply-accumulates (MACs) vs additions (ACs), with a 45 nm CMOS assumption of about 4.6 pJ per MAC and about 0.9 pJ per AC (`snn_vs_ann/metrics.py`).

## Experiments

1. **STDP demo:** a pre/post spike pairing protocol across Δt ∈ [-60, +60] ms. It shows potentiation (pre before post) and depression (post before pre).
2. **Static images:** classifies synthetic 8×8 digits (0–3) with small shifts and noise.
3. **Motion:** eight sensors watch a dot moving left→right or right→left. The order of the spikes is the information.

The full design is in [`docs/experiment.md`](docs/experiment.md).

## Project structure

```
.
├── main.py                 # runs everything and writes results/
├── requirements.txt
├── snn_vs_ann/
│   ├── ann_engine.py       # ANN: init, forward, backprop, training
│   ├── snn_engine.py       # SNN: rate encoding, LIF neurons, STDP, training
│   ├── datasets.py         # synthetic 8x8 digits + moving-dot sequences
│   ├── experiments.py      # STDP demo, Experiment 1, Experiment 2
│   ├── metrics.py          # op counts, energy estimates, comparison table
│   └── visualizer.py       # matplotlib figures
├── docs/
│   └── experiment.md       # experiment design notes
└── results/                # generated plots + run log (committed as a reference run)
```

## Quick start

Requires Python 3.10+.

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

The run prints a comparison table and writes four figures to `results/`:

| File | Shows |
| --- | --- |
| `stdp_demonstration.png` | the STDP timing curve |
| `experiment_1_static.png` | input digit, ANN prediction, input spikes and SNN membrane potentials |
| `experiment_2_temporal.png` | moving-dot spike trains and SNN detector voltages |
| `final_comparison_dashboard.png` | a side-by-side summary |

`results/run.log` holds the console output of the reference run.

## A note on the numbers

Operation counts and energy figures come from a **simplified model**. They are estimates, not measurements from physical hardware. Different chips and implementations could produce very different results. These are two small toy experiments. They don't show that spiking networks are generally better than modern AI, and the task and the hardware matter.

## Requirements

- numpy >= 1.26
- matplotlib >= 3.8
