# Physics-informed optimal control by control-only supervision

Code for **Physics-Informed Optimal Control by Control-Only Supervision with
Error Guarantees on Value and Policy**, NeurIPS 2026, by Zihua Wang,
Xiaopei Jiao, and Yunfeng Cai.

This release includes the main-text datasets and two trained value networks,
plus training and evaluation code for the main and supplementary experiments. Numerical results may vary with random seeds, hardware, and software
versions.

## Installation

Use Python 3.10 or later and install a suitable
[PyTorch build](https://pytorch.org/get-started/locally/) for your CPU/GPU.
Run all commands below from the repository root:

```bash
python -m pip install -r requirements.txt
```

Use `--system cs`, `ac`, or `both` to select systems. For training,
`--device original` means CS on CPU and AC on CUDA; use `--device cpu` or
`--device cuda` to choose explicitly.

## Included main-text models and data

| System | Dataset | Value-network weights | Training/held-out |
|---|---|---|---|
| Cucker--Smale (CS) | `datasets/cs_main.mat` | `pretrained/cs_value.pth` | 16,000 / 4,000 |
| Allen--Cahn (AC) | `datasets/ac_main.mat` | `pretrained/ac_value.pth` | 9,000 / 1,000 |

The CS state/control dimensions are 40/20; AC dimensions are 51/1. Normalization
is stored in the weights and applied by the network. The control policy is
computed from the value gradient, so no separate policy checkpoint is needed.

Evaluate these models without training:

```bash
python reproduce.py main --stage evaluate --system both --device cpu --run-name provided_models --checkpoint-cs pretrained/cs_value.pth --checkpoint-ac pretrained/ac_value.pth
```

State, control, and value plots are saved under `outputs/main/provided_models/`.
CS uses positions evenly spaced in [0,0.2] and velocities in [0.2,0.4], with
horizon 15. AC uses `1+xi*(1-xi)`, with horizon 20. Controls update every 0.01.
Use a different `--run-name` to keep multiple evaluation runs.

## Prepare data and train

To train with the included samples, import them once:

```bash
python reproduce.py data import --cs-main datasets/cs_main.mat --ac-main datasets/ac_main.mat
python reproduce.py main --stage train --system both --device original
python reproduce.py main --stage evaluate --system both --device cpu
```

Training saves full-data models under
`outputs/generalization/main_protocol/training/{cs,ac}/full/`.
Existing inputs and evaluation outputs are not overwritten.

The MAT files contain `X` (sampled states), `U_all` (SDRE control labels), and
`V_all`/`DV_all` (SDRE value and gradient proxies). Training uses control labels
and HJB states, without supervised value targets. Preprocessing uses the full
state pool; held-out points monitor losses and are separate from closed-loop
test initial conditions.

Alternatively, to generate a new sample pool in an empty `data/` input folder:

```bash
python reproduce.py data generate --system both --seed 42
```

This saves 20,000 CS samples and 10,000 AC samples, including copies for the
noise studies. Generation requires SDRE trajectory simulation.

## Ablations and loss-weight sensitivity

```bash
python reproduce.py ablation --stage train --system both --device original
python reproduce.py ablation --stage evaluate --system both --device cpu
```

The settings include HJB-only, control-only, PolicyNet (a direct state-to-control
network), and gamma values 0.001, 0.01, 0.1, 1, 10, 100, 1000. Use `--gammas`
to select settings. Both systems use 80/20 splits here; the AC split differs
from the main experiments. Results go to `outputs/ablation/`.

## Repeated label-noise experiments

The commands below use the included main-text sample pools for noise reruns;
the separate AC noise sample pool used in the reported table is not included.
Skip the import command if the data generator already created these inputs.

```bash
python reproduce.py data import --cs-noise datasets/cs_main.mat --ac-noise datasets/ac_main.mat
python reproduce.py noise --stage prepare --system both --device cpu
python reproduce.py noise --stage train --system both --device cuda
python reproduce.py noise --stage evaluate --system both --device cpu
python reproduce.py noise --stage summarize --system both --device cpu
```

Defaults use four noise ratios (0.10, 0.25, 0.50, 0.75), five independent noise
realizations, five common test initial conditions, and an 80/20 split. Costs
are averaged over initial conditions before the mean and sample standard
deviation across noise realizations. Proposed/control-only/PolicyNet networks
are 2x128 / 2x256 / 3x512 for CS and 2x128 / 2x128 / 3x512 for AC.
Results go to `outputs/noise/`.

## OOD, data-size, and dynamics studies

Run these after training the full-data models with `main --stage train`:

```bash
python reproduce.py ood --stage evaluate --system both --device cpu
python reproduce.py labels --stage train --budget half --system both --device original
python reproduce.py labels --stage evaluate --system both --device cpu
python reproduce.py dynamics --stage evaluate --system both --device cpu
```

These studies use horizon 50 and five initial conditions per group. OOD scales
are 1, 1.25, 1.5, and 2. The half-data study uses CS splits of 8,000/2,000 and
AC splits of 4,500/500 while retaining full-pool normalization. The dynamics
study freezes the trained policy and varies CS interaction strength by factors
0.9, 1, 1.1 or AC diffusion to 0.09, 0.10, 0.11.
Results go to `outputs/generalization/main_protocol/`.

Use `python reproduce.py <experiment> --help` to list options.
Generated files remain under `data/` and `outputs/`.

## License

Code and included model weights: MIT ([LICENSE](LICENSE)). Simulation datasets
and numerical results: CC BY 4.0 ([DATA_LICENSE.md](DATA_LICENSE.md)).
