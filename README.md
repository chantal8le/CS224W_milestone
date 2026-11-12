# Graph Construction for Early Sepsis Prediction

CS224W project. Hourly early warning of sepsis on the PhysioNet/Computing in Cardiology Challenge 2019 training set, using a graph attention network (GAT) over physiological variables. We compare how the choice of graph affects prediction and whether clinically informed graphs transfer across hospitals better than graphs estimated from data. The score is the official normalized utility: a perfect alarm sequence is 1, never alarming is 0, and false alarms can push the score below 0.

## Split

40,336 stays from hospitals A and B, split 70/15/15 by patient. The split is stratified on hospital, whether the stay is ever septic, and stay length, so those distributions match across the three sets.

| Split | Patients | Septic stays | Hospital A | Median length | Positive hours |
|---|---:|---:|---:|---:|---:|
| Train | 28,235 | 7.27% | 50.42% | 38 h | 1.80% |
| Dev | 6,053 | 7.29% | 50.42% | 39 h | 1.81% |
| Test | 6,048 | 7.26% | 50.40% | 39 h | 1.78% |

Files live in `data/train`, `data/dev`, and `data/test`, with one pipe-separated file per stay. Patient lists are in `data/manifests`. The public dataset and generated splits are kept locally and ignored by Git. The alarm threshold and checkpoint epoch for every model are chosen on dev and then frozen. `SepsisLabel` is already positive from six hours before onset and is not shifted again.

## Graphs

Each patient-hour is a graph with 34 nodes, one per physiological variable. The topology is shared across patients and hours; only node features change. Each node sees its most recent observed value (forward-filled from past hours only, or the training median before the first measurement), whether it was measured this hour, and the hours since its last measurement, capped at 24. Age, sex, ICU unit, hospital-to-ICU admission time, and current ICU hour skip the graph and join the prediction head.

| Graph | Edges | How edges are chosen |
|---|---:|---|
| Self-loops only | 0 | No edges between variables. This is the model without message passing. |
| Clinical | 47 | Hand-built from physiology: circulation, respiration and acid-base, kidney, liver, blood, electrolytes, and shock. Listed in `compare_graphs.py`. |
| Correlation | 91 | Pearson \|r\| ≥ 0.15 on jointly observed training values, with at least 500 joint observations. The threshold was chosen on dev. |
| Hybrid | — | Not yet run. Union of clinical and correlation edges. |
| Fully connected | 561 | Not yet run. |

Every node also keeps a self-loop.

## Results

One GAT configuration is used for every graph, so only the topology changes: 2 layers (4 heads, then 1), 64 hidden units, SiLU, dropout 0.307, mean pooling, Adam at 1.7e-3, batches of 64 length-grouped stays, 10 epochs. It is the best trial of a 36-trial random search on the correlation graph. All numbers are from a single seed (230).

### Pooled (train on A and B)

| Graph | Dev | Test | Test A | Test B | Test AUPRC | Test AUROC | False alarms / 100 non-septic h |
|---|---:|---:|---:|---:|---:|---:|---:|
| Self-loops only | 0.297 | 0.292 | 0.306 | 0.269 | 0.069 | 0.714 | 4.73 |
| Clinical | 0.308 | 0.300 | 0.307 | 0.289 | 0.072 | 0.739 | 8.35 |
| Correlation | 0.318 | 0.284 | 0.299 | 0.260 | 0.071 | 0.739 | 8.70 |

### Transfer (train on A only)

Epoch and threshold are chosen on dev patients from hospital A. Test hospital B is scored once.

| Graph | Dev A | Test A | Test B | Test B AUROC |
|---|---:|---:|---:|---:|
| Self-loops only | 0.277 | 0.289 | 0.192 | 0.675 |
| Clinical | 0.279 | 0.283 | 0.194 | 0.716 |
| Correlation | 0.276 | 0.274 | 0.198 | 0.698 |

The transfer runs are not yet fully source-only. Normalization statistics, imputation medians, and correlation edges are still fit on training patients from both hospitals. Fitting them on hospital A alone, and adding the B to A direction, is planned.

For reference, a window-feature XGBoost model from the shared CS230 repository reaches 0.397 test utility on the same split. Its features are richer than our node features, so it is a reference point, not a controlled comparison.

Saved metrics are in `runs/graph_compare/metrics.json` and `runs/gat_search/search.json`.

## Run

```bash
python make_split.py /path/to/PhysioNet_2019_Full_Training_Data.zip
python train_gat.py          # builds the training correlation matrix and picks the edge threshold
python search_gat.py         # optional: GAT hyperparameter search (--smoke for a one-epoch test)
python compare_graphs.py     # trains every graph, pooled and train-on-A, and writes runs/graph_compare/
```

`dataset.py` builds and caches the processed features on first use. `train_gat.py` caches the correlation matrix in `runs/gat/`. `compare_graphs.py` uses the fixed configuration above and reads that cache.

| File | Purpose |
|---|---|
| `compare_graphs.py` | Graph comparison and the clinical edge list |
| `train_gat.py` | Correlation matrix, edge rule, and threshold selection |
| `search_gat.py` | Random search over GAT hyperparameters |
| `model.py` | `PhysioGAT` and its dense attention layer |
| `dataset.py` | Causal preprocessing and node features |
| `utility.py` | Official normalized utility and threshold choice |
| `make_split.py` | Patient split from the PhysioNet ZIP |
| `train_cnn_baseline.py`, `train_lstm.py`, `winner_data.py` | Shared helpers for metrics, seeding, per-hospital scoring, and raw data loading |

Checkpoints for the six comparison runs are in `runs/graph_compare/`, and the best search trial is `runs/gat_search/best.pt`.

## Milestone

The CS224W milestone is in `224w_milestone/`. Compile `224w_milestone.tex` with `neurips_2022.sty` in the same folder.

## Data

PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0. Reyna et al., DOI 10.13026/v64v-d857. License: CC BY 4.0. [Dataset and license](https://physionet.org/content/challenge-2019/1.0.0/). The 40,336 public PSV files and generated `data/` directory are not tracked in Git. Download them from PhysioNet. Do not commit the extracted patient files.

To reproduce the split used in the reported runs, obtain `PhysioNet_2019_Full_Training_Data.zip` and run:

```bash
python make_split.py /path/to/PhysioNet_2019_Full_Training_Data.zip
```

The script creates `data/train`, `data/dev`, `data/test`, and `data/manifests`.
