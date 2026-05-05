# CryptoGrapher

Official anonymous reference implementation of **CryptoGrapher: Revisiting Cryptocurrency Price Forecasting through a Graph Learning Lens** (under review).

A simple yet effective graph-based baseline for pure-price cryptocurrency forecasting. The model takes raw OHLCV sequences as input and predicts next-day returns through three components:

```
OHLCV  ──►  LSTM encoder  ──►  2-layer GAT on correlation graph  ──►  Linear head  ──►  r̂_i
            (per-coin)         (cross-asset)                          (prediction)
```

The graph is constructed *purely from training-period closing prices* via Pearson correlation thresholding (**τ = 0.9**) — no external priors, no fundamentals, no social data.

## Architecture

![CryptoGrapher Pipeline](figures/Overview_CryptoGrapher.png)

The whole model has **~60K parameters** and runs in **<1 ms per inference step**.

## Repository Layout

```
CryptoGrapher/
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/
│   ├── Binance_USDT/                 # 68 raw daily OHLCV CSVs (Binance)
│   ├── CRYPTO_1D_ALL/                # processed pickles consumed by training
│   │   ├── eod_data.pkl              # (66, 999, 5)  OHLCV
│   │   ├── mask_data.pkl             # (66, 999)     trading mask
│   │   ├── gt_data.pkl               # (66, 999)     next-day returns
│   │   ├── price_data.pkl            # (66, 999)     close prices
│   │   └── coin_names.txt
│   └── process_all_to_stockmixer_format.py    # CSV → pickle pipeline
├── src/
│   ├── train.py                      # training entry point
│   ├── evaluator.py                  # IC / ICIR / Sharpe@5 / Prec@10
│   ├── configs/
│   │   └── base.yaml                 # all hyperparameters + seed
│   ├── data/
│   │   └── loader.py                 # CRYPTO_1D_ALL loader
│   └── models/
│       ├── cryptographer.py          # CryptoGrapher main model
│       └── gat_layer.py              # GAT layer + correlation graph builder
└── figures/
    └── Overview_CryptoGrapher.png
```

Total core code: **~350 lines**. Everything else is hyperparameters and I/O.

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset from raw CSVs

The processed `dataset/CRYPTO_1D_ALL/` is shipped with the repo, so you can skip this step. To rebuild from scratch:

```bash
cd dataset
python process_all_to_stockmixer_format.py
```

This produces 4 pickle files of shape `(66, 999, 5)` / `(66, 999)` from the raw Binance CSVs (66 coins after excluding USDC and TUSD stablecoins, 999 common trading days from 2023-04-16 to 2026-01-08).

### 3. Run training

```bash
# Full run (100 epochs, seed 123456789, matches paper Table I)
python src/train.py --device cuda:0

# Quick smoke test (1 epoch on CPU)
python src/train.py --epochs 1 --device cpu

# Pairwise-rank loss variant
python src/train.py --loss rank --device cuda:0
```

The script auto-loads `src/configs/base.yaml` and reads pickle files from `dataset/CRYPTO_1D_ALL/`.

### 4. Reference numbers

On `CRYPTO_1D_ALL` (66 coins, 30-day lookback, train/val/test = 0–598 / 599–798 / 799+, seed 123456789, τ = 0.9):

| Metric              | Value     |
|---------------------|----------:|
| Parameters          | 59,969    |
| Test IC             | +0.025    |
| Test ICIR           | +0.108    |
| Test Sharpe (Ann.)  | +3.072    |
| Test Prec@10        | 0.504     |

## Hyperparameters (paper defaults)

| Component                   | Value             |
|-----------------------------|-------------------|
| Lookback `T`                | 30 trading days   |
| Hidden size `d`             | 64                |
| LSTM layers                 | 2                 |
| GAT layers `L`              | 2                 |
| Correlation threshold `τ`   | **0.9**           |
| Optimizer                   | Adam, lr = 2.5e-4 |
| Epochs                      | 100               |
| Loss                        | MSE (rank α = 0.1 also available) |
| Dropout                     | 0                 |
| Seed                        | 123456789         |

Override any of these on the command line — see `python src/train.py --help`.

## Dataset Format

`dataset/CRYPTO_1D_ALL/` contains four pickle files describing **66 cryptocurrencies over 999 trading days** (2023-04-16 → 2026-01-08):

```
CRYPTO_1D_ALL/
├── eod_data.pkl       (66, 999, 5)   OHLCV (normalized)
├── mask_data.pkl      (66, 999)      binary trading mask
├── gt_data.pkl        (66, 999)      next-day return labels
└── price_data.pkl     (66, 999)      close prices (used for graph)
```

The graph at **τ = 0.9** has **2.9 % density** (126 directed edges) — see paper §4.2 for the full threshold ablation. The sparse graph isolates only the strongest co-movement signals; lower τ admits noisy edges that degrade ranking quality.

## Code-to-Paper Reference

| Paper component               | File                                               |
|-------------------------------|----------------------------------------------------|
| Eq. (1) task formulation      | `src/train.py` (training loop)                     |
| Algorithm 1 graph construction| `src/models/gat_layer.py:build_correlation_matrix` |
| Eq. (4) LSTM encoder          | `src/models/cryptographer.py` (`self.rnn`)         |
| Eq. (5–6) graph attention     | `src/models/gat_layer.py:GATLayer.forward`         |
| Eq. (7) linear head           | `src/models/cryptographer.py` (`self.fc`)          |
| Eq. (8–9) loss                | `src/train.py:compute_loss`                        |

## Reproducibility

- Deterministic CUDA: `torch.backends.cudnn.deterministic = True`
- Fixed seed: `123456789` (set in `src/train.py:set_seed`)
- All RNG state set before model construction

## License

Anonymous submission — license to be added upon acceptance.

## Citation

```bibtex
@inproceedings{cryptographer2026,
  title     = {CryptoGrapher: Revisiting Cryptocurrency Price Forecasting through a Graph Learning Lens},
  author    = {Anonymous Authors},
  booktitle = {Under Review},
  year      = {2026}
}
```
