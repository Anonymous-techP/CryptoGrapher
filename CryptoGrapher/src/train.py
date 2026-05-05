"""Train and evaluate CryptoGrapher on CRYPTO_1D_ALL."""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_data, get_batch
from evaluator import evaluate
from models import CryptoGrapher, build_correlation_matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(pref: str) -> torch.device:
    if pref.startswith('cuda') or (pref == 'auto' and torch.cuda.is_available()):
        return torch.device(pref if pref.startswith('cuda') else 'cuda')
    if pref == 'mps' or (pref == 'auto' and torch.backends.mps.is_available()):
        return torch.device('mps')
    return torch.device('cpu')


def load_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), 'configs', 'base.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_data_path(cfg: dict) -> str:
    p = cfg['data']['data_path']
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), p
    ))


def compute_loss(pred, gt, mask, loss_name: str, alpha: float) -> torch.Tensor:
    mse = F.mse_loss(pred * mask, gt * mask)
    if loss_name == 'mse':
        return mse
    diff_pred = pred - pred.T
    diff_gt = gt - gt.T
    rank = F.relu(-diff_pred * diff_gt).mean()
    return mse + alpha * rank


def validate(model, data, relation, device, start, end,
             lookback, steps, loss_name, alpha):
    model.eval()
    preds, gts, masks, total = [], [], [], 0.0
    n = 0
    with torch.no_grad():
        for offset in range(start - lookback - steps + 1,
                            end - lookback - steps + 1):
            x_np, m_np, _, y_np = get_batch(data, offset, lookback, steps)
            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            m = torch.tensor(m_np, dtype=torch.float32, device=device)
            y = torch.tensor(y_np, dtype=torch.float32, device=device)
            p = model(x, relation)
            total += compute_loss(p, y, m, loss_name, alpha).item()
            n += 1
            preds.append(p[:, 0].cpu().numpy())
            gts.append(y[:, 0].cpu().numpy())
            masks.append(m[:, 0].cpu().numpy())
    perf = evaluate(np.array(preds).T, np.array(gts).T, np.array(masks).T)
    return total / max(1, n), perf


def train_run(args) -> dict:
    cfg = load_config()
    seed = args.seed if args.seed is not None else cfg['seed']
    epochs = args.epochs if args.epochs is not None else cfg['train']['epochs']
    lr = args.lr if args.lr is not None else cfg['train']['learning_rate']
    loss_name = args.loss if args.loss is not None else cfg['train']['loss']
    device_pref = args.device if args.device else cfg['device']

    lookback = cfg['data']['lookback']
    steps = cfg['data']['steps']
    valid_index = cfg['data']['valid_index']
    test_index = cfg['data']['test_index']
    tau = cfg['model']['correlation_threshold']
    alpha = cfg['train']['rank_alpha']
    weight_decay = cfg['train']['weight_decay']
    validate_every = cfg['train']['validate_every']

    set_seed(seed)
    device = pick_device(device_pref)

    data = load_data(resolve_data_path(cfg), cfg['data']['market_name'])
    n_assets = data['stock_num']
    trade_dates = data['trade_dates']

    rel_np = build_correlation_matrix(data['price_data'], threshold=tau)
    relation = torch.tensor(rel_np, dtype=torch.float32, device=device)
    n_edges = int((rel_np > 0).sum()) - rel_np.shape[0]
    density = n_edges / (rel_np.shape[0] * (rel_np.shape[0] - 1))
    print(f"[graph] tau={tau:.2f}  edges={n_edges}  density={density*100:.1f}%")

    model = CryptoGrapher(
        d_feat=cfg['model']['d_feat'],
        hidden_size=cfg['model']['hidden_size'],
        num_layers=cfg['model']['num_layers'],
        dropout=cfg['model']['dropout'],
        num_graph_layer=cfg['model']['num_graph_layer'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CryptoGrapher] device={device}  seed={seed}  params={n_params:,}  "
          f"epochs={epochs}  lr={lr}  loss={loss_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if cfg['train']['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=cfg['train']['plateau_factor'],
            patience=cfg['train']['plateau_patience'],
        )

    best = {'valid_loss': float('inf'), 'epoch': 0, 'valid': None, 'test': None}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        offsets = list(range(0, valid_index - lookback - steps + 1))
        random.shuffle(offsets)
        for offset in offsets:
            x_np, m_np, _, y_np = get_batch(data, offset, lookback, steps)
            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            m = torch.tensor(m_np, dtype=torch.float32, device=device)
            y = torch.tensor(y_np, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            p = model(x, relation)
            loss = compute_loss(p, y, m, loss_name, alpha)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(offsets))

        if (epoch + 1) % validate_every == 0 or epoch == 0 or epoch == epochs - 1:
            v_loss, v_perf = validate(model, data, relation, device,
                                       valid_index, test_index,
                                       lookback, steps, loss_name, alpha)
            t_loss, t_perf = validate(model, data, relation, device,
                                       test_index, trade_dates,
                                       lookback, steps, loss_name, alpha)
            if scheduler is not None:
                scheduler.step(v_loss)
            improved = v_loss < best['valid_loss']
            if improved:
                best = {'valid_loss': v_loss, 'epoch': epoch + 1,
                        'valid': v_perf, 'test': t_perf}
            marker = ' *' if improved else ''
            print(f"  epoch {epoch+1:3d}/{epochs} | "
                  f"train {train_loss:.2e} | val {v_loss:.2e} | "
                  f"val_IC {v_perf['IC']:+.4f} | "
                  f"test_IC {t_perf['IC']:+.4f} | "
                  f"test_Sharpe {t_perf['sharpe5']:+.4f}{marker}")

    elapsed_min = (time.time() - t0) / 60.0
    print(f"[CryptoGrapher] done in {elapsed_min:.1f} min  "
          f"best_epoch={best['epoch']}  "
          f"test_IC={best['test']['IC']:+.4f}  "
          f"test_ICIR={best['test']['ICIR']:+.4f}  "
          f"test_Sharpe={best['test']['sharpe5']:+.4f}  "
          f"test_Prec@10={best['test']['prec_10']:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"cryptographer_seed{seed}"
    result = {
        'tag': tag,
        'params': n_params,
        'epochs': epochs,
        'learning_rate': lr,
        'loss': loss_name,
        'seed': seed,
        'elapsed_min': elapsed_min,
        'best_epoch': best['epoch'],
        'best_valid': {k: float(v) for k, v in best['valid'].items()},
        'best_test': {k: float(v) for k, v in best['test'].items()},
    }
    path = os.path.join(out_dir, f"{tag}_{ts}.json")
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  -> {path}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--loss', choices=['mse', 'rank'], default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--tag', default=None)
    args = p.parse_args()
    train_run(args)


if __name__ == '__main__':
    main()
