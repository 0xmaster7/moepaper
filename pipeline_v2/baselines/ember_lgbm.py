"""Baseline 1: EMBER features + LightGBM (the standard static PE baseline).

Same EMBER vectors the gate sees (thrember, header-parity bytes), same splits.
Hyper-parameters are the EMBER benchmark settings; early stopping on VAL.

    python baselines/ember_lgbm.py --seed 42 [--tags]
--tags also trains one LightGBM per behaviour tag on malware rows (tag baseline).
"""
import argparse

import lightgbm as lgb
import numpy as np

from _base import load, save_preds

EMBER_PARAMS = {"objective": "binary", "boosting": "gbdt", "learning_rate": 0.05, "num_leaves": 2048,
                "max_depth": 15, "min_data_in_leaf": 50, "feature_fraction": 0.5, "bagging_fraction": 0.5,
                "bagging_freq": 1, "verbose": -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default=None)
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--tags", action="store_true")
    args = ap.parse_args()
    cfg, split, D, tr, va, te = load(args.split)
    X, y, Y = D["X_ember"], D["y"], D["Y_type"]
    params = dict(EMBER_PARAMS, seed=args.seed, num_threads=0)
    dtr, dva = lgb.Dataset(X[tr], y[tr]), lgb.Dataset(X[va], y[va])
    model = lgb.train(params, dtr, args.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])
    score = model.predict(X, num_iteration=model.best_iteration)
    tag_scores = None
    if args.tags:
        tag_scores = np.zeros_like(Y)
        mtr, mva = tr[y[tr] == 1], va[y[va] == 1]
        small = dict(params, num_leaves=64, max_depth=-1)
        for j in range(Y.shape[1]):
            if Y[mtr, j].sum() < 5:
                continue
            mj = lgb.train(small, lgb.Dataset(X[mtr], Y[mtr, j]), 500,
                           valid_sets=[lgb.Dataset(X[mva], Y[mva, j])],
                           callbacks=[lgb.early_stopping(50, verbose=False)])
            tag_scores[:, j] = mj.predict(X, num_iteration=mj.best_iteration)
    save_preds("ember_lgbm", args.seed, split, D, score, tag_scores)


if __name__ == "__main__":
    main()
