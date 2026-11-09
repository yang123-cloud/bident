#!/usr/bin/env python3
"""Bident command line.

  python run_bident.py run     --data_dir pcaps/ --out_dir out/ [--n_known 1] [--known A,B]
  python run_bident.py predict --model out/bident.pt --input new.pcap --out pred.csv

`run`     folder with one pcap per class (class name = file name).  Protocol of the paper:
          only `n_known` classes are known at the start, all other classes are unknown;
          unknown flows are buffered, clustered with DBSCAN and added as new learners.
`predict` classify the flows of a pcap / folder of pcaps with a saved model.
"""
import argparse
import csv
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config                      # noqa: E402
from dat import evaluate_open_set              # noqa: E402
from detector import Detector, pick_device     # noqa: E402
from pcap_io import build_dataset              # noqa: E402


def add_cfg_args(p):
    for f in dataclasses.fields(Config):
        if f.type is bool:
            p.add_argument("--" + f.name, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument("--" + f.name, type=f.type, default=f.default)


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
def cmd_run(args):
    cfg = Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Config)})
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    log(f"[env] device = {device}")

    ds = build_dataset(args.data_dir, cfg, args.cache, log)
    names, n_cls = ds.class_names, len(ds.class_names)
    rng = np.random.default_rng(cfg.seed)

    # known / unknown class partition
    if cfg.known:
        known = [names.index(k.strip()) for k in cfg.known.split(",")]
    else:
        known = sorted(rng.choice(n_cls, size=min(cfg.n_known, n_cls), replace=False).tolist())
    log(f"[protocol] known   : {[names[c] for c in known]}")
    log(f"[protocol] unknown : {[names[c] for c in range(n_cls) if c not in known]}")

    # per-class train / test split
    train, test = {}, {}
    for c in range(n_cls):
        idx = rng.permutation(np.where(ds.label == c)[0])
        n_test = max(1, int(round(len(idx) * cfg.test_ratio)))
        test[c], train[c] = idx[:n_test], idx[n_test:]
        if len(train[c]) < 20:
            log(f"[data] WARNING: class {names[c]} has only {len(train[c])} training flows")

    det = Detector(cfg, device)
    t0 = time.time()
    det.fit_shared(ds, np.concatenate([train[c] for c in known]), log)
    for c in known:
        det.add_class(names[c], ds, train[c], pool_class=c, log=log)
    log(f"[train] done in {time.time() - t0:.0f}s")

    test_idx = rng.permutation(np.concatenate([test[c] for c in range(n_cls)]))
    pred, first_unknown = det.run_stream(ds, test_idx, discover=True, log=log)
    res = evaluate_open_set(ds.label[test_idx], pred, det.pool_class, known, n_cls)

    log("\n================ results (test flows, all classes) ================")
    log(f"ACC {res['acc'] * 100:.2f}   macro-P {res['precision'] * 100:.2f}   "
        f"macro-R {res['recall'] * 100:.2f}   macro-F1 {res['f1'] * 100:.2f}   AMI {res['ami'] * 100:.2f}")
    if "unknown_detection" in res:
        u = res["unknown_detection"]
        log(f"known vs unknown (class-agnostic): ACC {u['acc'] * 100:.2f}  P {u['precision'] * 100:.2f}  "
            f"R {u['recall'] * 100:.2f}  F1 {u['f1'] * 100:.2f}")
    log(f"discovered learners: {res['n_discovered']}  cluster->class: "
        f"{ {det.names[k]: names[v] for k, v in res['cluster_to_class'].items()} }")
    for c in range(n_cls):
        tag = "known  " if c in known else "unknown"
        log(f"  {tag} {names[c]:<24s} F1 {res['per_class_f1'][c] * 100:6.2f}")

    # outputs
    res["classes"] = names
    res["known"] = [names[c] for c in known]
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    mapping = res["cluster_to_class"]
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["flow_start", "true_class", "pred_learner", "pred_class", "first_pass_unknown"])
        for i, fi in enumerate(test_idx):
            p = int(pred[i])
            pl = det.names[p] if p >= 0 else "unknown"
            if p < 0:
                pc = "unknown"
            elif det.pool_class[p] >= 0:
                pc = names[det.pool_class[p]]
            else:
                pc = names[mapping[p]] if p in mapping else "unmatched_cluster"
            w.writerow([f"{ds.start[fi]:.6f}", names[ds.label[fi]], pl, pc, int(first_unknown[i])])
    det.save(os.path.join(args.out_dir, "bident.pt"))
    log(f"\n[out] {args.out_dir}/metrics.json, predictions.csv, bident.pt")


# --------------------------------------------------------------------------- #
def cmd_predict(args):
    device = pick_device(args.device)
    det = Detector.load(args.model, device)
    cfg = det.cfg
    if args.batch_size:
        cfg.batch_size = args.batch_size
    ds = build_dataset(args.input, cfg, None, log)
    files = ds.class_names                       # one entry per input pcap
    idx = np.arange(len(ds))
    pred, first_unknown = det.run_stream(ds, idx, discover=args.discover, log=log)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "flow_start", "pred"])
        for i in idx:
            p = int(pred[i])
            w.writerow([files[ds.label[i]], f"{ds.start[i]:.6f}", det.names[p] if p >= 0 else "unknown"])
    n_unk = int((pred < 0).sum())
    log(f"[predict] {len(idx)} flows, {n_unk} unknown -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train + evaluate with the open-set protocol of the paper")
    r.add_argument("--data_dir", required=True, help="folder with one pcap per class")
    r.add_argument("--out_dir", default="bident_out")
    r.add_argument("--cache", default=None, help="optional .npz cache of the parsed flows")
    add_cfg_args(r)
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("predict", help="classify flows with a saved model")
    q.add_argument("--model", required=True)
    q.add_argument("--input", required=True, help="a pcap file or a folder of pcaps")
    q.add_argument("--out", default="predictions.csv")
    q.add_argument("--discover", action="store_true", help="also cluster unknown flows into new classes")
    q.add_argument("--device", default="auto")
    q.add_argument("--batch_size", type=int, default=0)
    q.set_defaults(fn=cmd_predict)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
