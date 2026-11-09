"""Training / inference logic of Bident: shared representation, per-class learners,
DAT calibration, dual verification, buffering and incremental class discovery."""
import math
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from config import Config
from dat import discover_clusters, dual_verify, fit_dat_threshold
from model import BidentEncoder, SingleClassLearner


def pick_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_batch(ds, idx, device):
    """Gather one mini-batch: bytes, packet features / masks of the whole contextual window."""
    idx = np.asarray(idx)
    xb = torch.from_numpy(ds.X[idx]).to(device).float().div_(255.0)
    ctx = ds.ctx[idx]                                   # (B,Nc)
    valid = ctx >= 0
    safe = np.where(valid, ctx, 0)
    pkt = torch.from_numpy(ds.pkt[safe]).to(device)     # (B,Nc,m,2)
    pm = torch.from_numpy(ds.pkt_mask[safe] & valid[:, :, None]).to(device)
    nm = torch.from_numpy(valid).to(device)
    return xb, pkt, pm, nm


class Detector:
    def __init__(self, cfg: Config, device):
        self.cfg, self.device = cfg, device
        self.enc = BidentEncoder(cfg).to(device)
        self.learners = nn.ModuleList()
        self.names, self.Tt, self.Tc, self.pool_class = [], [], [], []

    # ------------------------------------------------------------------ #
    # stage 1: shared representation (STR + CAR + FRL), fusion-reconstruction objective
    # ------------------------------------------------------------------ #
    def fit_shared(self, ds, idx, log=print):
        cfg, enc = self.cfg, self.enc
        enc.train()
        opt = torch.optim.Adam(enc.parameters(), lr=cfg.lr)
        rng = np.random.default_rng(cfg.seed)
        for ep in range(cfg.epochs):
            t0, tot, nb = time.time(), np.zeros(3), 0
            perm = rng.permutation(idx)
            for s in range(0, len(perm), cfg.batch_size):
                b = perm[s:s + cfg.batch_size]
                if len(b) < 2:
                    continue
                out = enc(*make_batch(ds, b, self.device))
                opt.zero_grad(set_to_none=True)
                out["loss"].backward()
                nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()
                tot += [out["loss"].item(), out["lt"].item(), out["lc"].item()]
                nb += 1
            tot /= max(nb, 1)
            log(f"[shared] epoch {ep + 1:3d}/{cfg.epochs}  loss {tot[0]:.4f}  "
                f"L_t {tot[1]:.4f}  L_c {tot[2]:.4f}  ({time.time() - t0:.1f}s)")
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # stage 2: one learner per class + DAT thresholds
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _extract(self, ds, idx):
        """Frozen-encoder features of `idx`, cached on the CPU in fp16."""
        self.enc.eval()
        Zt, Zc, nm = [], [], []
        for s in range(0, len(idx), self.cfg.batch_size):
            xb, pkt, pm, m = make_batch(ds, idx[s:s + self.cfg.batch_size], self.device)
            zt, zc, _ = self.enc.features(xb, pkt, pm, m, want_desc=False)
            Zt.append(zt.half().cpu())
            Zc.append(zc.half().cpu())
            nm.append(m.cpu())
        return torch.cat(Zt), torch.cat(Zc), torch.cat(nm)

    def _fit_learner(self, cache):
        cfg, dev = self.cfg, self.device
        Zt, Zc, nm = cache
        n = Zt.shape[0]
        lrn = SingleClassLearner(cfg.d, cfg.d_c, cfg.h).to(dev)
        opt = torch.optim.Adam(lrn.parameters(), lr=cfg.ae_lr)
        steps_per_epoch = math.ceil(n / cfg.batch_size)
        epochs = max(cfg.ae_epochs, math.ceil(cfg.ae_min_steps / steps_per_epoch))
        lrn.train()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, cfg.batch_size):
                b = perm[s:s + cfg.batch_size]
                lt, lc = lrn.losses(Zt[b].to(dev).float(), Zc[b].to(dev).float(), nm[b].to(dev))
                loss = lt.mean() + lc.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        lrn.eval()
        return lrn

    @torch.no_grad()
    def _cached_losses(self, lrn, cache):
        Zt, Zc, nm = cache
        lt_all, lc_all = [], []
        for s in range(0, Zt.shape[0], self.cfg.batch_size):
            sl = slice(s, s + self.cfg.batch_size)
            lt, lc = lrn.losses(Zt[sl].to(self.device).float(), Zc[sl].to(self.device).float(),
                                nm[sl].to(self.device))
            lt_all.append(lt.cpu().numpy())
            lc_all.append(lc.cpu().numpy())
        return np.concatenate(lt_all), np.concatenate(lc_all)

    def add_class(self, name, ds, idx, pool_class=-1, log=print):
        """Train a new single-class learner on flows `idx` and calibrate its DAT thresholds.
        Existing learners / thresholds are not touched.  Returns the new learner index."""
        cfg = self.cfg
        idx = np.asarray(idx)
        if len(idx) > cfg.ae_max_samples:
            idx = np.random.default_rng(cfg.seed + len(self.names)).choice(idx, cfg.ae_max_samples, replace=False)
        t0 = time.time()
        cache = self._extract(ds, idx)
        lrn = self._fit_learner(cache)
        lt, lc = self._cached_losses(lrn, cache)
        kw = dict(alpha=cfg.alpha, k_max=cfg.k_max, max_iter=cfg.gmm_iter,
                  floor_q=cfg.thr_floor_q, seed=cfg.seed)
        Tt, Tc = fit_dat_threshold(lt, **kw), fit_dat_threshold(lc, **kw)
        self.learners.append(lrn)
        self.names.append(name)
        self.Tt.append(Tt)
        self.Tc.append(Tc)
        self.pool_class.append(int(pool_class))
        log(f"[learner] +{name:<22s} n={len(idx):5d}  T_t={Tt:.5f}  T_c={Tc:.5f}  "
            f"train-accept t/c = {(lt < Tt).mean():.2f}/{(lc < Tc).mean():.2f}  ({time.time() - t0:.1f}s)")
        return len(self.names) - 1

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score(self, ds, idx):
        """Reconstruction losses of every learner: Lt, Lc (len(idx), P) and fused descriptors."""
        self.enc.eval()
        self.learners.eval()
        idx = np.asarray(idx)
        P = len(self.learners)
        Lt = np.zeros((len(idx), P), np.float32)
        Lc = np.zeros((len(idx), P), np.float32)
        desc = []
        for s in range(0, len(idx), self.cfg.batch_size):
            e = min(s + self.cfg.batch_size, len(idx))
            xb, pkt, pm, nm = make_batch(ds, idx[s:e], self.device)
            Zt, Zc, d = self.enc.features(xb, pkt, pm, nm, want_desc=True)
            for p, lrn in enumerate(self.learners):
                lt, lc = lrn.losses(Zt, Zc, nm)
                Lt[s:e, p], Lc[s:e, p] = lt.cpu().numpy(), lc.cpu().numpy()
            desc.append(d.cpu().numpy())
        return Lt, Lc, np.concatenate(desc)

    def classify(self, ds, idx):
        Lt, Lc, desc = self.score(ds, idx)
        return dual_verify(Lt, Lc, self.Tt, self.Tc), desc

    def run_stream(self, ds, idx, discover=True, log=print):
        """Process flows `idx` in order (Algorithm 1 + buffer + incremental learning).

        Flows rejected by every learner are buffered; when the buffer holds NB flows,
        DBSCAN groups them (on the fused descriptors) and each sufficiently large cluster
        becomes a NEW single-class learner.  Returns
          pred           learner index per flow (-1 = unknown, never assigned)
          first_unknown  True if the flow was rejected by the learners existing at that time"""
        cfg, idx = self.cfg, np.asarray(idx)
        n = len(idx)
        pred = np.full(n, -1, np.int64)
        first_unknown = np.zeros(n, bool)
        buf_pos, buf_desc = np.zeros(0, np.int64), np.zeros((0, cfg.d_f))
        next_trigger = cfg.buffer

        def run_discovery():
            nonlocal buf_pos, buf_desc, next_trigger
            clusters = discover_clusters(buf_desc, cfg.eps, cfg.min_samples)
            keep = np.ones(len(buf_pos), bool)
            for c in clusters:
                if len(c) < cfg.min_cluster:
                    continue
                members = buf_pos[c]
                p = self.add_class(f"new_{len(self.names)}", ds, idx[members], -1, log)
                pred[members] = p
                keep[c] = False
            buf_pos, buf_desc = buf_pos[keep], buf_desc[keep]
            next_trigger = len(buf_pos) + cfg.buffer
            log(f"[discover] {len(clusters)} cluster(s) found, {len(buf_pos)} flows stay in the buffer")

        for s in range(0, n, cfg.batch_size):
            pos = np.arange(s, min(n, s + cfg.batch_size))
            lab, desc = self.classify(ds, idx[pos])
            pred[pos] = lab
            unk = lab < 0
            first_unknown[pos[unk]] = True
            buf_pos = np.concatenate([buf_pos, pos[unk]])
            buf_desc = np.concatenate([buf_desc, desc[unk]])
            if discover and len(buf_pos) >= next_trigger:
                run_discovery()
        if discover and len(buf_pos) >= max(cfg.min_samples, cfg.min_cluster):
            run_discovery()                                   # final flush
        return pred, first_unknown

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path):
        torch.save(dict(cfg=asdict(self.cfg), enc=self.enc.state_dict(),
                        learners=[l.state_dict() for l in self.learners], names=self.names,
                        Tt=self.Tt, Tc=self.Tc, pool_class=self.pool_class), path)

    @classmethod
    def load(cls, path, device):
        ck = torch.load(path, map_location=device)
        cfg = Config(**ck["cfg"])
        det = cls(cfg, device)
        det.enc.load_state_dict(ck["enc"])
        det.enc.eval()
        for sd in ck["learners"]:
            lrn = SingleClassLearner(cfg.d, cfg.d_c, cfg.h).to(device)
            lrn.load_state_dict(sd)
            lrn.eval()
            det.learners.append(lrn)
        det.names, det.Tt, det.Tc, det.pool_class = ck["names"], ck["Tt"], ck["Tc"], ck["pool_class"]
        return det
