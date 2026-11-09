"""Torch-free parts of Bident: Distribution-based Adaptive Threshold (DAT),
dual verification (Algorithm 1), unknown-class discovery (DBSCAN) and evaluation
(Hungarian matching of discovered clusters to the true unknown classes)."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.metrics import (adjusted_mutual_info_score, precision_recall_fscore_support)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# DAT  (Eq. 26-29)
# --------------------------------------------------------------------------- #
def fit_dat_threshold(losses, alpha=2.0, k_max=7, max_iter=100, floor_q=0.9, seed=0):
    """Class- and branch-specific threshold from the reconstruction losses of the
    training samples of ONE class.

    1. fit a 1-D GMM, K in 1..k_max chosen by BIC            (Eq. 26)
    2. take the component with the largest mean              (Eq. 27)
    3. T = mu - alpha * sigma of that component              (Eq. 28)
    4. T = max(T, min(losses))                               (Eq. 29)
    5. (extra) T = max(T, quantile(losses, floor_q)); guards against a degenerate
       fit (e.g. K=1, where mu-2*sigma would reject ~98 % of the class itself).
       floor_q=0 gives the paper-literal behaviour.
    A sample is accepted when loss < T.
    """
    x = np.asarray(losses, np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    if x.size < 20:                       # too few points for a mixture model
        return float(np.nextafter(x.max(), np.inf))
    s = x.std() + 1e-12
    z = (x / s).reshape(-1, 1)            # work in standardised units
    best, best_bic = None, np.inf
    for k in range(1, min(k_max, max(1, x.size // 10)) + 1):
        try:
            g = GaussianMixture(k, covariance_type="full", max_iter=max_iter, n_init=2,
                                reg_covar=1e-6, random_state=seed).fit(z)
        except Exception:
            continue
        bic = g.bic(z)
        if bic < best_bic:
            best, best_bic = g, bic
    if best is None:
        T = np.quantile(x, 0.95)
    else:
        mu, sd = best.means_.ravel(), np.sqrt(best.covariances_.ravel())
        k = int(np.argmax(mu))
        T = (mu[k] - alpha * sd[k]) * s
    T = max(T, x.min())
    if floor_q > 0:
        T = max(T, float(np.quantile(x, floor_q)))
    return float(T)


# --------------------------------------------------------------------------- #
# Dual verification  (Algorithm 1)
# --------------------------------------------------------------------------- #
def dual_verify(Lt, Lc, Tt, Tc):
    """Lt, Lc: (B, P) temporal / contextual reconstruction losses of the P learners.
    Tt, Tc: (P,) thresholds.  Returns (B,) predicted learner index, -1 = unknown.

    A sample is assigned to a known class only if the best temporal candidate and the
    best contextual candidate (each below its own threshold) come from the same learner."""
    Lt, Lc = np.asarray(Lt), np.asarray(Lc)
    B = Lt.shape[0]
    if Lt.shape[1] == 0:
        return np.full(B, -1, np.int64)
    okt = Lt < np.asarray(Tt)[None, :]
    okc = Lc < np.asarray(Tc)[None, :]
    it = np.where(okt, Lt, np.inf).argmin(1)
    ic = np.where(okc, Lc, np.inf).argmin(1)
    accept = okt.any(1) & okc.any(1) & (it == ic)
    return np.where(accept, it, -1).astype(np.int64)


# --------------------------------------------------------------------------- #
# Unknown class discovery
# --------------------------------------------------------------------------- #
def discover_clusters(desc, eps=0.0, min_samples=5):
    """DBSCAN on the fused descriptors of the rejected samples.
    Returns a list of index arrays (one per cluster, noise excluded)."""
    desc = np.asarray(desc, np.float64)
    n = len(desc)
    if n < max(min_samples, 2):
        return []
    x = StandardScaler().fit_transform(desc)
    if eps <= 0:                          # automatic: 90th percentile of the k-NN distance
        k = min(min_samples, n - 1)
        dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
        eps = float(np.percentile(dist[:, -1], 90)) + 1e-12
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
    return [np.where(lab == c)[0] for c in sorted(set(lab.tolist())) if c != -1]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_open_set(y_true, pred, pool_class, known_classes, n_classes):
    """y_true: ground-truth class index per test flow.
    pred: learner index per flow (-1 = rejected and never assigned to a discovered class).
    pool_class[p]: ground-truth class index of learner p, or -1 for a discovered learner.
    Discovered clusters are matched to the true unknown classes with the Hungarian
    algorithm (maximising the number of correctly assigned flows)."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    pool_class = np.asarray(pool_class, np.int64)
    known = set(int(c) for c in known_classes)
    disc = [p for p in range(len(pool_class)) if pool_class[p] < 0]
    unk = [c for c in range(n_classes) if c not in known]

    mapping = {}
    if disc and unk:
        C = np.array([[np.sum((pred == p) & (y_true == u)) for u in unk] for p in disc])
        rows, cols = linear_sum_assignment(-C)
        mapping = {disc[r]: unk[c] for r, c in zip(rows, cols) if C[r, c] > 0}

    lut = np.full(len(pool_class) + 1, -2, np.int64)       # -2: unmatched cluster
    for p in range(len(pool_class)):
        lut[p] = pool_class[p] if pool_class[p] >= 0 else mapping.get(p, -2)
    y_map = np.where(pred >= 0, lut[np.maximum(pred, 0)], -1)

    labels = list(range(n_classes))
    P, R, F, _ = precision_recall_fscore_support(y_true, y_map, labels=labels,
                                                 average="macro", zero_division=0)
    _, _, Fc, _ = precision_recall_fscore_support(y_true, y_map, labels=labels,
                                                  average=None, zero_division=0)
    out = dict(acc=float((y_map == y_true).mean()), precision=float(P), recall=float(R),
               f1=float(F), ami=float(adjusted_mutual_info_score(y_true, pred)),
               per_class_f1={int(c): float(f) for c, f in zip(labels, Fc)},
               cluster_to_class={int(k): int(v) for k, v in mapping.items()},
               n_discovered=len(disc))

    # class-agnostic known/unknown decision: a flow counts as "unknown" when it was not
    # accepted by one of the initially known classes (rejected, or put in a discovered cluster)
    unk_true = ~np.isin(y_true, list(known))
    unk_pred = np.where(pred >= 0, pool_class[np.maximum(pred, 0)] < 0, True)
    if unk_true.any() and (~unk_true).any():
        p, r, f, _ = precision_recall_fscore_support(unk_true, unk_pred, average="binary",
                                                     zero_division=0)
        out["unknown_detection"] = dict(acc=float((unk_true == unk_pred).mean()),
                                        precision=float(p), recall=float(r), f1=float(f))
    return out
