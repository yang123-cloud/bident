"""Hyper-parameters. Defaults follow Table III of the Bident paper; the entries
marked (extra) are not specified in the paper and are implementation choices."""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- STR: Strided Temporal Representation -------------------------------
    n_pkt: int = 64            # Np  packets taken from each flow
    pkt_len: int = 128         # Lp  bytes kept per packet (IP header onwards)
    stride: int = 8            # Ls  stride length (must divide pkt_len)
    d: int = 256               # d   hidden dim of the SSM (= temporal dim d_t)
    d_state: int = 16          # (extra) SSM state size N
    d_conv: int = 4            # (extra) causal conv kernel
    ssm_layers: int = 1        # (extra) number of SSM blocks
    scan_chunk: int = 64       # (extra) chunk length for the checkpointed scan
    use_mamba_ssm: bool = False  # (extra) use the CUDA `mamba_ssm` package if installed

    # ---- CAR: Contextual Attention Representation ---------------------------
    m: int = 10                # m   packets per session used by multi-instance embedding
    d_c: int = 128             # dc  hidden dim of CAR
    n_ctx: int = 64            # Nc  sessions in the contextual subgraph (target + Nc-1 predecessors)
    heads: int = 4             # (extra) attention heads

    # ---- FRL: Fusion-Reconstruction Learner ---------------------------------
    d_f: int = 128             # (extra) fused dimension df
    anchor_w: float = 1.0      # (extra) weight of the anti-collapse anchor loss (0 = paper-literal)

    # ---- single-class AutoEncoder learner -----------------------------------
    h: int = 128               # h   hidden dim of the AutoEncoders

    # ---- DAT: Distribution-based Adaptive Threshold -------------------------
    k_max: int = 7             # Kmax GMM components (chosen by BIC)
    gmm_iter: int = 100        # I   GMM fitting iterations
    alpha: float = 2.0         # alpha  boundary strictness, T = mu - alpha*sigma
    thr_floor_q: float = 0.9   # (extra) never reject more than 1-q of the training losses (0 = paper-literal)

    # ---- unknown-class discovery -------------------------------------------
    buffer: int = 100          # NB  buffer size that triggers DBSCAN
    eps: float = 0.0           # DBSCAN eps (0 = automatic k-distance heuristic)
    min_samples: int = 5       # DBSCAN min_samples
    min_cluster: int = 10      # (extra) smallest cluster that becomes a new learner

    # ---- optimisation --------------------------------------------------------
    batch_size: int = 64
    epochs: int = 30           # epochs for the shared STR/CAR/FRL training
    lr: float = 1e-4
    dropout: float = 0.2
    ae_epochs: int = 30        # (extra) epochs for each single-class learner
    ae_lr: float = 1e-3        # (extra)
    ae_min_steps: int = 300    # (extra) minimum optimisation steps for a learner (small clusters)
    ae_max_samples: int = 1500  # (extra) cap on samples used to fit/calibrate one learner

    # ---- data / protocol ----------------------------------------------------
    min_pkts: int = 2          # drop flows shorter than this
    idle_timeout: float = 120.0  # seconds of silence that closes a flow
    max_flows: int = 2000      # max flows read from each pcap (earliest first)
    mask_addr: bool = True     # zero IP addresses / ports / IP checksum (avoid shortcut learning)
    test_ratio: float = 0.3
    n_known: int = 1           # number of initially known classes (paper: 1)
    known: str = ""            # comma separated class names; overrides n_known
    seed: int = 0
    device: str = "auto"

    # parameters that change the extracted features (used as cache key)
    def data_key(self):
        return dict(n_pkt=self.n_pkt, pkt_len=self.pkt_len, m=self.m, n_ctx=self.n_ctx,
                    min_pkts=self.min_pkts, idle_timeout=self.idle_timeout,
                    max_flows=self.max_flows, mask_addr=self.mask_addr)
