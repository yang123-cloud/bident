"""Bident network modules (PyTorch).

  STR  Strided Temporal Representation      Eq. (1)-(9)
  CAR  Contextual Attention Representation  Eq. (10)-(14)
  FRL  Fusion-Reconstruction Learner        Eq. (15)-(25)
  SingleClassLearner  per-class temporal + contextual AutoEncoders (Fig. 4)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# --------------------------------------------------------------------------- #
# STR
# --------------------------------------------------------------------------- #
def _scan_chunk(h, x, delta, Bm, Cm, A):
    """Sequential selective scan over one chunk (Eq. 7-8).
    h: (B,d,n) state; x, delta: (B,T,d); Bm, Cm: (B,T,n); A: (d,n) (negative)."""
    ys = []
    for t in range(x.shape[1]):
        dt = delta[:, t]                                         # (B,d)
        dA = torch.exp(dt.unsqueeze(-1) * A)                     # A_bar = exp(delta*A)
        dBx = (dt * x[:, t]).unsqueeze(-1) * Bm[:, t].unsqueeze(1)  # B_bar x = delta*B*x
        h = dA * h + dBx                                         # h_t = A_bar h_{t-1} + B_bar x_t
        ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))           # y_t = C h_t
    return torch.stack(ys, 1), h


class SSMBlock(nn.Module):
    """Unidirectional selective state-space block, pure PyTorch (Eq. 4-9).
    Linear time in the sequence length; the scan is gradient-checkpointed per chunk so
    that memory stays small.  (Install `mamba_ssm` and pass --use_mamba_ssm for the
    fused CUDA kernel, which is much faster.)"""

    def __init__(self, d, d_state=16, d_conv=4, chunk=64):
        super().__init__()
        self.d, self.n, self.chunk = d, d_state, chunk
        self.dt_rank = max(1, math.ceil(d / 16))
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * d, bias=False)                 # W_x, W_z
        self.conv = nn.Conv1d(d, d, d_conv, groups=d, padding=d_conv - 1)
        self.x_proj = nn.Linear(d, self.dt_rank + 2 * d_state, bias=False)  # W_delta(low rank), W_B, W_C
        self.dt_proj = nn.Linear(self.dt_rank, d)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d, 1))
        self.out_proj = nn.Linear(d, d, bias=False)                    # W_o
        # initialise delta in [1e-3, 1e-1] as in Mamba
        dt = torch.exp(torch.rand(d) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, h):                                              # h: (B,L,d)
        L = h.shape[1]
        x, z = self.in_proj(self.norm(h)).chunk(2, dim=-1)             # Eq. 4
        x = F.silu(self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2))  # Eq. 5 (causal)
        dt, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, self.n, self.n], dim=-1)
        delta = F.softplus(self.dt_proj(dt))                           # Eq. 6
        A = -torch.exp(self.A_log)
        state = x.new_zeros(x.shape[0], self.d, self.n)
        ys = []
        for s in range(0, L, self.chunk):
            e = min(s + self.chunk, L)
            args = (state, x[:, s:e], delta[:, s:e], Bm[:, s:e], Cm[:, s:e], A)
            if torch.is_grad_enabled() and x.requires_grad:
                y, state = checkpoint(_scan_chunk, *args, use_reentrant=False)
            else:
                y, state = _scan_chunk(*args)
            ys.append(y)
        y = torch.cat(ys, 1) * F.silu(z)                               # gate
        return self.out_proj(y) + h                                    # Eq. 9 (+ residual)


class MambaBlock(nn.Module):
    """Same block built on the fused CUDA kernel of the `mamba_ssm` package (optional)."""

    def __init__(self, d, d_state=16, d_conv=4):
        super().__init__()
        from mamba_ssm import Mamba
        self.norm = nn.LayerNorm(d)
        self.mamba = Mamba(d_model=d, d_state=d_state, d_conv=d_conv, expand=1)

    def forward(self, h):
        return self.mamba(self.norm(h)) + h


class STR(nn.Module):
    """bytes (B, Np*Lp) in [0,1]  ->  Zt (B, Ns, d),  Ns = Np*Lp/Ls."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.pkt_len % cfg.stride == 0, "stride must divide pkt_len (strides must not cross packets)"
        self.Ls = cfg.stride
        self.Ns = cfg.n_pkt * cfg.pkt_len // cfg.stride
        self.emb = nn.Linear(cfg.stride, cfg.d, bias=False)            # W_emb
        self.pos = nn.Parameter(torch.zeros(1, self.Ns, cfg.d))        # E_pos
        nn.init.trunc_normal_(self.pos, std=0.02)
        if cfg.use_mamba_ssm:
            blocks = [MambaBlock(cfg.d, cfg.d_state, cfg.d_conv) for _ in range(cfg.ssm_layers)]
        else:
            blocks = [SSMBlock(cfg.d, cfg.d_state, cfg.d_conv, cfg.scan_chunk) for _ in range(cfg.ssm_layers)]
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(cfg.d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, xb):
        S = xb.reshape(xb.shape[0], self.Ns, self.Ls)                  # Eq. 1-2: stride cutting
        H = self.drop(self.emb(S) + self.pos)                          # Eq. 3
        for blk in self.blocks:
            H = blk(H)
        return self.out_norm(H), S


# --------------------------------------------------------------------------- #
# CAR
# --------------------------------------------------------------------------- #
def _ffn(d, p):
    return nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(p), nn.Linear(2 * d, d))


class CAR(nn.Module):
    """Multi-instance packet attention inside a session, then attention across the
    sessions of the contextual subgraph.
    pkt (B,Nc,m,2), pkt_mask (B,Nc,m), node_mask (B,Nc)  ->  Zc (B,Nc,dc)"""

    def __init__(self, cfg):
        super().__init__()
        d, p = cfg.d_c, cfg.dropout
        self.embed = nn.Sequential(nn.Linear(2, d), nn.ReLU(), nn.Linear(d, d))   # Eq. 10
        self.pkt_attn = nn.MultiheadAttention(d, cfg.heads, dropout=p, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn1 = _ffn(d, p)
        self.ctx_attn = nn.MultiheadAttention(d, cfg.heads, dropout=p, batch_first=True)
        self.ln3, self.ln4 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn2 = _ffn(d, p)

    def forward(self, pkt, pkt_mask, node_mask):
        B, Nc, m, _ = pkt.shape
        pm = pkt_mask.clone()
        pm[..., 0] = True                       # keep >=1 key per (padded) session -> no NaN
        x = self.embed(pkt.reshape(B * Nc, m, 2))
        a, _ = self.pkt_attn(x, x, x, key_padding_mask=~pm.reshape(B * Nc, m), need_weights=False)
        x = self.ln1(x + a)                                            # Eq. 11
        x = self.ln2(x + self.ffn1(x))                                 # Eq. 12
        w = pm.reshape(B * Nc, m, 1).to(x.dtype)
        node = ((x * w).sum(1) / w.sum(1)).reshape(B, Nc, -1)          # Eq. 13 (masked mean pool)
        a, _ = self.ctx_attn(node, node, node, key_padding_mask=~node_mask, need_weights=False)  # Eq. 14
        z = self.ln3(node + a)
        z = self.ln4(z + self.ffn2(z))
        return z * node_mask.unsqueeze(-1).to(z.dtype)


# --------------------------------------------------------------------------- #
# FRL
# --------------------------------------------------------------------------- #
class FRL(nn.Module):
    """Fuse temporal (queries) and contextual (keys/values) features and reconstruct BOTH
    views from the fused representation instead of classifying."""

    def __init__(self, Ns, Nc, d_t, d_c, d_f, heads, dropout):
        super().__init__()
        self.d_f = d_f
        self.Wq, self.Wk, self.Wv = nn.Linear(d_t, d_f), nn.Linear(d_c, d_f), nn.Linear(d_c, d_f)  # Eq. 15
        layer = nn.TransformerEncoderLayer(d_f, heads, 2 * d_f, dropout, batch_first=True, norm_first=True)
        self.trans = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.phi = nn.Parameter(torch.randn(Ns, d_f) / math.sqrt(d_f))   # temporal reference vectors
        self.psi = nn.Parameter(torch.randn(Nc, d_f) / math.sqrt(d_f))   # contextual reference vectors
        self.P_t, self.b_t = nn.Linear(d_f, d_t, bias=False), nn.Parameter(torch.zeros(Ns, d_t))
        self.P_c, self.b_c = nn.Linear(d_f, d_c, bias=False), nn.Parameter(torch.zeros(Nc, d_c))

    def fuse(self, Zt, Zc, node_mask):
        q, k, v = self.Wq(Zt), self.Wk(Zc), self.Wv(Zc)
        logit = q @ k.transpose(1, 2) / math.sqrt(self.d_f)             # (B,Ns,Nc)
        logit = logit.masked_fill(~node_mask.unsqueeze(1), float("-inf"))
        alpha = logit.softmax(-1)                                       # Eq. 16
        return self.trans(q + alpha @ v)                                # Eq. 17-18  -> (B,Ns,df)

    def reconstruct(self, E):
        gamma = (E @ self.phi.t()).softmax(-1)                          # Eq. 19  (B,Ns,Ns)
        Zt_hat = self.P_t(gamma @ E) + self.b_t                         # Eq. 20-21
        omega = (self.psi @ E.transpose(1, 2)).softmax(-1)              # Eq. 22  (B,Nc,Ns)
        Zc_hat = self.P_c(omega @ E) + self.b_c                         # Eq. 23-24
        return Zt_hat, Zc_hat


class BidentEncoder(nn.Module):
    """Shared representation: STR + CAR + FRL (trained once, then frozen)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.str_ = STR(cfg)
        self.car = CAR(cfg)
        self.frl = FRL(self.str_.Ns, cfg.n_ctx, cfg.d, cfg.d_c, cfg.d_f, cfg.heads, cfg.dropout)
        # anchor heads (extra): keep Zt / Zc informative about their raw inputs so the
        # self-reconstruction objective cannot be solved by collapsing the targets.
        self.head_t = nn.Linear(cfg.d, cfg.stride)
        self.head_c = nn.Linear(cfg.d_c, cfg.m * 2)

    def features(self, xb, pkt, pkt_mask, node_mask, want_desc=True):
        Zt, _ = self.str_(xb)
        Zc = self.car(pkt, pkt_mask, node_mask)
        desc = self.frl.fuse(Zt, Zc, node_mask).mean(1) if want_desc else None   # Pool(E_f)
        return Zt, Zc, desc

    def forward(self, xb, pkt, pkt_mask, node_mask):
        Zt, S = self.str_(xb)
        Zc = self.car(pkt, pkt_mask, node_mask)
        E = self.frl.fuse(Zt, Zc, node_mask)
        Zt_hat, Zc_hat = self.frl.reconstruct(E)
        w = node_mask.float()
        # Eq. 25 (targets are detached: the reconstruction target is not pulled towards the prediction)
        lt = ((Zt_hat - Zt.detach()) ** 2).mean((1, 2))
        lc = (((Zc_hat - Zc.detach()) ** 2).mean(-1) * w).sum(1) / w.sum(1)
        # anti-collapse anchors
        at = ((self.head_t(Zt) - S) ** 2).mean((1, 2))
        tgt = (pkt * pkt_mask.unsqueeze(-1)).flatten(2)                 # (B,Nc,m*2)
        ac = (((self.head_c(Zc) - tgt) ** 2).mean(-1) * w).sum(1) / w.sum(1)
        loss = (lt + lc).mean() + self.cfg.anchor_w * (at + ac).mean()
        return dict(loss=loss, lt=lt.mean().detach(), lc=lc.mean().detach())


# --------------------------------------------------------------------------- #
# single-class learner
# --------------------------------------------------------------------------- #
def _mlp_ae(d, h):
    return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h // 2), nn.ReLU(),
                         nn.Linear(h // 2, h), nn.ReLU(), nn.Linear(h, d))


class SingleClassLearner(nn.Module):
    """Two independent fully-connected AutoEncoders (ReLU) for ONE known class:
    one reconstructs the temporal features Zt (token-wise), one the contextual features Zc."""

    def __init__(self, d_t, d_c, h):
        super().__init__()
        self.ae_t, self.ae_c = _mlp_ae(d_t, h), _mlp_ae(d_c, h)

    def losses(self, Zt, Zc, node_mask):
        """per-sample MSE reconstruction losses (L_t, L_c), each of shape (B,)"""
        lt = ((self.ae_t(Zt) - Zt) ** 2).mean((1, 2))
        w = node_mask.float()
        lc = (((self.ae_c(Zc) - Zc) ** 2).mean(-1) * w).sum(1) / w.sum(1)
        return lt, lc
