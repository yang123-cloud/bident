"""Pure-Python pcap / pcapng reader and flow feature extraction.

No scapy / dpkt needed.  Every pcap file in a folder is one traffic class and the
class name is the file name without extension.

For every flow (bidirectional 5-tuple, split by an idle timeout) we keep
  * X        first n_pkt packets x pkt_len bytes (IP header onwards), uint8      -> STR
  * pkt      first m packets: [signed length / 1500, log1p(interval_ms) / 10]    -> CAR
  * ctx      indices of the n_ctx flows that END at this flow in the same pcap
             (only flows that started EARLIER; the target flow is the last entry,
             -1 = padding)                                                       -> CAR
"""
import json
import math
import os
import struct
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# pcap / pcapng readers
# --------------------------------------------------------------------------- #
def _iter_pcap(f):
    hdr = f.read(24)
    if len(hdr) < 24:
        return
    magic = hdr[:4]
    if magic == b"\xd4\xc3\xb2\xa1":
        end, scale = "<", 1e-6
    elif magic == b"\xa1\xb2\xc3\xd4":
        end, scale = ">", 1e-6
    elif magic == b"\x4d\x3c\xb2\xa1":
        end, scale = "<", 1e-9
    elif magic == b"\xa1\xb2\x3c\x4d":
        end, scale = ">", 1e-9
    else:
        raise ValueError("not a pcap file")
    linktype = struct.unpack(end + "I", hdr[20:24])[0] & 0x0FFFFFFF
    rec = struct.Struct(end + "IIII")
    while True:
        h = f.read(16)
        if len(h) < 16:
            break
        sec, frac, caplen, _ = rec.unpack(h)
        data = f.read(caplen)
        if len(data) < caplen:
            break
        yield sec + frac * scale, linktype, data


def _iter_pcapng(f):
    end = "<"
    ifaces = []  # (linktype, seconds per tick)
    while True:
        h = f.read(8)
        if len(h) < 8:
            return
        if h[:4] == b"\x0a\x0d\x0d\x0a":  # section header block
            bom = f.read(4)
            if len(bom) < 4:
                return
            end = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            blen = struct.unpack(end + "I", h[4:8])[0]
            f.read(max(0, blen - 12))
            ifaces = []
            continue
        btype, blen = struct.unpack(end + "II", h)
        body = f.read(blen - 8)
        if len(body) < blen - 8:
            return
        body = body[:-4]
        if btype == 1 and len(body) >= 8:  # interface description block
            lt = struct.unpack(end + "H", body[:2])[0]
            scale, off = 1e-6, 8
            while off + 4 <= len(body):
                code, olen = struct.unpack(end + "HH", body[off:off + 4])
                if code == 0:
                    break
                if code == 9 and olen >= 1:  # if_tsresol
                    v = body[off + 4]
                    scale = 2.0 ** -(v & 0x7F) if v & 0x80 else 10.0 ** -v
                off += 4 + ((olen + 3) & ~3)
            ifaces.append((lt, scale))
        elif btype == 6 and len(body) >= 20:  # enhanced packet block
            ifid, th, tl, caplen, _ = struct.unpack(end + "IIIII", body[:20])
            if ifid >= len(ifaces):
                continue
            lt, scale = ifaces[ifid]
            yield ((th << 32) | tl) * scale, lt, body[20:20 + caplen]


def iter_packets(path):
    """Yield (timestamp, linktype, raw frame bytes)."""
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(0)
        if head == b"\x0a\x0d\x0d\x0a":
            yield from _iter_pcapng(f)
        else:
            yield from _iter_pcap(f)


# --------------------------------------------------------------------------- #
# L2 / L3 parsing
# --------------------------------------------------------------------------- #
def _ip_offset(linktype, d):
    if linktype == 1:  # Ethernet (+ up to a few VLAN tags)
        if len(d) < 14:
            return None
        et, off = (d[12] << 8) | d[13], 14
        while et in (0x8100, 0x88A8, 0x9100) and len(d) >= off + 4:
            et, off = (d[off + 2] << 8) | d[off + 3], off + 4
        return off if et in (0x0800, 0x86DD) else None
    if linktype in (101, 228, 229, 12, 14):  # raw IP
        return 0
    if linktype == 113:  # Linux cooked v1
        if len(d) < 16:
            return None
        return 16 if ((d[14] << 8) | d[15]) in (0x0800, 0x86DD) else None
    if linktype == 276:  # Linux cooked v2
        if len(d) < 20:
            return None
        return 20 if ((d[0] << 8) | d[1]) in (0x0800, 0x86DD) else None
    if linktype in (0, 108):  # BSD loopback
        return 4 if len(d) > 4 else None
    return None


def _parse_ip(d, off):
    """-> (proto, src, dst, sport, dport, l4_offset, total_len, ip_version, has_ports)"""
    if len(d) <= off:
        return None
    ver = d[off] >> 4
    frag = 0
    if ver == 4:
        if len(d) < off + 20:
            return None
        l4 = (d[off] & 0x0F) * 4
        if l4 < 20:
            return None
        total = (d[off + 2] << 8) | d[off + 3]
        proto = d[off + 9]
        frag = ((d[off + 6] & 0x1F) << 8) | d[off + 7]
        src, dst = bytes(d[off + 12:off + 16]), bytes(d[off + 16:off + 20])
    elif ver == 6:
        if len(d) < off + 40:
            return None
        total = 40 + ((d[off + 4] << 8) | d[off + 5])
        proto = d[off + 6]
        src, dst = bytes(d[off + 8:off + 24]), bytes(d[off + 24:off + 40])
        l4 = 40
        for _ in range(4):  # skip a few extension headers
            if proto in (0, 43, 60) and len(d) >= off + l4 + 2:
                nxt = d[off + l4]
                l4 += (d[off + l4 + 1] + 1) * 8
                proto = nxt
            elif proto == 44 and len(d) >= off + l4 + 8:
                nxt = d[off + l4]
                frag = ((d[off + l4 + 2] << 8) | d[off + l4 + 3]) >> 3
                l4 += 8
                proto = nxt
            else:
                break
    else:
        return None
    if total <= 0:          # e.g. IPv4 total length 0 with segmentation offload
        total = len(d) - off
    has_ports = proto in (6, 17) and frag == 0 and len(d) >= off + l4 + 4
    sport = dport = 0
    if has_ports:
        sport = (d[off + l4] << 8) | d[off + l4 + 1]
        dport = (d[off + l4 + 2] << 8) | d[off + l4 + 3]
    return proto, src, dst, sport, dport, l4, total, ver, has_ports


# --------------------------------------------------------------------------- #
# flow assembly
# --------------------------------------------------------------------------- #
class _Flow:
    __slots__ = ("start", "last", "first", "arr", "n", "lens", "ts", "dirs")

    def __init__(self, ts, first, n_pkt, pkt_len):
        self.start = self.last = ts
        self.first = first
        self.arr = np.zeros((n_pkt, pkt_len), np.uint8)
        self.n = 0
        self.lens, self.ts, self.dirs = [], [], []


def read_flows(path, cfg):
    """Return dict(X, pkt, pkt_mask, start) for the flows of one pcap, ordered by start time."""
    n_pkt, pkt_len, m = cfg.n_pkt, cfg.pkt_len, cfg.m
    assert m <= n_pkt
    active, done, created = {}, [], 0

    def finish(fl):
        if fl.n >= cfg.min_pkts:
            done.append(fl)

    for ts, lt, data in iter_packets(path):
        off = _ip_offset(lt, data)
        if off is None:
            continue
        info = _parse_ip(data, off)
        if info is None:
            continue
        proto, src, dst, sport, dport, l4, total, ver, has_ports = info
        a, b = (src, sport), (dst, dport)
        key = (proto, a, b) if a <= b else (proto, b, a)
        fl = active.get(key)
        if fl is not None and ts - fl.last > cfg.idle_timeout:
            finish(fl)
            del active[key]
            fl = None
        if fl is None:
            if created >= cfg.max_flows:
                break
            fl = _Flow(ts, a, n_pkt, pkt_len)
            active[key] = fl
            created += 1
        fl.last = ts
        if fl.n < n_pkt:
            take = min(pkt_len, len(data) - off, total)
            row = fl.arr[fl.n]
            row[:take] = np.frombuffer(data, np.uint8, take, off)
            if cfg.mask_addr:
                if ver == 4:
                    row[10:20] = 0          # IP checksum + src + dst
                else:
                    row[8:40] = 0           # src + dst
                if has_ports:
                    row[l4:l4 + 4] = 0      # transport ports
            if fl.n < m:
                fl.lens.append(total)
                fl.ts.append(ts)
                fl.dirs.append(1 if a == fl.first else -1)
        fl.n += 1
    for fl in active.values():
        finish(fl)
    done.sort(key=lambda f: f.start)

    nf = len(done)
    X = np.zeros((nf, n_pkt * pkt_len), np.uint8)
    pkt = np.zeros((nf, m, 2), np.float32)
    pm = np.zeros((nf, m), bool)
    start = np.zeros(nf, np.float64)
    for i, fl in enumerate(done):
        X[i] = fl.arr.reshape(-1)
        start[i] = fl.start
        for j in range(len(fl.lens)):
            gap = 0.0 if j == 0 else max(0.0, fl.ts[j] - fl.ts[j - 1])
            pkt[i, j, 0] = fl.dirs[j] * min(fl.lens[j], 1500) / 1500.0
            pkt[i, j, 1] = math.log1p(gap * 1000.0) / 10.0
            pm[i, j] = True
    return dict(X=X, pkt=pkt, pkt_mask=pm, start=start)


def build_ctx(n, base, n_ctx):
    """Causal context windows: row i = [i-(n_ctx-1), ..., i] (global indices, -1 = padding)."""
    rel = np.arange(n)[:, None] + np.arange(-(n_ctx - 1), 1)[None, :]
    return np.where(rel >= 0, rel + base, -1).astype(np.int32)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
@dataclass
class FlowData:
    X: np.ndarray          # (N, n_pkt*pkt_len) uint8
    pkt: np.ndarray        # (N, m, 2) float32
    pkt_mask: np.ndarray   # (N, m) bool
    ctx: np.ndarray        # (N, n_ctx) int32
    start: np.ndarray      # (N,) float64
    label: np.ndarray      # (N,) int32   index into class_names (file index)
    class_names: list

    def __len__(self):
        return len(self.label)


def list_pcaps(path):
    if os.path.isfile(path):
        return [path]
    files = [os.path.join(path, f) for f in sorted(os.listdir(path))
             if f.lower().endswith((".pcap", ".pcapng", ".cap"))]
    if not files:
        raise FileNotFoundError(f"no .pcap/.pcapng/.cap files in {path}")
    return files


def build_dataset(path, cfg, cache=None, log=print):
    if cache and not cache.endswith(".npz"):
        cache += ".npz"
    files = list_pcaps(path)
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    if len(set(names)) != len(names):
        raise ValueError("two pcap files share the same class name: " + str(names))
    meta = json.dumps(dict(cfg=cfg.data_key(),
                           files=[(os.path.basename(f), os.path.getsize(f)) for f in files]),
                      sort_keys=True)
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if str(z["meta"]) == meta:
            log(f"[data] loaded cache {cache}")
            return FlowData(z["X"], z["pkt"], z["pkt_mask"], z["ctx"], z["start"], z["label"],
                            str(z["names"]).split("\n"))
    parts, kept_names, base = [], [], 0
    for f, nm in zip(files, names):
        r = read_flows(f, cfg)
        n = len(r["start"])
        log(f"[data] {nm:<24s} {n:6d} flows")
        if n == 0:
            log(f"[data] WARNING: {nm} has no usable flow, skipped")
            continue
        r["ctx"] = build_ctx(n, base, cfg.n_ctx)
        r["label"] = np.full(n, len(kept_names), np.int32)
        parts.append(r)
        kept_names.append(nm)
        base += n
    if not parts:
        raise RuntimeError("no flows extracted")
    cat = {k: np.concatenate([p[k] for p in parts]) for k in
           ("X", "pkt", "pkt_mask", "ctx", "start", "label")}
    ds = FlowData(class_names=kept_names, **cat)
    if cache:
        np.savez(cache, meta=meta, names="\n".join(kept_names), **cat)
        log(f"[data] cache written to {cache}")
    return ds
