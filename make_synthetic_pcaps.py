#!/usr/bin/env python3
"""Write a few synthetic pcap files (one per traffic class) for smoke-testing the pipeline.

  python make_synthetic_pcaps.py --out_dir demo_pcaps --flows 300
"""
import argparse
import os
import random
import struct


def _ip(src, dst, proto, payload):
    total = 20 + len(payload)
    hdr = struct.pack("!BBHHHBBH4s4s", 0x45, 0, total, random.randint(0, 65535), 0x4000, 64, proto, 0,
                      bytes(src), bytes(dst))
    return hdr + payload


def _frame(src, dst, proto, sport, dport, payload):
    if proto == 6:
        l4 = struct.pack("!HHIIBBHHH", sport, dport, random.getrandbits(32), random.getrandbits(32),
                         0x50, 0x18, 65535, 0, 0) + payload
    else:
        l4 = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    return b"\x02\x00\x00\x00\x00\x01" + b"\x02\x00\x00\x00\x00\x02" + b"\x08\x00" + _ip(src, dst, proto, l4)


def _payload(kind, n):
    if kind == "text":
        return bytes(random.randint(32, 126) for _ in range(n))
    if kind == "random":
        return os.urandom(n)
    if kind == "const":
        return bytes([0xAB, 0xCD]) * (n // 2) + b"\xab" * (n % 2)
    if kind == "dns":
        return bytes(random.choice([0, 0, 0, 1, 3, 7, 0x77, 0x63]) for _ in range(n))
    return bytes(random.randint(100, 140) for _ in range(n))  # voice-like, narrow band


# name: (proto, payload kind, packet count range, size sampler, gap seconds sampler, bidirectional)
CLASSES = {
    "web":       (6,  "text",   (12, 40), lambda i: random.randint(200, 400) if i % 3 == 0 else random.randint(900, 1400),
                  lambda: random.expovariate(1 / 0.02), True),
    "keepalive": (17, "const",  (6, 12),  lambda i: random.randint(20, 40), lambda: random.uniform(0.8, 1.2), False),
    "bulk":      (6,  "random", (40, 90), lambda i: 1400 if i % 5 else 0, lambda: random.expovariate(1 / 0.001), True),
    "dns":       (17, "dns",    (2, 4),   lambda i: random.randint(30, 60) if i == 0 else random.randint(80, 140),
                  lambda: random.expovariate(1 / 0.03), True),
    "voip":      (17, "voice",  (30, 60), lambda i: 160, lambda: random.gauss(0.02, 0.001), True),
}


def write_class(path, spec, n_flows, seed):
    random.seed(seed)
    proto, kind, (lo, hi), size, gap, bidir = spec
    pkts, t = [], 1_700_000_000.0
    for _ in range(n_flows):
        t += random.expovariate(1 / 0.5)
        a = (10, 0, random.randint(0, 255), random.randint(1, 254))
        b = (192, 168, 1, random.randint(1, 254))
        sp, dp = random.randint(20000, 60000), random.choice([80, 443, 53, 5060, 8000])
        ts = t
        for i in range(random.randint(lo, hi)):
            ts += max(1e-5, gap())
            fwd = (not bidir) or (i % 2 == 0)
            n = max(0, size(i))
            f = _frame(a, b, proto, sp, dp, _payload(kind, n)) if fwd else _frame(b, a, proto, dp, sp, _payload(kind, n))
            pkts.append((ts, f))
    pkts.sort(key=lambda x: x[0])
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for ts, fr in pkts:
            sec = int(ts)
            f.write(struct.pack("<IIII", sec, int((ts - sec) * 1e6), len(fr), len(fr)) + fr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="demo_pcaps")
    ap.add_argument("--flows", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for k, (name, spec) in enumerate(CLASSES.items()):
        write_class(os.path.join(a.out_dir, name + ".pcap"), spec, a.flows, seed=k)
        print("wrote", os.path.join(a.out_dir, name + ".pcap"))
