"""Shared helpers for the comparison scripts: reading the data, timing, output."""

import json
import statistics
import time

import numpy as np

BINNED_SEED = dict(A=300.0, mu=5.0, sigma=0.8, c0=6.5, c1=-0.35)
UNBINNED_SEED = dict(Ns=8_000.0, mu=5.0, sigma=0.8, Nb=80_000.0, lam=0.3)
LO, HI = 0.0, 10.0
UNBINNED = [("events_100k", 1.0, 10), ("events_1M", 10.0, 3)]


def read_hist(d):
    x, y = np.loadtxt(f"{d}/peak_hist.txt", unpack=True)
    return x, y


def read_events(d, name):
    return np.loadtxt(f"{d}/{name}.txt")


def timed(repeats, fit):
    """Run `fit()` `repeats` times; return the last result, the median and the first time (ms)."""
    times, result = [], None
    for _ in range(repeats):
        t = time.perf_counter()
        result = fit()
        times.append((time.perf_counter() - t) * 1e3)
    return result, statistics.median(times), times[0]


def emit(tool, problem, params, errors, fval, nfcn, valid, ms, repeats, first_ms=None):
    out = dict(tool=tool, problem=problem, params=[float(p) for p in params],
               errors=[float(e) for e in errors], fval=float(fval), nfcn=int(nfcn),
               valid=bool(valid), time_ms=ms, repeats=repeats)
    if first_ms is not None:
        out["first_ms"] = first_ms
    print(json.dumps(out), flush=True)
