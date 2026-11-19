"""Write the shared datasets every fitter in this comparison reads.

A Gaussian peak on a falling exponential background, as
 - `peak_hist.txt`: a 100-bin histogram on [0, 10) — lines `center count`;
 - `events_100k.txt`, `events_1M.txt`: unbinned events on [0, 10), one per line.

Fixed seed, so every run (and every tool) sees exactly the same numbers.
Usage: python make_data.py [outdir]
"""

import os
import sys

import numpy as np

LO, HI = 0.0, 10.0
MU, SIGMA, SLOPE = 5.2, 0.6, 0.4  # truth: peak position/width, background decay rate


def mixture(rng, n_sig, n_bkg):
    sig = rng.normal(MU, SIGMA, n_sig)
    # Exponential exp(-SLOPE x) truncated to [LO, HI), by inversion.
    u = rng.uniform(size=n_bkg)
    a, b = np.exp(-SLOPE * LO), np.exp(-SLOPE * HI)
    bkg = -np.log(a - u * (a - b)) / SLOPE
    x = np.concatenate([sig, bkg])
    return x[(x >= LO) & (x < HI)]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261119)

    counts, edges = np.histogram(mixture(rng, 5_000, 20_000), bins=100, range=(LO, HI))
    centers = 0.5 * (edges[:-1] + edges[1:])
    np.savetxt(os.path.join(out, "peak_hist.txt"), np.column_stack([centers, counts]), fmt="%.6f %d")

    for name, scale in [("events_100k", 1), ("events_1M", 10)]:
        x = mixture(rng, 10_000 * scale, 90_000 * scale)
        np.savetxt(os.path.join(out, f"{name}.txt"), x, fmt="%.17g")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
