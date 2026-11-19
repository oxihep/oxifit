"""Compare the fitters' outputs (JSON lines from the fit_* scripts and the
oxifit example) and write the results tables.

Usage: python report.py results/*.jsonl > results/RESULTS.md
Exits with status 1 if any valid fit's parameters differ from iminuit's by
more than 0.05 of their uncertainty — the agreement check.
"""

import json
import sys
from collections import defaultdict

TOLERANCE = 0.05  # in units of the reference uncertainty
NAMES = {
    "binned": ["A", "μ", "σ", "c₀", "c₁"],
    "unbinned": ["Nₛ", "μ", "σ", "N_b", "λ"],
}
TITLES = {
    "binned_chi2": "Binned χ² (100 bins)",
    "binned_likelihood": "Binned Poisson likelihood (100 bins)",
    "unbinned_events_100k": "Extended unbinned likelihood, 10⁵ events",
    "unbinned_events_1M": "Extended unbinned likelihood, 10⁶ events",
}

fits = defaultdict(list)
for path in sys.argv[1:]:
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            fits[d["problem"]].append(d)

failures = []
for problem, title in TITLES.items():
    rows = fits.get(problem, [])
    ref = next((r for r in rows if r["tool"] == "iminuit"), None)
    if ref is None:
        continue
    names = NAMES["binned" if problem.startswith("binned") else "unbinned"]
    print(f"### {title}\n")
    print("| Tool | valid | max \\|Δθ\\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r["time_ms"]):
        pulls = [abs(a - b) / e for a, b, e in zip(r["params"], ref["params"], ref["errors"])]
        ratios = [a / b for a, b in zip(r["errors"], ref["errors"])]
        worst = max(pulls)
        if r["valid"] and worst > TOLERANCE:
            failures.append(f"{r['tool']} on {problem}: {worst:.3f}σ")
        t = r["time_ms"]
        time = f"{t:.3f} ms" if t < 10 else f"{t:.0f} ms"
        if "first_ms" in r:
            time += f" (first {r['first_ms']/1e3:.1f} s)"
        lo, hi = min(ratios), max(ratios)
        span = f"{lo:.4f}" if abs(hi - lo) < 5e-5 else f"{lo:.4f} – {hi:.4f}"
        calls = r["nfcn"] if r["nfcn"] >= 0 else "—"
        print(f"| {r['tool']} | {'yes' if r['valid'] else '**no**'} | {worst:.1e} | {span} "
              f"| {calls} | {time} | {ref['time_ms'] / t:.2f}× |")
    print()
    print("Reference values (iminuit): " + ", ".join(
        f"{n} = {v:.6g} ± {e:.3g}" for n, v, e in zip(names, ref["params"], ref["errors"])) + "\n")

if failures:
    print("**Agreement check failed:** " + "; ".join(failures))
    sys.exit(1)
print(f"All valid fits agree with iminuit to within {TOLERANCE}σ on every parameter.")
