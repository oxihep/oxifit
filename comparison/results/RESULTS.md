### Binned χ² (100 bins)

| Tool | valid | max \|Δθ\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |
|---|---|---|---|---|---|---|
| scipy | yes | 3.0e-03 | 0.9990 – 1.0368 | 31 | 0.149 ms | 6.70× |
| oxifit-closure+rayon | yes | 2.8e-07 | 1.0000 – 1.0001 | 185 | 0.185 ms | 5.40× |
| ROOT-Minuit2 | yes | 3.0e-03 | 0.9702 – 0.9994 | 112 | 0.192 ms | 5.21× |
| oxifit-closure | yes | 2.8e-07 | 1.0000 – 1.0001 | 185 | 0.196 ms | 5.12× |
| oxifit-builtin+rayon | yes | 2.8e-07 | 1.0000 | 219 | 0.257 ms | 3.89× |
| oxifit-builtin | yes | 2.8e-07 | 1.0000 | 219 | 0.264 ms | 3.80× |
| ROOT-Minuit2 (option E) | yes | 3.0e-03 | 1.0000 – 1.0002 | 143 | 0.882 ms | 1.14× |
| iminuit | yes | 0.0e+00 | 1.0000 | 165 | 1.001 ms | 1.00× |

Reference values (iminuit): A = 334.286 ± 7.24, μ = 5.18461 ± 0.0128, σ = 0.59458 ± 0.0119, c₀ = 6.69347 ± 0.0108, c₁ = -0.397488 ± 0.00426

### Binned Poisson likelihood (100 bins)

| Tool | valid | max \|Δθ\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |
|---|---|---|---|---|---|---|
| oxifit-builtin | yes | 1.8e-04 | 1.0000 | 135 | 0.271 ms | 4.61× |
| oxifit-builtin+rayon | yes | 1.8e-04 | 1.0000 | 135 | 0.271 ms | 4.60× |
| oxifit-closure | yes | 8.1e-08 | 1.0000 | 196 | 0.385 ms | 3.24× |
| oxifit-closure+rayon | yes | 8.1e-08 | 1.0000 | 196 | 0.388 ms | 3.22× |
| ROOT-Minuit2 | yes | 2.1e-10 | 1.0000 – 1.0000 | 143 | 0.528 ms | 2.37× |
| iminuit | yes | 0.0e+00 | 1.0000 | 178 | 1.248 ms | 1.00× |

Reference values (iminuit): A = 333.98 ± 7.28, μ = 5.18569 ± 0.0127, σ = 0.596299 ± 0.0119, c₀ = 6.69246 ± 0.0108, c₁ = -0.395224 ± 0.00419

### Extended unbinned likelihood, 10⁵ events

| Tool | valid | max \|Δθ\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |
|---|---|---|---|---|---|---|
| oxifit-builtin+rayon | yes | 4.9e-03 | 1.0001 – 1.0003 | 133 | 38 ms | 15.20× |
| oxifit-closure+rayon | yes | 2.1e-03 | 0.9954 – 0.9991 | 194 | 46 ms | 12.59× |
| oxifit-gradient+rayon | yes | 3.5e-03 | 1.0001 – 1.0003 | 135 | 47 ms | 12.26× |
| RooFit-Minuit2 | yes | 6.0e-03 | 0.9999 – 1.0001 | — | 102 ms (first 0.1 s) | 5.65× |
| oxifit-builtin | yes | 4.9e-03 | 1.0001 – 1.0003 | 133 | 161 ms | 3.58× |
| zfit | yes | 5.9e-03 | 1.0001 – 1.0003 | 143 | 188 ms (first 0.4 s) | 3.06× |
| oxifit-gradient | yes | 3.5e-03 | 1.0001 – 1.0003 | 135 | 223 ms | 2.58× |
| oxifit-closure | yes | 2.1e-03 | 0.9954 – 0.9991 | 194 | 295 ms | 1.95× |
| iminuit | yes | 0.0e+00 | 1.0000 | 184 | 575 ms | 1.00× |

Reference values (iminuit): Nₛ = 9817.48 ± 196, μ = 5.19483 ± 0.0114, σ = 0.597253 ± 0.0116, N_b = 90181.2 ± 345, λ = 0.396069 ± 0.00198

### Extended unbinned likelihood, 10⁶ events

| Tool | valid | max \|Δθ\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |
|---|---|---|---|---|---|---|
| oxifit-builtin+rayon | yes | 1.5e-02 | 1.0013 – 1.0074 | 85 | 174 ms | 33.20× |
| oxifit-gradient+rayon | yes | 2.2e-02 | 1.0013 – 1.0074 | 135 | 226 ms | 25.59× |
| oxifit-closure+rayon | yes | 8.3e-03 | 1.0004 – 1.0021 | 165 | 231 ms | 25.01× |
| zfit | yes | 1.7e-02 | 1.0013 – 1.0074 | 139 | 360 ms (first 0.6 s) | 16.07× |
| RooFit-Minuit2 | yes | 1.6e-02 | 1.0013 – 1.0072 | — | 1032 ms (first 1.1 s) | 5.60× |
| oxifit-builtin | yes | 1.5e-02 | 1.0013 – 1.0074 | 85 | 1116 ms | 5.18× |
| oxifit-gradient | yes | 2.2e-02 | 1.0013 – 1.0074 | 135 | 2131 ms | 2.71× |
| oxifit-closure | yes | 8.3e-03 | 1.0004 – 1.0021 | 165 | 2495 ms | 2.32× |
| iminuit | yes | 0.0e+00 | 1.0000 | 162 | 5778 ms | 1.00× |

Reference values (iminuit): Nₛ = 100613 ± 619, μ = 5.20119 ± 0.00353, σ = 0.600737 ± 0.00358, N_b = 899389 ± 1.09e+03, λ = 0.400184 ± 0.000632

All valid fits agree with iminuit to within 0.05σ on every parameter.
