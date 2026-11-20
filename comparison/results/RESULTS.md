### Binned χ² (100 bins)

| Tool | valid | max \|Δθ\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |
|---|---|---|---|---|---|---|
| oxifit-builtin | yes | 5.6e-06 | 1.0000 | 39 | 0.071 ms | 14.09× |
| oxifit-builtin+rayon | yes | 5.6e-06 | 1.0000 | 39 | 0.072 ms | 13.86× |
| scipy | yes | 3.0e-03 | 0.9990 – 1.0368 | 31 | 0.149 ms | 6.74× |
| oxifit-closure | yes | 2.8e-07 | 1.0000 – 1.0001 | 185 | 0.166 ms | 6.05× |
| oxifit-closure+rayon | yes | 2.8e-07 | 1.0000 – 1.0001 | 185 | 0.180 ms | 5.56× |
| ROOT-Minuit2 | yes | 3.0e-03 | 0.9702 – 0.9994 | 112 | 0.194 ms | 5.18× |
| ROOT-Minuit2 (option E) | yes | 3.0e-03 | 1.0000 – 1.0002 | 143 | 0.868 ms | 1.16× |
| iminuit | yes | 0.0e+00 | 1.0000 | 165 | 1.003 ms | 1.00× |

Reference values (iminuit): A = 334.286 ± 7.24, μ = 5.18461 ± 0.0128, σ = 0.59458 ± 0.0119, c₀ = 6.69347 ± 0.0108, c₁ = -0.397488 ± 0.00426

### Binned Poisson likelihood (100 bins)

| Tool | valid | max \|Δθ\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |
|---|---|---|---|---|---|---|
| oxifit-builtin | yes | 8.0e-07 | 1.0000 | 41 | 0.078 ms | 16.34× |
| oxifit-builtin+rayon | yes | 8.0e-07 | 1.0000 | 41 | 0.078 ms | 16.27× |
| oxifit-closure+rayon | yes | 8.1e-08 | 1.0000 | 196 | 0.394 ms | 3.22× |
| oxifit-closure | yes | 8.1e-08 | 1.0000 | 196 | 0.394 ms | 3.22× |
| ROOT-Minuit2 | yes | 2.1e-10 | 1.0000 – 1.0000 | 143 | 0.570 ms | 2.23× |
| iminuit | yes | 0.0e+00 | 1.0000 | 178 | 1.269 ms | 1.00× |

Reference values (iminuit): A = 333.98 ± 7.28, μ = 5.18569 ± 0.0127, σ = 0.596299 ± 0.0119, c₀ = 6.69246 ± 0.0108, c₁ = -0.395224 ± 0.00419

### Extended unbinned likelihood, 10⁵ events

| Tool | valid | max \|Δθ\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |
|---|---|---|---|---|---|---|
| oxifit-builtin+rayon | yes | 5.9e-03 | 1.0001 – 1.0003 | 47 | 14 ms | 40.50× |
| oxifit-gradient+rayon | yes | 5.6e-03 | 1.0001 – 1.0003 | 41 | 32 ms | 18.36× |
| oxifit-closure+rayon | yes | 2.1e-03 | 0.9954 – 0.9991 | 194 | 41 ms | 14.30× |
| oxifit-builtin | yes | 5.9e-03 | 1.0001 – 1.0003 | 47 | 60 ms | 9.69× |
| oxifit-gradient | yes | 5.6e-03 | 1.0001 – 1.0003 | 41 | 87 ms | 6.69× |
| RooFit-Minuit2 | yes | 6.0e-03 | 0.9999 – 1.0001 | — | 105 ms (first 0.1 s) | 5.51× |
| zfit | yes | 5.9e-03 | 1.0001 – 1.0003 | 143 | 189 ms (first 0.4 s) | 3.07× |
| oxifit-closure | yes | 2.1e-03 | 0.9954 – 0.9991 | 194 | 295 ms | 1.97× |
| iminuit | yes | 0.0e+00 | 1.0000 | 184 | 580 ms | 1.00× |

Reference values (iminuit): Nₛ = 9817.48 ± 196, μ = 5.19483 ± 0.0114, σ = 0.597253 ± 0.0116, N_b = 90181.2 ± 345, λ = 0.396069 ± 0.00198

### Extended unbinned likelihood, 10⁶ events

| Tool | valid | max \|Δθ\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |
|---|---|---|---|---|---|---|
| oxifit-builtin+rayon | yes | 1.8e-02 | 1.0013 – 1.0074 | 47 | 107 ms | 55.13× |
| oxifit-gradient+rayon | yes | 1.8e-02 | 1.0013 – 1.0074 | 41 | 117 ms | 50.30× |
| oxifit-closure+rayon | yes | 8.3e-03 | 1.0004 – 1.0021 | 165 | 249 ms | 23.57× |
| zfit | yes | 1.7e-02 | 1.0013 – 1.0074 | 139 | 365 ms (first 0.6 s) | 16.07× |
| oxifit-builtin | yes | 1.8e-02 | 1.0013 – 1.0074 | 47 | 604 ms | 9.73× |
| oxifit-gradient | yes | 1.8e-02 | 1.0013 – 1.0074 | 41 | 741 ms | 7.92× |
| RooFit-Minuit2 | yes | 1.6e-02 | 1.0013 – 1.0072 | — | 1064 ms (first 1.1 s) | 5.52× |
| oxifit-closure | yes | 8.3e-03 | 1.0004 – 1.0021 | 165 | 2495 ms | 2.35× |
| iminuit | yes | 0.0e+00 | 1.0000 | 162 | 5873 ms | 1.00× |

Reference values (iminuit): Nₛ = 100613 ± 619, μ = 5.20119 ± 0.00353, σ = 0.600737 ± 0.00358, N_b = 899389 ± 1.09e+03, λ = 0.400184 ± 0.000632

All valid fits agree with iminuit to within 0.05σ on every parameter.
