### Binned χ² (100 bins)

| Tool | valid | max \|Δθ\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |
|---|---|---|---|---|---|---|
| scipy | yes | 3.0e-03 | 0.9990 – 1.0368 | 31 | 0.151 ms | 6.13× |
| oxifit-closure+rayon | yes | 2.8e-07 | 1.0000 – 1.0001 | 185 | 0.158 ms | 5.85× |
| oxifit-closure | yes | 2.8e-07 | 1.0000 – 1.0001 | 185 | 0.161 ms | 5.76× |
| ROOT-Minuit2 | yes | 3.0e-03 | 0.9702 – 0.9994 | 112 | 0.179 ms | 5.16× |
| oxifit-builtin+rayon | yes | 2.8e-07 | 1.0000 | 219 | 0.228 ms | 4.06× |
| oxifit-builtin | yes | 2.8e-07 | 1.0000 | 219 | 0.230 ms | 4.04× |
| ROOT-Minuit2 (option E) | yes | 3.0e-03 | 1.0000 – 1.0002 | 143 | 0.835 ms | 1.11× |
| iminuit | yes | 0.0e+00 | 1.0000 | 165 | 0.926 ms | 1.00× |

Reference values (iminuit): A = 334.286 ± 7.24, μ = 5.18461 ± 0.0128, σ = 0.59458 ± 0.0119, c₀ = 6.69347 ± 0.0108, c₁ = -0.397488 ± 0.00426

### Binned Poisson likelihood (100 bins)

| Tool | valid | max \|Δθ\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |
|---|---|---|---|---|---|---|
| oxifit-builtin | yes | 1.8e-04 | 1.0000 | 135 | 0.185 ms | 6.21× |
| oxifit-builtin+rayon | yes | 1.8e-04 | 1.0000 | 135 | 0.199 ms | 5.79× |
| oxifit-closure | yes | 8.1e-08 | 1.0000 | 196 | 0.326 ms | 3.53× |
| oxifit-closure+rayon | yes | 8.1e-08 | 1.0000 | 196 | 0.327 ms | 3.51× |
| ROOT-Minuit2 | yes | 2.1e-10 | 1.0000 – 1.0000 | 143 | 0.385 ms | 2.98× |
| iminuit | yes | 0.0e+00 | 1.0000 | 178 | 1.150 ms | 1.00× |

Reference values (iminuit): A = 333.98 ± 7.28, μ = 5.18569 ± 0.0127, σ = 0.596299 ± 0.0119, c₀ = 6.69246 ± 0.0108, c₁ = -0.395224 ± 0.00419

### Extended unbinned likelihood, 10⁵ events

| Tool | valid | max \|Δθ\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |
|---|---|---|---|---|---|---|
| oxifit-gradient+rayon | yes | 3.5e-03 | 1.0001 – 1.0003 | 135 | 44 ms | 12.01× |
| oxifit-closure+rayon | yes | 2.1e-03 | 0.9954 – 0.9991 | 194 | 45 ms | 11.74× |
| RooFit-Minuit2 | yes | 6.0e-03 | 0.9999 – 1.0001 | — | 96 ms (first 0.1 s) | 5.55× |
| zfit | yes | 5.9e-03 | 1.0001 – 1.0003 | 143 | 190 ms (first 0.4 s) | 2.82× |
| oxifit-gradient | yes | 3.5e-03 | 1.0001 – 1.0003 | 135 | 207 ms | 2.58× |
| oxifit-closure | yes | 2.1e-03 | 0.9954 – 0.9991 | 194 | 276 ms | 1.93× |
| iminuit | yes | 0.0e+00 | 1.0000 | 184 | 534 ms | 1.00× |

Reference values (iminuit): Nₛ = 9817.48 ± 196, μ = 5.19483 ± 0.0114, σ = 0.597253 ± 0.0116, N_b = 90181.2 ± 345, λ = 0.396069 ± 0.00198

### Extended unbinned likelihood, 10⁶ events

| Tool | valid | max \|Δθ\|/σ vs iminuit | errors / iminuit's | calls | time per fit | vs iminuit |
|---|---|---|---|---|---|---|
| oxifit-gradient+rayon | yes | 2.2e-02 | 1.0013 – 1.0074 | 135 | 218 ms | 25.01× |
| oxifit-closure+rayon | yes | 8.3e-03 | 1.0004 – 1.0021 | 165 | 234 ms | 23.32× |
| zfit | yes | 1.7e-02 | 1.0013 – 1.0074 | 139 | 370 ms (first 0.5 s) | 14.73× |
| RooFit-Minuit2 | yes | 1.6e-02 | 1.0013 – 1.0072 | — | 988 ms (first 1.1 s) | 5.52× |
| oxifit-gradient | yes | 2.2e-02 | 1.0013 – 1.0074 | 135 | 1993 ms | 2.74× |
| oxifit-closure | yes | 8.3e-03 | 1.0004 – 1.0021 | 165 | 2350 ms | 2.32× |
| iminuit | yes | 0.0e+00 | 1.0000 | 162 | 5451 ms | 1.00× |

Reference values (iminuit): Nₛ = 100613 ± 619, μ = 5.20119 ± 0.00353, σ = 0.600737 ± 0.00358, N_b = 899389 ± 1.09e+03, λ = 0.400184 ± 0.000632

All valid fits agree with iminuit to within 0.05σ on every parameter.
