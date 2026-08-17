# RoboCasa365 atomic_seen matrix

Primary held-out scope: 1 tasks.
| Config | Primary macro SR | 95% task-cluster CI | Task macro SR | Episodes |
|---|---:|---:|---:|---:|
| cscka_8to1 | 0.500 | [0.500, 0.500] | 0.500 | 1/2 |
| cscka_16to1 | 0.500 | [0.500, 0.500] | 0.500 | 1/2 |
| cscka_32to1 | 0.500 | [0.500, 0.500] | 0.500 | 1/2 |
| cscka_64to1 | 0.500 | [0.500, 0.500] | 0.500 | 1/2 |

## Per-task success rate

| Task | cscka_8to1 | cscka_16to1 | cscka_32to1 | cscka_64to1 |
|---|---:|---:|---:|---:|
| OpenStandMixerHead | 0.500 | 0.500 | 0.500 | 0.500 |

## Prespecified paired comparisons

| Contrast | Delta | 95% task-cluster CI | Permutation p | Holm p |
|---|---:|---:|---:|---:|
| cscka_64to1_vs_cscka_32to1 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 |
| cscka_64to1_vs_cscka_16to1 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 |
| cscka_64to1_vs_cscka_8to1 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 |

## LIBERO context

v1.4 macro Avg 89.2%; uniform W6 88.2%; v1.3 85.2%. These values are contextual and are not pooled with RoboCasa365.


## Efficiency

| Config | Mean episode wall (s) | Env construct (s) | Inference/replan (s) | Env step (s) | Peak GPU memory (MiB) | Mean GPU util |
|---|---:|---:|---:|---:|---:|---:|
| cscka_8to1 | 25.3 | 6.9 | 0.304 | 0.051 | 14776 | 1.8% |
| cscka_16to1 | 24.8 | 5.7 | 0.285 | 0.050 | 14776 | 7.7% |
| cscka_32to1 | 25.9 | 5.8 | 0.295 | 0.052 | 14546 | 4.4% |
| cscka_64to1 | 25.6 | 6.4 | 0.290 | 0.052 | 14273 | 7.0% |
