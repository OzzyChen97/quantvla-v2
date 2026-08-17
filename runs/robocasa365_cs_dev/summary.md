# RoboCasa365 atomic_seen expanded ablation

Primary held-out scope: 4 tasks.
Acceptance default: **ckaonly**.

| Config | Held-out macro SR | 95% task-cluster CI | All-18 macro SR | Episodes |
|---|---:|---:|---:|---:|
| ckaonly | 0.750 | [0.667, 0.917] | 0.750 | 9/12 |
| cscka_2to1 | 0.667 | [0.417, 0.917] | 0.667 | 8/12 |
| cscka_16to1 | 0.667 | [0.417, 0.917] | 0.667 | 8/12 |
| cscka_64to1 | 0.917 | [0.750, 1.000] | 0.917 | 11/12 |

## Per-task success rate

| Task | ckaonly | cscka_2to1 | cscka_16to1 | cscka_64to1 |
|---|---:|---:|---:|---:|
| OpenCabinet | 0.667 | 1.000 | 1.000 | 1.000 |
| OpenStandMixerHead | 0.667 | 0.667 | 0.667 | 0.667 |
| PickPlaceDrawerToCounter | 1.000 | 0.667 | 0.667 | 1.000 |
| CoffeeSetupMug | 0.667 | 0.333 | 0.333 | 1.000 |

## Prespecified paired comparisons

| Contrast | Delta | 95% task-cluster CI | Permutation p | Holm p |
|---|---:|---:|---:|---:|

## LIBERO context

v1.4 macro Avg 89.2%; uniform W6 88.2%; v1.3 85.2%. These values are contextual and are not pooled with RoboCasa365.
