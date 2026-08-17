# RoboCasa365 atomic_seen matrix

Primary held-out scope: 4 tasks.
| Config | Primary macro SR | 95% task-cluster CI | Task macro SR | Episodes |
|---|---:|---:|---:|---:|
| ckaonly | 0.775 | [0.660, 0.890] | 0.775 | 155/200 |

## Per-task success rate

| Task | ckaonly |
|---|---:|
| OpenCabinet | 0.900 |
| OpenStandMixerHead | 0.880 |
| PickPlaceDrawerToCounter | 0.620 |
| CoffeeSetupMug | 0.700 |

## Prespecified paired comparisons

| Contrast | Delta | 95% task-cluster CI | Permutation p | Holm p |
|---|---:|---:|---:|---:|

## LIBERO context

v1.4 macro Avg 89.2%; uniform W6 88.2%; v1.3 85.2%. These values are contextual and are not pooled with RoboCasa365.


## Efficiency

| Config | Mean episode wall (s) | Env construct (s) | Inference/replan (s) | Env step (s) | Peak GPU memory (MiB) | Mean GPU util |
|---|---:|---:|---:|---:|---:|---:|
| ckaonly | 29.4 | 10.1 | 0.271 | 0.047 | 21067 | 20.1% |