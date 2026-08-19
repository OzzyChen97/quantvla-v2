# RoboCasa365 composite_unseen matrix

Primary held-out scope: 16 tasks.
| Config | Primary macro SR | 95% task-cluster CI | Task macro SR | Episodes |
|---|---:|---:|---:|---:|
| fp16 | 0.453 | [0.322, 0.589] | 0.453 | 362/800 |
| w4a8_atmohb | 0.198 | [0.116, 0.285] | 0.198 | 158/800 |
| cscka_final | 0.404 | [0.269, 0.545] | 0.404 | 323/800 |
| cscka_final_atmohb | 0.403 | [0.284, 0.526] | 0.403 | 322/800 |

## Per-task success rate

| Task | fp16 | w4a8_atmohb | cscka_final | cscka_final_atmohb |
|---|---:|---:|---:|---:|
| ArrangeBreadBasket | 0.780 | 0.340 | 0.680 | 0.740 |
| ArrangeTea | 0.640 | 0.380 | 0.600 | 0.520 |
| BreadSelection | 0.800 | 0.500 | 0.740 | 0.720 |
| CategorizeCondiments | 0.220 | 0.080 | 0.180 | 0.260 |
| CuttingToolSelection | 0.460 | 0.220 | 0.520 | 0.500 |
| GarnishPancake | 0.460 | 0.220 | 0.500 | 0.460 |
| GatherTableware | 0.180 | 0.060 | 0.140 | 0.140 |
| HeatKebabSandwich | 0.260 | 0.020 | 0.080 | 0.160 |
| MakeIceLemonade | 0.300 | 0.020 | 0.280 | 0.240 |
| PanTransfer | 0.240 | 0.040 | 0.100 | 0.140 |
| PortionHotDogs | 0.200 | 0.100 | 0.240 | 0.260 |
| RecycleBottlesByType | 0.940 | 0.520 | 0.900 | 0.840 |
| SeparateFreezerRack | 0.020 | 0.000 | 0.000 | 0.040 |
| WaffleReheat | 0.880 | 0.420 | 0.880 | 0.800 |
| WashFruitColander | 0.380 | 0.060 | 0.280 | 0.340 |
| WeighIngredients | 0.480 | 0.180 | 0.340 | 0.280 |

## Prespecified paired comparisons

| Contrast | Delta | 95% task-cluster CI | Permutation p | Holm p |
|---|---:|---:|---:|---:|
| w4a8_atmohb_vs_fp16 | -0.255 | [-0.312, -0.196] | 0.0000 | 0.0002 |
| cscka_final_vs_fp16 | -0.049 | [-0.081, -0.017] | 0.0158 | 0.0316 |
| cscka_final_vs_w4a8_atmohb | +0.206 | [+0.145, +0.268] | 0.0001 | 0.0002 |
| cscka_final_atmohb_vs_cscka_final | -0.001 | [-0.028, +0.026] | 1.0000 | 1.0000 |
| cscka_final_atmohb_vs_w4a8_atmohb | +0.205 | [+0.155, +0.258] | 0.0000 | 0.0002 |

## LIBERO context

v1.4 macro Avg 89.2%; uniform W6 88.2%; v1.3 85.2%. These values are contextual and are not pooled with RoboCasa365.


## Paper-style LLM+DiT component memory

Theoretical tightly-packed deployment storage (QuantVLA Tables 1/2 scope); this is not live CUDA memory.

| Config | Quantized Linear layers | Component memory (GiB) | Savings vs FP16 | Compression |
|---|---:|---:|---:|---:|
| fp16 | 0/180 | 1.993 | 0.0% | 1.00× |
| w4a8_atmohb | 116/180 | 0.898 | 55.0% | 2.22× |
| cscka_final | 100/180 | 1.001 | 49.8% | 1.99× |
| cscka_final_atmohb | 100/180 | 1.001 | 49.8% | 1.99× |

## Efficiency

| Config | Mean episode wall (s) | Env construct (s) | Inference/replan (s) | Env step (s) | Peak server memory (MiB) | Peak device memory (MiB) | Mean GPU util |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 219.1 | 11.2 | 0.821 | 0.066 | 7902 | 24889 | 23.5% |
| w4a8_atmohb | 498.2 | 12.9 | 2.102 | 0.080 | 13418 | 41619 | 19.4% |
| cscka_final | 438.6 | 13.4 | 2.174 | 0.087 | 13004 | 37449 | 54.8% |
| cscka_final_atmohb | 442.7 | 14.1 | 2.200 | 0.087 | 13004 | 43021 | 48.7% |

Scheduling: 8 shards/config; EGL pool=[1, 2, 3, 4, 5, 6, 7]; GPU sampling scope=model-GPU device totals; replicas and EGL clients use shared GPUs 1-7.