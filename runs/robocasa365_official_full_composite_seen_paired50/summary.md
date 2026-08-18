# RoboCasa365 composite_seen matrix

Primary held-out scope: 16 tasks.
| Config | Primary macro SR | 95% task-cluster CI | Task macro SR | Episodes |
|---|---:|---:|---:|---:|
| fp16 | 0.419 | [0.302, 0.534] | 0.419 | 335/800 |
| w4a8_atmohb | 0.196 | [0.108, 0.296] | 0.196 | 157/800 |
| cscka_final | 0.406 | [0.289, 0.525] | 0.406 | 325/800 |
| cscka_final_atmohb | 0.391 | [0.285, 0.498] | 0.391 | 313/800 |

## Per-task success rate

| Task | fp16 | w4a8_atmohb | cscka_final | cscka_final_atmohb |
|---|---:|---:|---:|---:|
| DeliverStraw | 0.140 | 0.000 | 0.060 | 0.160 |
| GetToastedBread | 0.000 | 0.000 | 0.000 | 0.000 |
| KettleBoiling | 0.700 | 0.260 | 0.520 | 0.520 |
| LoadDishwasher | 0.600 | 0.580 | 0.660 | 0.560 |
| PackIdenticalLunches | 0.140 | 0.060 | 0.100 | 0.120 |
| PreSoakPan | 0.600 | 0.280 | 0.740 | 0.660 |
| PrepareCoffee | 0.180 | 0.020 | 0.160 | 0.240 |
| RinseSinkBasin | 0.480 | 0.200 | 0.560 | 0.460 |
| ScrubCuttingBoard | 0.700 | 0.280 | 0.660 | 0.540 |
| SearingMeat | 0.260 | 0.060 | 0.180 | 0.160 |
| SetUpCuttingStation | 0.520 | 0.260 | 0.380 | 0.460 |
| StackBowlsCabinet | 0.840 | 0.660 | 0.820 | 0.760 |
| SteamInMicrowave | 0.420 | 0.060 | 0.320 | 0.320 |
| StirVegetables | 0.320 | 0.180 | 0.360 | 0.420 |
| StoreLeftoversInBowl | 0.240 | 0.000 | 0.360 | 0.200 |
| WashLettuce | 0.560 | 0.240 | 0.620 | 0.680 |

## Prespecified paired comparisons

| Contrast | Delta | 95% task-cluster CI | Permutation p | Holm p |
|---|---:|---:|---:|---:|
| w4a8_atmohb_vs_fp16 | -0.223 | [-0.284, -0.161] | 0.0001 | 0.0003 |
| cscka_final_vs_fp16 | -0.012 | [-0.057, +0.030] | 0.6268 | 0.9985 |
| cscka_final_vs_w4a8_atmohb | +0.210 | [+0.143, +0.280] | 0.0001 | 0.0003 |
| cscka_final_atmohb_vs_cscka_final | -0.015 | [-0.054, +0.023] | 0.4993 | 0.9985 |
| cscka_final_atmohb_vs_w4a8_atmohb | +0.195 | [+0.136, +0.255] | 0.0001 | 0.0004 |

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
| fp16 | 382.4 | 13.8 | 1.850 | 0.088 | 7654 | 45392 | 17.9% |
| w4a8_atmohb | 459.5 | 11.0 | 2.226 | 0.074 | 13170 | 45218 | 16.8% |
| cscka_final | 383.9 | 14.1 | 1.829 | 0.087 | 12998 | 45269 | 15.8% |
| cscka_final_atmohb | 394.2 | 14.9 | 1.901 | 0.087 | 12756 | 45414 | 18.2% |

Scheduling: 4 shards/config; EGL pool=None; GPU sampling scope=dedicated model-GPU device total.