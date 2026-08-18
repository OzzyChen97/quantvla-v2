# RoboCasa365 atomic_seen matrix

Primary held-out scope: 18 tasks.
| Config | Primary macro SR | 95% task-cluster CI | Task macro SR | Episodes |
|---|---:|---:|---:|---:|
| fp16 | 0.756 | [0.688, 0.818] | 0.756 | 680/900 |
| w4a8_atmohb | 0.496 | [0.394, 0.597] | 0.496 | 446/900 |
| cscka_final | 0.690 | [0.614, 0.759] | 0.690 | 621/900 |
| cscka_final_atmohb | 0.667 | [0.571, 0.753] | 0.667 | 600/900 |

## Per-task success rate

| Task | fp16 | w4a8_atmohb | cscka_final | cscka_final_atmohb |
|---|---:|---:|---:|---:|
| CloseBlenderLid | 0.440 | 0.220 | 0.380 | 0.220 |
| CloseFridge | 0.880 | 0.700 | 0.800 | 0.760 |
| CloseToasterOvenDoor | 0.920 | 0.780 | 0.820 | 0.800 |
| CoffeeSetupMug | 0.680 | 0.700 | 0.720 | 0.660 |
| NavigateKitchen | 0.800 | 0.280 | 0.620 | 0.420 |
| OpenCabinet | 0.940 | 0.760 | 0.920 | 0.940 |
| OpenDrawer | 0.880 | 0.780 | 0.780 | 0.800 |
| OpenStandMixerHead | 0.940 | 0.740 | 0.860 | 0.840 |
| PickPlaceCounterToCabinet | 0.680 | 0.400 | 0.680 | 0.740 |
| PickPlaceCounterToStove | 0.580 | 0.580 | 0.660 | 0.600 |
| PickPlaceDrawerToCounter | 0.700 | 0.340 | 0.700 | 0.740 |
| PickPlaceSinkToCounter | 0.740 | 0.320 | 0.720 | 0.680 |
| PickPlaceToasterToCounter | 0.860 | 0.680 | 0.840 | 0.920 |
| SlideDishwasherRack | 0.800 | 0.360 | 0.600 | 0.560 |
| TurnOffStove | 0.520 | 0.120 | 0.320 | 0.300 |
| TurnOnElectricKettle | 0.860 | 0.620 | 0.840 | 0.840 |
| TurnOnMicrowave | 0.740 | 0.240 | 0.660 | 0.660 |
| TurnOnSinkFaucet | 0.640 | 0.300 | 0.500 | 0.520 |

## Prespecified paired comparisons

| Contrast | Delta | 95% task-cluster CI | Permutation p | Holm p |
|---|---:|---:|---:|---:|
| w4a8_atmohb_vs_fp16 | -0.260 | [-0.330, -0.187] | 0.0000 | 0.0001 |
| cscka_final_vs_fp16 | -0.066 | [-0.102, -0.031] | 0.0034 | 0.0067 |
| cscka_final_vs_w4a8_atmohb | +0.194 | [+0.138, +0.252] | 0.0000 | 0.0001 |
| cscka_final_atmohb_vs_cscka_final | -0.023 | [-0.057, +0.006] | 0.2086 | 0.2086 |
| cscka_final_atmohb_vs_w4a8_atmohb | +0.171 | [+0.108, +0.234] | 0.0001 | 0.0004 |

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
| fp16 | 26.9 | 7.5 | 0.139 | 0.048 | n/a | 12769 | 11.4% |
| w4a8_atmohb | 37.5 | 7.3 | 0.268 | 0.048 | n/a | 17973 | 14.9% |
| cscka_final | 30.9 | 7.6 | 0.254 | 0.048 | n/a | 17671 | 13.3% |
| cscka_final_atmohb | 31.6 | 7.5 | 0.259 | 0.048 | n/a | 17719 | 13.2% |

Scheduling: 2 shards/config; EGL pool=None; GPU sampling scope=dedicated model-GPU device total.