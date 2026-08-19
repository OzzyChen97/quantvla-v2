# Official RoboCasa365 50-task paired evaluation

50 tasks × 50 scenarios; 2500 episodes/config.

| Config | 50-task macro SR | 95% task-cluster CI | Episode SR |
|---|---:|---:|---:|
| fp16 | 0.551 | [0.476, 0.625] | 1377/2500 (0.551) |
| w4a8_atmohb | 0.304 | [0.239, 0.374] | 761/2500 (0.304) |
| cscka_final | 0.508 | [0.432, 0.581] | 1269/2500 (0.508) |
| cscka_final_atmohb | 0.494 | [0.421, 0.565] | 1235/2500 (0.494) |

## Task-set macro SR

| Task set | fp16 | w4a8_atmohb | cscka_final | cscka_final_atmohb |
|---|---:|---:|---:|---:|
| atomic_seen | 0.756 | 0.496 | 0.690 | 0.667 |
| composite_seen | 0.419 | 0.196 | 0.406 | 0.391 |
| composite_unseen | 0.453 | 0.198 | 0.404 | 0.403 |

## Prespecified paired comparisons

| Contrast | Delta | 95% task-cluster CI | Permutation p | Holm p |
|---|---:|---:|---:|---:|
| w4a8_atmohb_vs_fp16 | -0.246 | [-0.284, -0.208] | 0.0000 | 0.0000 |
| cscka_final_vs_fp16 | -0.043 | [-0.066, -0.021] | 0.0006 | 0.0012 |
| cscka_final_vs_w4a8_atmohb | +0.203 | [+0.169, +0.240] | 0.0000 | 0.0000 |
| cscka_final_atmohb_vs_cscka_final | -0.014 | [-0.033, +0.005] | 0.1804 | 0.1804 |
| cscka_final_atmohb_vs_w4a8_atmohb | +0.190 | [+0.156, +0.223] | 0.0000 | 0.0000 |

## Per-task SR

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

## Efficiency

| Config | Episode wall (s) | Inference/replan (s) | Env step (s) | Peak server memory (MiB) | Peak device memory (MiB) | Mean util | Mean power (W) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 202.2 | 1.218 | 0.074 | 7902 | 45392 | 18.6% | 301.8 |
| w4a8_atmohb | 320.0 | 1.953 | 0.074 | 13418 | 45218 | 18.0% | 109.9 |
| cscka_final | 274.3 | 1.825 | 0.083 | 13004 | 45269 | 35.1% | 150.4 |
| cscka_final_atmohb | 279.2 | 1.864 | 0.083 | 13004 | 45414 | 33.1% | 233.8 |

Aggregate efficiency above mixes task horizons and execution concurrency; use the task-set table below for scheduling-aware comparisons.

## Efficiency by task set

| Task set | Config | Shards | Episode wall (s) | Inference/replan (s) | Env step (s) | Peak server memory (MiB) | Peak device memory (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| atomic_seen | fp16 | 2 | 26.9 | 0.139 | 0.048 | nan | 12769 |
| atomic_seen | w4a8_atmohb | 2 | 37.5 | 0.268 | 0.048 | nan | 17973 |
| atomic_seen | cscka_final | 2 | 30.9 | 0.254 | 0.048 | nan | 17671 |
| atomic_seen | cscka_final_atmohb | 2 | 31.6 | 0.259 | 0.048 | nan | 17719 |
| composite_seen | fp16 | 4 | 382.4 | 1.850 | 0.088 | 7654 | 45392 |
| composite_seen | w4a8_atmohb | 4 | 459.5 | 2.226 | 0.074 | 13170 | 45218 |
| composite_seen | cscka_final | 4 | 383.9 | 1.829 | 0.087 | 12998 | 45269 |
| composite_seen | cscka_final_atmohb | 4 | 394.2 | 1.901 | 0.087 | 12756 | 45414 |
| composite_unseen | fp16 | 8 | 219.1 | 0.821 | 0.066 | 7902 | 24889 |
| composite_unseen | w4a8_atmohb | 8 | 498.2 | 2.102 | 0.080 | 13418 | 41619 |
| composite_unseen | cscka_final | 8 | 438.6 | 2.174 | 0.087 | 13004 | 37449 |
| composite_unseen | cscka_final_atmohb | 8 | 442.7 | 2.200 | 0.087 | 13004 | 43021 |

Scheduling/GPU sampling notes:

- atomic_seen: 2 shards/config; EGL pool=None; GPU scope=dedicated model-GPU device total.
- composite_seen: 4 shards/config; EGL pool=None; GPU scope=dedicated model-GPU device total.
- composite_unseen: 8 shards/config; EGL pool=[1, 2, 3, 4, 5, 6, 7]; GPU scope=model-GPU device totals; replicas and EGL clients use shared GPUs 1-7.

## Paper-style LLM+DiT component memory

Theoretical tightly-packed deployment storage (QuantVLA Tables 1/2 scope); this is distinct from live CUDA memory.

| Config | atomic_seen (GiB) | composite_seen (GiB) | composite_unseen (GiB) | 50-task weighted mean (GiB) | Savings vs FP16 |
|---|---:|---:|---:|---:|---:|
| fp16 | 1.993 | 1.993 | 1.993 | 1.993 | 0.0% |
| w4a8_atmohb | 0.898 | 0.898 | 0.898 | 0.898 | 55.0% |
| cscka_final | 1.001 | 1.001 | 1.001 | 1.001 | 49.8% |
| cscka_final_atmohb | 1.001 | 1.001 | 1.001 | 1.001 | 49.8% |

LIBERO context: v1.4 macro Avg 89.2%, uniform W6 88.2%, v1.3 85.2%. These values are not pooled with RoboCasa365.
