# RoboCasa365 atomic_seen expanded ablation

Primary held-out scope: 14 tasks.
Acceptance default: **ckaonly**.

| Config | Held-out macro SR | 95% task-cluster CI | All-18 macro SR | Episodes |
|---|---:|---:|---:|---:|
| fp16 | 0.714 | [0.586, 0.843] | 0.733 | 66/90 |
| w4a8_atmohb | 0.529 | [0.357, 0.700] | 0.522 | 47/90 |
| csonly | 0.586 | [0.414, 0.743] | 0.611 | 55/90 |
| ckaonly | 0.571 | [0.400, 0.743] | 0.600 | 54/90 |
| cscka_adjusted | 0.671 | [0.543, 0.800] | 0.678 | 61/90 |
| cka_atmohb | 0.600 | [0.500, 0.714] | 0.633 | 57/90 |

## Per-task success rate

| Task | fp16 | w4a8_atmohb | csonly | ckaonly | cscka_adjusted | cka_atmohb |
|---|---:|---:|---:|---:|---:|---:|
| CloseBlenderLid | 0.200 | 0.000 | 0.200 | 0.000 | 0.200 | 0.400 |
| CloseFridge | 1.000 | 0.800 | 1.000 | 1.000 | 1.000 | 0.600 |
| CloseToasterOvenDoor | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 1.000 |
| CoffeeSetupMug | 0.800 | 0.600 | 0.600 | 0.600 | 0.800 | 0.600 |
| NavigateKitchen | 1.000 | 0.600 | 0.600 | 0.600 | 0.800 | 0.400 |
| OpenCabinet | 0.600 | 0.400 | 0.600 | 0.600 | 0.800 | 0.800 |
| OpenDrawer | 1.000 | 0.800 | 0.400 | 0.600 | 0.400 | 0.600 |
| OpenStandMixerHead | 1.000 | 0.600 | 0.800 | 0.800 | 0.600 | 0.800 |
| PickPlaceCounterToCabinet | 0.600 | 0.400 | 0.400 | 0.400 | 0.400 | 0.600 |
| PickPlaceCounterToStove | 0.600 | 0.600 | 0.800 | 0.800 | 0.800 | 0.600 |
| PickPlaceDrawerToCounter | 0.800 | 0.400 | 0.800 | 0.800 | 0.600 | 0.800 |
| PickPlaceSinkToCounter | 0.800 | 0.200 | 0.000 | 0.400 | 0.800 | 0.800 |
| PickPlaceToasterToCounter | 0.600 | 1.000 | 0.800 | 1.000 | 1.000 | 1.000 |
| SlideDishwasherRack | 0.800 | 0.600 | 1.000 | 0.600 | 1.000 | 0.600 |
| TurnOffStove | 0.600 | 0.000 | 0.400 | 0.000 | 0.600 | 0.400 |
| TurnOnElectricKettle | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.600 |
| TurnOnMicrowave | 0.600 | 0.400 | 0.200 | 0.200 | 0.400 | 0.400 |
| TurnOnSinkFaucet | 0.400 | 0.200 | 0.600 | 0.600 | 0.400 | 0.400 |

## Prespecified paired comparisons

| Contrast | Delta | 95% task-cluster CI | Permutation p | Holm p |
|---|---:|---:|---:|---:|
| cscka_adjusted_vs_ckaonly | +0.100 | [-0.014, +0.229] | 0.2227 | 0.8906 |
| cscka_adjusted_vs_csonly | +0.086 | [-0.029, +0.214] | 0.3438 | 1.0000 |
| cka_atmohb_vs_ckaonly | +0.029 | [-0.100, +0.157] | 0.8359 | 1.0000 |
| cka_atmohb_vs_w4a8_atmohb | +0.071 | [-0.043, +0.200] | 0.4258 | 1.0000 |

## LIBERO context

v1.4 macro Avg 89.2%; uniform W6 88.2%; v1.3 85.2%. These values are contextual and are not pooled with RoboCasa365.
