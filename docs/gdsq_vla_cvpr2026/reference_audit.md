# Reference audit

Audit date: 2026-08-18. This file is internal and is not included in the submission PDF.

## Acceptance policy

- A citation is retained only when its title and authors match an arXiv record or an official proceedings/publisher record.
- Published venue, year, pages, and DOI fields follow the proceedings/publisher record when available; arXiv identifiers are retained as a second locator.
- arXiv revision year is not substituted for the original publication year. This matters for records such as AWQ, whose current arXiv BibTeX export reflects a later revision.
- `and others` is used only to shorten verified long author lists; it does not stand in for an unknown author list.
- The Semantic Scholar public Graph API returned HTTP 429 during this audit, so no unverified field was imported from it. Metadata instead comes from arXiv plus Crossref, DBLP/OpenReview, CVF, PMLR, or the original paper source.

## Primary paper sources

- QuantVLA: arXiv `2602.20309v4`; source archive SHA256 `a719a574b3ed8a58533c75f2273288a479b900d385db0a6aef47dfca25893228`.
- RoboCasa365: arXiv `2603.04356v1`; source archive SHA256 `f10bbc07c60b72d81332b5d1c796690b87613ab82304565e30daccc600b8e568`.
- The QuantVLA source was used to identify the closest technical literature, but every imported record was checked against its own canonical metadata source.

## Canonical arXiv checks

- VLA and robot policies: RT-1 `2212.06817`, PaLM-E `2303.03378`, RT-2 `2307.15818`, Octo `2405.12213`, OpenVLA `2406.09246`, RDT-1B `2410.07864`, $\pi_0$ `2410.24164`, $\pi_{0.5}$ `2504.16054`, GR00T N1 `2503.14734`, SmolVLA `2506.01844`, EfficientVLA `2506.10100`, VLA-Cache `2502.02175`, and MoLe-VLA `2503.20384`.
- Transformer/VLM PTQ: SmoothQuant `2211.10438`, GPTQ `2210.17323`, AWQ `2306.00978`, OmniQuant `2308.13137`, BRECQ `2102.05426`, QuaRot `2404.00456`, DuQuant `2406.01721`, SpinQuant `2405.16406`, FlatQuant `2410.09426`, OstQuant `2501.13987`, Q-VLM `2410.08119`, and MBQ `2412.19509`.
- Diffusion PTQ: PTQ4DM `2211.15736`, Q-Diffusion `2302.04304`, PTQ4DiT `2405.16005`, Q-DiT `2406.17343`, ViDiT-Q `2406.02540`, MixDQ `2405.17873`, and SVDQuant `2411.05007`.
- Similarity and allocation: SVCCA `1706.05806`, PWCCA `1806.05759`, CKA `1905.00414`, HAWQ `1905.03696`, HAWQ-V2 `1911.03852`, HAQ `1811.08886`, and CS-Aligner `2502.17028`.

## Publisher and proceedings cross-checks

- Diffusion Policy: RSS 2023, DOI `10.15607/RSS.2023.XIX.026`.
- PTQ4DM: CVPR 2023, pages 1972--1981, DOI `10.1109/CVPR52729.2023.00196`.
- Q-Diffusion: ICCV 2023, pages 17489--17499, DOI `10.1109/ICCV51070.2023.01608`.
- PTQ4DiT: NeurIPS 2024, pages 62732--62755, DOI `10.52202/079017-2006`.
- MixDQ: ECCV 2024, pages 285--302, DOI `10.1007/978-3-031-72630-9_17`.
- Q-DiT: CVPR 2025, pages 28306--28315, DOI `10.1109/CVPR52734.2025.02636`.
- MBQ: CVPR 2025, pages 4167--4177, DOI `10.1109/CVPR52734.2025.00394`.
- Accurate PTQ with Small Calibration Sets: PMLR volume 139, pages 4466--4475.
- DBLP/OpenReview records confirm BRECQ (ICLR 2021), SpinQuant (ICLR 2025), and FlatQuant (ICML 2025).

## Unified RoboCasa365 result policy

`tables/main_results.tex` is the unified official-task comparison. It groups Atomic (All 18), Composite-Seen, and Composite-Unseen columns, then groups rows by GR00T N1.5 and the planned $\pi_{0.5}$ evaluation. `tables/primary14_results.tex` separately reports the internal held-out split after the CKA:CS ratio is frozen on four disjoint development tasks. All three frozen GR00T N1.5 task-group results are populated. Incomplete $\pi_{0.5}$ cells remain explicit `TBD`; partial logs and externally reported values from different protocols are not inserted.

## Rejected or corrected records

- Removed `Improving Language Model Distillation Through Hidden State Matching`: the exact record could not be corroborated in arXiv, Crossref, OpenAlex, or DBLP during the audit.
- Corrected GPTQ from an arXiv-only 2022 record to ICLR 2023 while retaining arXiv `2210.17323`.
- Corrected ViDiT-Q to ICLR 2025 and Q-DiT to the complete eight-author CVPR 2025 record.
- Corrected GR00T N1 to retain the NVIDIA group author and added canonical arXiv identifiers to RoboCasa365 and QuantVLA.

## Build-level reference checks

- `main.bib` contains 43 entries.
- The compiled bibliography contains 43 cited entries; there are no uncited placeholders.
- BibTeX reports zero warnings, and the final LaTeX log has no undefined citation or cross-reference warning.
