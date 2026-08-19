# GDSQ-VLA CVPR 2026 draft

This directory contains an anonymous CVPR 2026 review draft using the official
`CVPR2026-v1(latex)` author kit at commit
`12909ae437f6dbc7435069cfdb4ca44c18e6a02f`.

Build the paper with:

```bash
make pdf
```

The build intentionally uses a paper-local TeX tree for three packages missing
from the host's TeX Live 2017 installation. The official `cvpr.sty` is unchanged.

`figures/gdsq_pipeline.png` is generated only when a standard GPT Image endpoint
and a locally exported, non-disclosed `OPENAI_API_KEY` are available. Until then,
the source renders a clearly marked, compile-safe pipeline placeholder. Replace
all red `TBD` results only from completed frozen summaries.
