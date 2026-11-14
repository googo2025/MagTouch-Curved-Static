# MagTouch Curved Static

Reproducibility companion for **Single touch localization and sparse magnetic field reconstruction on a cylinder with a hemispherical cap**.

- [English manuscript](paper/paper1_en.pdf) · [中文论文](paper/paper1_zh.pdf)
- [Interactive 3D demo](https://googo2025.github.io/MagTouch-Curved-Static/) · [Demo source](docs/)
- [Dataset and protocol](data/) · [Frozen models](models/) · [Evaluation results](results/) · [Forward-simulation reference](forward/)
- [Versioned downloads](https://github.com/googo2025/MagTouch-Curved-Static/releases)

## Scientific scope

The experiments are static single-contact simulations at a fixed 10 mN normal load with Gaussian width σ = 12 mm. The input uses 51 signed channels (17 tri-axial Hall sensors) and the output is a surface position in millimeters. Independent positions support a numerical comparison under these conditions. Physical prototype accuracy, force sensing and dynamic or multi-contact performance are outside this study.

| Input on the same 30 independent sites | Two-hidden-layer mean | RMSE | P95 |
|---|---:|---:|---:|
| Direct field | 0.375 mm | 0.463 mm | 0.929 mm |
| 121-anchor engineering interpolation | 0.963 mm | 1.096 mm | 1.817 mm |
| 147 outside-region anchors | 5.474 mm | 8.983 mm | 19.925 mm |

Strict region exclusion on the original engineering grid gave 0/30 supported matches. Boundary-enriched sampling and the local thin-plate spline reduced P95 in their respective independent comparisons. Each cohort retains its own denominator.

## Reproduce the tables

```sh
python -m venv .venv
# Activate the virtual environment using your shell.
python -m pip install -r requirements.txt
python src/reproduce.py --output reproduced.json
```

The script reads the frozen signed fields and six published network checkpoints, recomputes all 30-site crossed comparisons, checks the 17 historical sites, and recomputes the four independent 12/12/14/10-site spatial comparisons. Each matched 3D error is checked against its frozen result at absolute tolerance 1e-8 mm. The script uses the shipped training-site order for template tie handling. The original training implementation is included in `src/paper1_static_fit.py` and the manual entry point in `src/train.py`. It uses the archived 820/17 split, 12,000 full-batch Adam updates and three fixed seeds. The training entry point writes to a user-specified fresh output directory and leaves the published weights intact.

`data/fields.npz` is a NumPy archive containing field matrices (`*_fields`), reference locations (`*_xyz_mm`) and, where applicable, site identifiers and regions. Each field row is Hall-major with three successive local components per sensor, stored as double-precision µT/T. `grid_keys` and `strict_allowed` define the engineering grid and strict allowed anchors. The 147 outside-region anchors used in the primary evaluation are distinct from the 147-anchor layouts used in the later matched studies. `anchor-layouts.json` records the latter cohorts. The manifest lists file-level SHA-256 hashes.

In the released implementation the 121-anchor engineering grid uses a periodic azimuth × level bilinear stencil with shared pole and seam handling. Strict support requires every nonzero stencil weight to refer to an allowed anchor. Periodic Delaunay is used for the 147-anchor and matched-layout comparisons. `forward/reference/` contains the original solver algorithms with local deployment identifiers redacted. It is an archival implementation, with no dependency on the Pages site for metric reproduction. See `forward/README.md` for the scope and external inputs of the full forward pipeline.

## Interactive demo

For local use, open `docs/index.html` directly. The page supports touch selection, dragging, rotation and zoom. JavaScript evaluates the two frozen networks for seed 20261110 and the cosine-template baseline using the 121-anchor engineering field. Below the engineering support boundary, the page reports missing support. The teaching signal visualization and the research inference panels are labeled separately. The published summary tables report all three seeds where applicable. The browser needs only the static files included here.

## Citation and reuse

See [CITATION.cff](CITATION.cff) for the study title and scope. The manuscript is a review draft. Author and journal metadata will be completed by the authors. Code, data and weights are distributed as study artifacts. No third-party dependency or publication is relicensed here.
