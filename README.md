# CROS: risk-constrained stopping for clinical diagnosis agents

Research code for **Safe to Stop? Risk-Constrained Stopping for Sequential Clinical Diagnosis Agents**, Yuexin Wu and Vasile Rus, GenAI4Health 2026 (paper 98). [Paper (arXiv)](https://arxiv.org/abs/2609.09678).

CROS studies when a sequential diagnosis agent should stop, acquire another result, or defer. A development-fitted risk ranker defines threshold-and-horizon policies; deterministic policies and episode-level randomized mixtures trade diagnostic error, autonomous coverage, and acquisition cost. The retrospective experiment uses a fixed common information path from a Qwen2.5-7B-Instruct backbone adapted using an official-code-derived LA-CDM pipeline. It compares stopping and deferral decisions while holding the recorded diagnosis/acquisition trajectory fixed.

This release contains explicitly allowlisted source code, aggregate paper tables, and synthetic smoke tests. It excludes all patient records, clinical arrays, identifiers, outcomes, reconstructed assignment vectors, restricted bundles, model weights, and development history. See [reproducibility](docs/REPRODUCIBILITY.md), [artifact mapping](docs/ARTIFACTS.md), and [license status](LICENSE_STATUS.md).

## Quick start

Use Python 3.11 in a virtual environment. Historical software versions were not fully retained; NumPy 1.26.4 is the recorded reconstruction environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/synthetic_smoke.py
# Pure synthetic finite-state proof-of-value experiment (not clinical evidence):
python cros_mix_validation/validate_cros_mix.py --output-dir /tmp/cros-synthetic --repeats 10 --seed 20260825
```

## Optional generation and adapter checks

The quick start above uses only cached-analysis dependencies. Install `requirements-generation.txt` in a separate GPU/model environment for the trajectory scorer and Torch-based adapter. The optional synthetic adapter test is:

```bash
python tests/synthetic_adapter_smoke.py
```

## Interpretation

The theoretical certificate assumes i.i.d. episodes, independent calibration, policies and ordering frozen on development data, and independent per-episode mixture draws. Mixture weights must not be fitted on calibration outcomes. The reported retrospective evidence is exploratory; it is not a realized clinical safety certificate, causal clinical benefit, subgroup guarantee, or deployment validation.

Selective-risk comparisons can reflect different accepted patients. The additional matched-cohort analysis holds each controller's accepted subset fixed when comparing initial, stopping, and full-workup errors. It uses 10,000 paired patient-bootstrap replicates, seed 20261010. The mixture is a reconstructed prespecified draw: frozen weights/support, sorted episode order, and seed 20260905 with NumPy PCG64 reproduce evaluator outputs and historical counts. Neither the historical assignment vector nor historical NumPy version was retained; historical bitwise identity cannot be established. Machine metadata named `original_assignment_verified` describes the reconstruction check, not independent historical verification.

No training, backbone inference, or full clinical experiment was rerun for this release. See CITATION.cff for attribution; no DOI has been invented.
