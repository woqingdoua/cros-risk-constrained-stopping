# Reproducibility and controlled data

Obtain MIMIC-IV-ED v2.2 and MIMIC-IV-Ext-CDS v1.0.2 directly from PhysioNet after completing its credentialing, required training, and each applicable data-use agreement. Keep all original and derived clinical files inside the authorized environment. The release supplies no data download credentials and does not grant access.

The canonical cohort retains the earliest qualifying stay per patient: 1,834 episodes, nine abdominal-pain classes; 935 ranker-training, 165 policy-selection, 367 calibration, and 367 evaluation episodes. The targets are selective risk 0.25, coverage 0.70, and family-wise error 0.05. Development construction crosses 10 coverage targets with 13 horizons, producing 130 candidates; the tested deterministic family has 12 policies. Selection margins are 20% risk and 80% coverage. Thresholds, support, and weights must remain frozen.

In a controlled environment containing the canonical cached shards:

```bash
python cros_mix_validation/evaluate_cros_power_v2.py \
  --root cros_mix_validation/results/la_cdm_cros_power_v2 \
  --validation-root cros_mix_validation/results/la_cdm_trained_common_backbone_validation_v1/shards
python cros_mix_validation/verify_cros_power_v2.py --help
```

The evaluator includes development fitting and candidate freezing; running it with new inputs creates a new experiment, not a replay of historical frozen policies. Exact historical reconstruction additionally requires the controlled manifests, shards and frozen artifacts whose hashes are checked by `extract_verified_same_accepted.py`.

To replay only the reviewer analysis, create a fresh controlled run directory, copy the two reviewer scripts into it, then run:

```bash
python cros_mix_validation/extract_verified_same_accepted.py --root cros_mix_validation --run-dir /controlled/fresh-review-run
python /controlled/fresh-review-run/analyze_same_accepted_cached.py \
  --input /controlled/fresh-review-run/private/same_accepted_input.csv \
  --provenance /controlled/fresh-review-run/private/extraction_provenance.json \
  --output-dir /controlled/fresh-review-run/aggregate
```

The extractor enforces original code/cache hashes. Do not bypass mismatches or export its private outputs. Its frozen analyzer hash refers to the unmodified historical source; retain exact source bytes. Only reviewable aggregate outputs may be considered for external release. Aggregate counts cannot reproduce a paired patient bootstrap by themselves.

Backbone provenance: Qwen2.5-7B-Instruct, LA-CDM reference commit 3f435a1, LoRA rank 8/scale 16/dropout 0.1, four epochs, 3,740 optimizer steps, seed 269. The exact historical generation commit was not recorded. The release now includes cohort construction, development partitioning, the author-created LA-CDM training additions and manual compatibility instructions, exact common-path trajectory scoring, stopping/ranking, frozen comparisons and matched reanalysis. See `integration/la_cdm/README.md` for training/generation commands and upstream access. The upstream trainer is an external dependency rather than vendored source. Controlled data, historical adapters, cached trajectories and incomplete historical generation provenance still prevent public bitwise end-to-end replay.
