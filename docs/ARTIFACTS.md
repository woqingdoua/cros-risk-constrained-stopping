# Artifact mapping

- `cros_mix_validation/evaluate_cros_power_v2.py`: canonical ranker, candidate family, mixture and calibration analysis.
- `cros_mix_validation/verify_cros_power_v2.py`: independent cached-result checks.
- `cros_mix_validation/build_ed_cds_action_benchmark.py`: controlled cohort/label construction.
- `cros_mix_validation/validate_cros_mix.py`: finite-state synthetic mixture validation; separate from clinical results.
- `cros_mix_validation/{extract_verified_same_accepted,analyze_same_accepted_cached}.py`: exact reviewer reanalysis source from camera-ready package.
- `results/cros_completion_20260828_v1/aggregate_completion/`: frontier, matched-coverage and stagewise aggregate tables.
- `results/genai4health_additions/v2_completion_20260828/`: aggregate cost sensitivity.
- `results/reviewer_same_accepted_20261010/same_accepted_summary.csv`: accepted-cohort paired risk/contrast summaries.
- `docs/source_code_manifest.json`: original and staged code hashes; filesystem-default changes only.

Camera-ready input: GenAI4Health-Paper-98-Camera-Ready-Source.zip (Library version 0), materialized and inspected locally. Manuscript/styles are not bundled into this code release; no third-party style license has been silently reassigned.

- `integration/la_cdm/`: author-created development-only dataset adapter, paper training/configuration overlay, prompts and manual upstream integration instructions.
- `cros_mix_validation/score_la_cdm_common_path_shard.py`, `score_la_cdm_zs_port_shard.py`: actual adapted common-path generation implementation.
- `cros_mix_validation/materialize_la_cdm_development_split.py`: development-only train/validation partitioning.
- `cros_mix_validation/evaluate_cros_vs_la_cdm_frozen.py`, `verify_cros_vs_la_cdm_frozen.py`: final frozen-controller comparisons and independent checks.
- `results/reviewer_same_accepted_20261010/same_accepted_summary.json`: aggregate result, frozen support/weights and reconstruction provenance; server path prefixes made relative.
