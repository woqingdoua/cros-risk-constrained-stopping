# LA-CDM training overlay and common-path generation

This directory supplies the study's author-created development-only dataset adapter, ED/Ext-CDS prompts and configuration. The official trainer, environment and modified upstream code remain external. Their frozen reference commit is `3f435a10f569c3255172c1ef8fe7880c6edb7fee` at https://github.com/dharouni/LA-CDM. No explicit LICENSE was found in the inspected frozen tree. Obtain separate authorization for the upstream source before reproducing this dependency. No patch context, removed upstream lines or full upstream files are redistributed in this public bundle. See LICENSE_STATUS.md for attribution.

## Manual integration in a separately authorized checkout

Verify the upstream commit above and the original-file SHA-256 values recorded in `docs/upstream_patch_validation.json` before editing. That record also contains the retained historical target hashes. Release QA previously verified the historical modifications privately against those hashes; the patches used for that check are excluded from the public release. Manual formatting changes can alter a file hash, so target hashes are historical provenance rather than a claim that these prose instructions uniquely reproduce exact source bytes.

Make these study-specific changes in your separately authorized checkout:

1. In the dataset preparation function in `src/utils/setup.py`, import the supplied `EDCDSDataset` adapter. When the dataset format is `ed_cds_jsonl`, construct that adapter with the designated split data file, test-name/action-key mapping and small-sample flag. Preserve the original path for other dataset formats.
2. In both training and evaluation classification-metric calls in `src/trainer/cdm_grpo_trainer.py`, pass the environment's complete disease list as the class set. This preserves the nine-class answer space.
3. In `src/train.py`, support an optional configured resume checkpoint. Validate its basename as checkpoint followed by a numeric global step; advance the custom loss scheduler that many steps before resuming the trainer. A resumed run must not restart the decision/hypothesis/confidence phase cycle. With no resume checkpoint, use the normal training path.
4. Update the upstream dependency configuration to the historical compatibility versions listed below. Require Python 3.11, rather than the upstream 3.11–3.12 range. No modified upstream dependency file is distributed here.

Copy only the author-created additions from this directory into matching paths in that authorized checkout: `src/dataset/ed_cds_dataset.py`, the five `configs/*/ed_cds*.yaml` files and the two `prompts/*_ed_cds.txt` files. Preserve upstream notices. These additions plus manual changes describe the actual paper training integration, but the public package is not a self-contained licensed distribution of the upstream trainer.

Historical training requires Python 3.11, Torch 2.6.0, TRL 0.15.2, PEFT 0.15.0, Accelerate 1.5.2, vLLM 0.8.3, xgrammar 0.1.17, Transformers 4.51.0, huggingface-hub 0.30.2, numba 0.61.0 and llvmlite 0.44.0. Consult the separately authorized upstream dependency configuration for its remaining packages. Do not interpret the release's smaller cached-analysis requirements as the training environment.

## Controlled data and training

Build the clinical benchmark with `cros_mix_validation/build_ed_cds_action_benchmark.py` in a credentialed environment. Then partition development only:

```bash
cd "$CROS_RELEASE"
python cros_mix_validation/materialize_la_cdm_development_split.py \
  --source "$CROS_DATA_ROOT/ed_cds_action_v1/development_episodes.jsonl.gz" \
  --output-root "$CROS_DATA_ROOT/la_cdm_train_v1" --seed 20260829
```

`CROS_DATA_ROOT` must be an absolute controlled-data directory; the overlay's Hydra paths use this variable. Its test_data_file deliberately points to development validation, and training loads only train/val. Keep every derived clinical file private. The adapter rejects non-development records and treats absent results as unknown.

With an already authorized local base checkpoint and your assembled training environment:

```bash
cd "$LA_CDM_ROOT"
export VLLM_USE_V1=0 WANDB_MODE=disabled
python -m accelerate.commands.launch --num_processes 1 -m src.train \
  dataset=ed_cds environment=ed_cds reward_function=ed_cds experiment=ed_cds_full \
  model.model_name="$QWEN_LOCAL_PATH" wandb.mode=disabled \
  hydra.run.dir="$CROS_PRIVATE_OUTPUT/final_adapter"
```

Four epochs and 3,740 historical optimizer steps, seed 269; LoRA rank 8/scale 16/dropout .1; 8 GRPO generations, DR-GRPO, beta .04, learning rate 1e-5. The 100-step decision-GRPO, hypothesis-SFT and confidence-GRPO phases repeat. Validation/save occur once per epoch. The final epoch-4 adapter was the primary checkpoint. The historical resume modification advances the custom loss scheduler to the checkpoint step, preserving the phase sequence.

## Forced-continuation trajectory generation

For each shard (0 through 10), run the shipped common-path scorer from the CROS root, using an existing local model/adapter and authorized episodes:

```bash
python cros_mix_validation/score_la_cdm_common_path_shard.py \
  --model-path "$QWEN_LOCAL_PATH" --adapter-path "$CROS_PRIVATE_OUTPUT/final_adapter" \
  --development-path "$CROS_DATA_ROOT/la_cdm_train_v1/validation_episodes.jsonl.gz" \
  --benchmark-manifest "$CROS_DATA_ROOT/ed_cds_action_v1/manifest.json" \
  --output-root "$CROS_PRIVATE_OUTPUT/selection/shards" \
  --shard-index 0 --num-shards 11 --batch-size 8 --max-tests 12 --seed 20260830
```

Change only the designated input/output split and required split label for ranker-training, calibration and evaluation, keeping frozen generation settings. Use the exact retained run metadata for historical replay; the precise original generation source commit was not recorded. The scorer uses local_files_only for model/tokenizer and does not itself download weights. Outputs are patient-level clinical arrays and must never leave the controlled environment.

The prompt-transfer common path is not an exact reproduction of upstream free-form LA-CDM rollouts. CROS controls stopping/deferral on shared cached trajectories; it does not estimate a causal advantage of a different acquisition agent.
