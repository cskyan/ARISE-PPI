# L131 Revision Experiments

This package implements the additional ARISE-PPI experiments on the L131 model
line. The primary pair model is the explicit:

```text
residue encoders -> EvidenceBridge -> L3Head
```

The primary route does not use L2Bridge or GC-EB/GPEH.

## Complete Revision Pipeline

The complete workflow is resumable. Existing checkpoints and completed outputs
are skipped unless `--force` is supplied:

```bash
cd /srv/storage1/ssd/ysk/jiangbo/PhD/new2

python -m experiments_l131.run_revision_pipeline \
  --project-root /srv/storage1/ssd/ysk/jiangbo/PhD/new2 \
  --pp-root /srv/storage1/ssd/ysk/jiangbo/PhD/new2/pp_prepared \
  --rbp-root /srv/storage1/ssd/ysk/jiangbo/PhD/new2/RBP400 \
  --output-root /srv/storage1/ssd/ysk/jiangbo/PhD/new2/results/plos_revision \
  --sequence-mode hybrid \
  --esm-local-dir /path/to/esm/resources \
  --seeds 1337,2027,3407,4517,5651 \
  --random-repetitions 100
```

To generate and audit only the CPU-side manifest and splits:

```bash
python -m experiments_l131.run_revision_pipeline \
  --project-root /srv/storage1/ssd/ysk/jiangbo/PhD/new2 \
  --prepare-only
```

The full pipeline produces:

```text
manifests/
  pp_prepared_labeled_pairs.tsv
  pp_prepared_labeled_pairs.summary.json
  pp_prepared_feature_audit.tsv
  pp_prepared_feature_audit.summary.json
results/splits/pp_component_seed1337/
  train.tsv
  val.tsv
  test.tsv
  split_manifest.tsv
  split_audit.json
results/plos_revision/
  pipeline_status.json
  logs/
  controls/<variant>/seed_<seed>/
  validation/{native_eb,symmetry,faithfulness}/
  test/{native_eb,symmetry,faithfulness}/
  structure/contact_enrichment/
  hcc/{native_eb,symmetry,faithfulness,degree_null}/
  summary/
    control_metrics_by_seed.tsv
    control_metrics_aggregate.tsv
    faithfulness_summary.tsv
    revision_results_summary.json
```

## Project Root

On the original server, run module commands from the repository root:

```bash
cd /srv/storage1/ssd/ysk/jiangbo/PhD/new2
export ARISE_PPI_ROOT=/srv/storage1/ssd/ysk/jiangbo/PhD/new2
```

`config_L131.py` uses its own directory as the default project root. When the
file is located at the path above, default data and output paths are resolved
under that directory instead of the current shell directory.

Do not run `python -m experiments_l131.<module>` while the shell is inside the
`experiments_l131` directory. Return to the repository root or execute the
script directly from that directory.

## Environment

Create an isolated environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

CUDA users should install the PyTorch build matching their CUDA runtime before
installing the remaining requirements.

## Pair Manifest

Training, validation, and test tables are TSV or CSV files with these required
columns:

```text
pair_id  protein_A  protein_B  label
```

`label` must be binary. Optional provenance columns such as `data_source`,
`anchor_id`, and `evidence_type` are preserved by the split tools.

The repository does not assume that a labeled pair table already exists. A
fixed development manifest can be built from `pp_prepared`:

```bash
cd /srv/storage1/ssd/ysk/jiangbo/PhD/new2
python -m experiments_l131.build_pair_manifest \
  --root /srv/storage1/ssd/ysk/jiangbo/PhD/new2/pp_prepared \
  --output /srv/storage1/ssd/ysk/jiangbo/PhD/new2/manifests/pp_prepared_labeled_pairs.tsv \
  --negative-ratio 1 \
  --seed 1337
```

Same-complex ligand/receptor chains are structural positives. Cross-complex
pairs are frozen synthetic negatives and are explicitly marked as such; they
must not be described as experimentally verified non-interactions.

## Build Frozen Splits

```bash
cd /srv/storage1/ssd/ysk/jiangbo/PhD/new2
python -m experiments_l131.build_splits \
  --manifest /path/to/labeled_pairs.tsv \
  --strategy both_unseen \
  --seed 1337 \
  --out-dir /srv/storage1/ssd/ysk/jiangbo/PhD/new2/results/splits/both_unseen
```

Equivalent command when already inside `experiments_l131`:

```bash
python build_splits.py \
  --manifest /path/to/labeled_pairs.tsv \
  --strategy both_unseen \
  --seed 1337 \
  --out-dir ../results/splits/both_unseen
```

Available strategies are `pair_random`, `one_unseen`, `both_unseen`,
`homology`, `component`, and `anchor`. Homology splitting additionally requires
`--clusters` with `protein_id` and `cluster_id` columns.

The command writes `train.tsv`, `val.tsv`, `test.tsv`, the combined split
manifest, excluded-pair ledger, and leakage audit.

## Train the Primary Model

```bash
DATA_ROOT=/path/to/prepared/proteins \
PAIR_TRAIN_MANIFEST=/path/to/train.tsv \
PAIR_VAL_MANIFEST=/path/to/val.tsv \
PAIR_TEST_MANIFEST=/path/to/test.tsv \
DATASET_MODE=pair \
PRIMARY_OBJECTIVE=pair \
PAIR_HEAD_TYPE=eb \
EB_PROPOSAL_MODE=top \
EB_SUPPORT_MODE=top \
MANUSCRIPT_PRIMARY=1 \
SAVE_DIR=results/full_eb/seed_1337 \
SEED=1337 \
python train_L131.py
```

Explicit manifests disable dynamic in-batch negative generation. The validation
threshold is selected during validation and frozen for test evaluation.

## Run Matched Controls

```bash
python -m experiments_l131.run_matched_controls \
  --data-root /path/to/prepared/proteins \
  --train-manifest /path/to/train.tsv \
  --val-manifest /path/to/val.tsv \
  --test-manifest /path/to/test.tsv \
  --output-root results/controls
```

The default matrix runs five seeds for the full EB model, no-EB model,
attention pooling, random proposals, low proposals, and all-candidate
aggregation. Use `--dry-run` to write and inspect the run manifest without
starting training.

## Export Native EB Evidence

```bash
python -m experiments_l131.export_native_eb \
  --root /path/to/prepared/proteins \
  --checkpoint /path/to/best_checkpoint.pt \
  --manifest /path/to/test.tsv \
  --out-dir results/native_eb \
  --require-native-eb \
  --export-candidates
```

This exporter reads retained supports and normalized support weights directly
from EvidenceBridge. It does not reconstruct residue pairs from independent
site probabilities.

## Symmetry and Faithfulness

```bash
python -m experiments_l131.evaluate_symmetry \
  --root /path/to/prepared/proteins \
  --checkpoint /path/to/best_checkpoint.pt \
  --manifest /path/to/test.tsv \
  --out-dir results/symmetry

python -m experiments_l131.evaluate_faithfulness \
  --root /path/to/prepared/proteins \
  --checkpoint /path/to/best_checkpoint.pt \
  --manifest /path/to/test.tsv \
  --out-dir results/faithfulness \
  --random-repetitions 100
```

Faithfulness includes top, low, and random residue deletion, zero and
background replacement, frozen-support deletion, comprehensiveness,
sufficiency, and AOPC.

## Structural Contact Enrichment

First build a structure manifest:

```bash
python -m experiments_l131.build_structure_manifest \
  --dataset pp \
  --root pp_prepared \
  --output results/structure/pp_structure_manifest.tsv

python -m experiments_l131.build_structure_manifest \
  --dataset dest \
  --root Dest_prepared \
  --output results/structure/dest_structure_manifest.tsv
```

Then evaluate native supports:

```bash
python -m experiments_l131.evaluate_contacts \
  --structure-manifest results/structure/pp_structure_manifest.tsv \
  --supports results/native_eb/native_eb_supports.tsv \
  --candidates results/native_eb/native_eb_candidates.tsv \
  --out-dir results/structure/contact_enrichment
```

The evaluator reports C-alpha contacts for every mapped complex and heavy-atom
contacts when an original PDB file is available. Both all-valid and
candidate-matched null distributions are included.

## HCC Evidence Ledger and Degree-Matched Null

```bash
python -m experiments_l131.build_hcc_esi \
  --predictions results/hcc/predictions.tsv \
  --faithfulness results/hcc/faithfulness.tsv \
  --symmetry results/hcc/symmetry.tsv \
  --validation-predictions results/validation/native_eb/pair_mode_native_EB_outputs.tsv \
  --validation-faithfulness results/validation/faithfulness/faithfulness_per_pair.tsv \
  --validation-symmetry results/validation/symmetry/swap_symmetry_per_pair.tsv \
  --output results/hcc/esi_ledger.tsv \
  --summary results/hcc/esi_summary.json

python -m experiments_l131.degree_matched_null \
  --network RBP400/annotations/string_network.tsv \
  --module results/hcc/selected_module.tsv \
  --universe results/hcc/candidate_universe.tsv \
  --out-dir results/hcc/degree_null \
  --repetitions 1000
```

All ESI thresholds are frozen from validation inputs. The ledger retains every
candidate pair, including failed gates and missing-evidence reasons.

## Reproducibility Rules

- Never infer negative labels from missing STRING edges or low model scores.
- Never use the legacy Cartesian-product residue-pair export as native EB evidence.
- Keep split manifests, exclusion ledgers, checkpoint provenance, and run manifests.
- Select thresholds on validation data only.
- Report seed-level results before aggregate statistics.
