# L131 Revision Experiments

This package implements the additional ARISE-PPI experiments on the L131 model
line. The primary pair model is the explicit:

```text
residue encoders -> EvidenceBridge -> L3Head
```

The primary route does not use L2Bridge or GC-EB/GPEH.

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

## Build Frozen Splits

```bash
python -m experiments_l131.build_splits \
  --manifest data/pairs.tsv \
  --strategy both_unseen \
  --seed 1337 \
  --out-dir results/splits/both_unseen
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
  --pp-root pp_prepared \
  --dest-root Dest_prepared \
  --out results/structure/structure_manifest.tsv
```

Then evaluate native supports:

```bash
python -m experiments_l131.evaluate_contacts \
  --structure-manifest results/structure/structure_manifest.tsv \
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
  --validation-predictions results/validation/predictions.tsv \
  --validation-faithfulness results/validation/faithfulness.tsv \
  --validation-symmetry results/validation/symmetry.tsv \
  --out-dir results/hcc/esi

python -m experiments_l131.degree_matched_null \
  --network RBP400/annotations/string_network.tsv \
  --module results/hcc/esi/selected_module.tsv \
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
