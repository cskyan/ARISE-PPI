# ARISE-PPI PLOS Experiment Code and Data Plan

## 1. Blocking provenance issues

The primary manuscript model must use the explicit pair-mode `EvidenceBridge -> L3Head`
path. The current model code contains three different pair routes:

1. explicit Top-K-by-Top-K `EvidenceBridge`;
2. explicit EB fused with `L2Bridge`;
3. GC-EB/GPEH, selected by the current default `pair_head_type="gpeh"`.

Before any new experiment, add a dedicated `pair_head_type="eb"` route that sends the
unmodified `EvidenceBridge.evi_vec` directly to `L3Head`. Primary outputs must assert:

```text
site_mode=false
pair_head_type=eb
l2_active=false
gc_eb_active=false
native_eb_export=true
```

The existing RBP400 case-study outputs are useful only as development fixtures:

- the exporter loads a residue-site `best_TOPK.pt` checkpoint and forces pair mode;
- `topK_residue_pairs.tsv` is generated from the Cartesian product of residue
  probabilities, not native EB support scores;
- `bridge_sufficiency_score` is an ad hoc summary, not an intervention-based
  sufficiency metric;
- current occlusion masks chain A only, uses zero replacement only, has 20 random
  repetitions, and does not evaluate low evidence or native support deletion.

These outputs must not be used as primary PLOS evidence.

## 2. Local data that can be reused

| Local resource | Observed content | Recommended use |
|---|---:|---|
| `RBP400/` | 400 labels, 400 DSSP files, 471 sequence files, 721 structure files | RBP candidate universe, residue features, HCC candidate scoring |
| `RBP400/pairs/anchor_candidate_pairs.tsv` | 1,318 pairs, 4 anchors, 336 proteins | Exploratory anchor-centred inference |
| `RBP400/pairs_strict/anchor_candidate_pairs.tsv` | 828 pairs, 4 anchors, 314 proteins | Frozen strict RBP candidate manifest |
| `RBP400/pairs_strict/hcc_all_pairs.tsv` | 16,471 unlabeled HCC candidates | HCC-ESI inference only |
| `RBP400/pairs_strict/lung_all_pairs.tsv` | 27,261 unlabeled lung candidates | Exploratory inference only |
| `RBP400/annotations/string_network.tsv` | 11,184 STRING edges and channel scores | Degree matching and external support, after version/provenance freeze |
| `pp_prepared/` | 1,251 chains; 271 PDB groups with both ligand/receptor sides | Primary structural contact-enrichment cohort |
| `Dest_prepared/` | 422 chains; 51 multi-chain PDB groups with usable coordinates | Independent contact-enrichment sensitivity cohort |
| `GPSite_dataset/` | Prepared residue-site benchmarks | Residue-site validation only; not pair-level EB validation |
| existing case outputs | 2,509 pair predictions and occlusion rows | Regression tests and pipeline debugging only |
| `pymol_package_strict/` | Two mapped example pairs | Qualitative visualization only |
| local BioLiP cache | BioLiP annotation archive | Optional structural/functional mapping support |

Important label boundary:

- RBP400 candidate pair tables do not contain supervised pair labels.
- Model predictions, STRING support, and unlabeled candidates must not be converted
  into ground-truth negatives.
- A frozen labelled pair dataset is still required for pair AUPRC/MCC experiments.
- No local TUnA, TCGA-LIHC, or DepMap files were found during this audit.

## 3. Required code architecture

Create a dedicated experiment package:

```text
experiments/
  common/
    io.py
    metrics.py
    bootstrap.py
    multiple_testing.py
    provenance.py
  splits/
    build_pair_manifest.py
    build_splits.py
    audit_splits.py
    cluster_sequences.py
  controls/
    variants.py
    run_matched_controls.py
  symmetry/
    evaluate_swap.py
  faithfulness/
    residue_interventions.py
    support_interventions.py
    run_faithfulness.py
  structure/
    build_contact_cohort.py
    evaluate_contact_enrichment.py
  hcc/
    build_esi_ledger.py
  disease/
    tcga_lihc.py
    depmap_liver.py
    degree_matched_null.py
    functional_enrichment.py
configs/
  experiments/
results/
  manifests/
  splits/
  controls/
  faithfulness/
  structure/
  hcc/
  disease/
```

## 4. Core model changes

### 4.1 Explicit EB-only route

Add a model route:

```text
xA, xB
  -> residue logits
  -> EvidenceBridge
  -> evi_vec, evi_score, native supports
  -> L3Head(global_A, global_B, evi_vec)
  -> pair_logit
```

Do not instantiate or call `L2Bridge`, GC-EB, or GPEH for the primary route.

### 4.2 Native EB export

`EvidenceBridge` must export:

```text
topk_idxA
topk_idxB
topk_valA
topk_valB
candidate_idxA
candidate_idxB
candidate_score
retained_idxA
retained_idxB
retained_score
retained_weight
evi_vec
evi_score
pair_logit
pair_prob
```

The normalized retained weights are currently computed but not returned. Return them
without detaching during training and with detached copies during export.

### 4.3 Intervention-safe decomposition

Split the current monolithic forward path into:

```text
encode_pair(...)
score_residues(...)
build_eb_candidates(...)
aggregate_eb_supports(...)
classify_pair(...)
```

This is required to delete supports without reranking. Faithfulness interventions
must freeze the original proposal and support order.

### 4.4 Mechanism variants

Add explicit config enums:

```text
pair_head_type: eb | no_eb | attention | mean | gc_eb_gpeh
proposal_mode: top | random | low | shuffled
support_mode: top | all_candidates | dense
partner_context: true | false
```

Every run must export the selected variant, trainable parameter count, runtime, peak
memory, candidate count, and retained support count.

## 5. Dataset and split changes

### 5.1 Frozen explicit pair table

Replace dynamic in-batch negative generation for primary evaluation with an explicit
pair manifest:

```text
pair_id
protein_A
protein_B
sequence_A
sequence_B
label
data_source
anchor_id
evidence_type
```

Canonicalize unordered pairs, remove self-pairs, remove duplicate orientations, and
stop on positive/negative conflicts.

Dynamic shuffled negatives may remain an auxiliary training strategy, but they cannot
define the held-out evaluation cohort.

### 5.2 Split builders

Implement:

- pair-random;
- one-unseen;
- both-unseen;
- homology-controlled both-unseen;
- component-disjoint;
- anchor-disjoint;
- all-role protein-disjoint sensitivity.

Sequence clustering should call a frozen external tool such as MMseqs2 and record the
tool version, command, identity threshold, coverage threshold, and cluster file hash.

### 5.3 Split audit

For every split, export:

```text
pair counts
positive prevalence
unique protein counts
unique anchor counts
protein overlap
anchor overlap
component counts
degree distribution
train-to-test sequence identity
removed/conflicting pair counts
```

Any overlap violation must fail the run.

## 6. Metrics and statistics

Add pair-level metrics:

- AUPRC;
- MCC;
- AUROC;
- F1;
- sensitivity;
- specificity;
- Brier score;
- expected calibration error.

The validation threshold must be selected once and frozen for test evaluation.

Add statistical utilities:

- paired bootstrap by pair or complex;
- seed-paired differences;
- paired Wilcoxon test;
- rank-biserial effect size;
- Benjamini-Hochberg correction;
- empirical null probability;
- exact sample and exclusion counts.

Do not bootstrap residues within a pair.

## 7. Experiment-specific implementation

### Experiment 1: strict splits

Status: missing.

Immediate dependency: a labelled pair manifest. RBP400 alone cannot satisfy this
requirement because its candidate pair tables are unlabeled.

### Experiment 2: matched EB controls

Status: explicit EB exists, but primary routing and controls are missing.

Implement the six primary variants first. Use identical split manifests, seeds,
optimization budgets, and checkpoint selection.

### Experiment 3: chain-order symmetry

Status: no complete evaluator exists. A config flag is present but not sufficient.

Run both orientations and align:

- pair probability;
- residue scores;
- Top-K residue sets;
- retained support indices.

Export one row per pair before aggregation.

### Experiment 4: faithfulness

Status: a limited residue zero-occlusion prototype exists.

Extend it to:

- both chains independently;
- proportional and fixed budgets;
- top, random, and low evidence;
- zero and chain-matched background replacement;
- native support deletion with frozen ranking;
- residue and support sufficiency;
- comprehensiveness;
- probability and logit AOPC;
- at least 100 random repetitions per pair.

### Experiment 5: structural contact enrichment

Status: data are available; native EB evaluation is missing.

Recommended cohorts:

1. `pp_prepared`: primary cohort, 271 PDB groups with both sides;
2. `Dest_prepared`: sensitivity cohort, 51 coordinate-complete multi-chain groups.

RBP400 AlphaFold monomers and the two-pair PyMOL package are not adequate as the
primary structural cohort.

Compute heavy-atom contacts from raw PDB/mmCIF when available. Use CA-distance below
8 A as the required sensitivity analysis. Evaluate both all-valid and
candidate-matched nulls within each complex.

### Experiment 6: HCC-ESI

Status: candidate predictions exist, but provenance and faithfulness criteria are not
valid under the new protocol.

Rebuild the edge ledger after Experiments 2-4. Keep provenance categories separate:

```text
confirmed_positive
curated_negative
unlabeled_candidate
```

### Experiments 7-8: TCGA-LIHC and DepMap

Status: data and analysis code are missing locally.

Do not start these analyses until the HCC-ESI rule is frozen without consulting
disease outcomes.

### Experiment 9: degree-matched null

Status: local STRING network is reusable; the sampler is missing.

Freeze the STRING version and degree definition. Match node count and degree bins for
at least 1,000 modules. Match edge count/density for edge-level tests.

### Experiment 10: enrichment

Status: exploratory enrichment outputs exist.

Rerun with:

- the frozen RBP400 universe as background;
- database/version metadata;
- BH correction;
- comparison modules;
- degree-matched nulls;
- separated STRING evidence channels.

## 8. Execution order

### Phase 0: provenance gate

1. Choose one canonical source file set.
2. Restore a clean source package.
3. implement and test the EB-only route.
4. obtain a genuine pair-mode checkpoint.
5. export a reproducibility manifest.

### Phase 1: supervised evaluation foundation

1. obtain/freeze the labelled pair manifest;
2. build and audit strict splits;
3. add pair-level calibration and statistical utilities;
4. run a one-seed smoke test.

### Phase 2: decisive model experiments

1. full EB and five primary controls;
2. five paired seeds;
3. chain-order symmetry;
4. paired bootstrap and multiplicity correction.

### Phase 3: mechanistic validation

1. residue faithfulness;
2. support faithfulness;
3. PP contact enrichment;
4. Dest contact-enrichment sensitivity.

### Phase 4: biological application

1. rebuild HCC-ESI;
2. degree-matched modules;
3. TCGA-LIHC;
4. DepMap;
5. enrichment with the frozen RBP400 background.

## 9. Immediate next implementation batch

The first coding batch should contain only:

1. explicit `pair_head_type="eb"` routing;
2. native support weights and candidate exports;
3. model metadata assertions;
4. explicit labelled pair-manifest loader;
5. split builder and overlap audit;
6. pair-level metrics including Brier score and ECE;
7. one smoke-test configuration.

Do not begin TCGA, DepMap, or publication figure generation before this batch passes.
