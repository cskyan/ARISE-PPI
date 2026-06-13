# ARISE-PPI

ARISE-PPI is a residue-level protein interaction analysis framework with two selectable operating modes:

- `topk`: residue ranking and top-k evidence discovery.
- `binary`: residue-site binary classification with validation-calibrated thresholds.

The repository is organized as a clean release package. Large datasets, ESM weights, checkpoints, and generated outputs are intentionally kept outside Git.

## Highlights

- Unified training entry point for top-k and binary objectives.
- Unified prediction entry point with split selection, explicit checkpoint loading, and configurable output directories.
- ESM, PSSM, DSSP, sequence, and structure-aware residue features supported by the core model.
- Top-k residue outputs for downstream case studies and evidence ranking.
- Binary prediction mode with threshold handling and standard classification metrics.
- One generic config file: `configs/config.example.json`.

## Repository Layout

```text
ARISE-PPI/
  configs/
    config.example.json
  scripts/
    train.py
    predict.py
  src/
    arise_ppi/
      __init__.py
      cli_utils.py
      config.py
      model.py
      train_core.py
      predict_core.py
  .gitignore
  pyproject.toml
  requirements.txt
  README.md
```

## Environment Setup

Python 3.9 or newer is recommended.

Create and activate an environment:

```bash
conda create -n arise-ppi python=3.9 -y
conda activate arise-ppi
```

Install PyTorch according to your CUDA version from the official PyTorch instructions, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

For editable development:

```bash
pip install -e .
```

Core packages:

```text
torch
fair-esm
numpy
scikit-learn
```

## Data Layout

Place data outside Git or under the ignored `data/` directory.

## Benchmark Data Sources

The benchmark datasets used by ARISE-PPI are distributed from the following sources.
RBP400 is curated by the authors and is intended to be provided by this repository
or the accompanying release package. The other public benchmarks should be
obtained from their original public resources.

| Local folder | Benchmark setting | Data access |
| --- | --- | --- |
| `Data/RBP400/` | RBP400 disease-aware resource | Curated by the authors and provided with this repository or its release package. |
| `Data/Dset/` | Dset_186_72_PDB164 residue-level PPI-site benchmark, combining Dset_186, Dset_72, and PDBset_164 | DeepPPISP GitHub repository: <https://github.com/CSUBioGroup/DeepPPISP>; CSBIO PPIS benchmark page: <https://csbio.njust.edu.cn/bioinf/PPIS/>. |
| `Data/PP/` | GraphRBF-PP protein-protein residue-level benchmark | GraphRBF GitHub repository: <https://github.com/Wssduer/GraphRBF>. |
| `Data/Tuna/` | TUnA protein-pair-level benchmark | TUnA GitHub repository: <https://github.com/Wang-lab-UCSD/TUnA>. |
| `Data/GPSite/` | GPSite-PRO external protein-binding residue benchmark | GPSite GitHub repository: <https://github.com/biomed-AI/GPSite>; GPSite eLife reviewed preprint: <https://elifesciences.org/reviewed-preprints/93695>. |

After downloading a public benchmark, place or convert the files into the
corresponding local folder shown above before running training or prediction.

Expected layout:

```text
data/
  dataset/
    seq/
    PSSM/
    dssp_rsa_asa/
    labels/
    structures/
  all_ids.txt
  train.txt
  val.txt
  test.txt
resources/
  esm/
    esm2_t33_650M_UR50D.pt
```

Required split files:

```text
all_ids.txt
train.txt
val.txt
test.txt
```

Typical feature folders:

```text
seq/            protein FASTA files
PSSM/           PSSM feature files
dssp_rsa_asa/   DSSP/RSA/ASA feature files
labels/         residue-level label arrays
structures/     structure files used by the structure feature pipeline
```

## Configuration

The generic config is:

```text
configs/config.example.json
```

Default template:

```json
{
  "task": "topk",
  "root": "...",
  "id_list": "...",
  "train_list": "...",
  "val_list": "...",
  "test_list": "...",
  "save_dir": "runs/arise_ppi",
  "esm_local_dir": "...",
  "structure_dir": "structures",
  "sequence_mode": "esm",
  "structure_source": "pdb",
  "epochs": 80,
  "batch_size": 4,
  "num_workers": 0
}
```

The placeholder values should be replaced with your local paths. Command-line options override config values.

## Training

Top-k mode:

```bash
python scripts/train.py \
  --config configs/config.example.json \
  --task topk \
  --root /path/to/data/dataset \
  --id-list /path/to/data/all_ids.txt \
  --train-list /path/to/data/train.txt \
  --val-list /path/to/data/val.txt \
  --test-list /path/to/data/test.txt \
  --save-dir runs/topk
```

Binary mode:

```bash
python scripts/train.py \
  --config configs/config.example.json \
  --task binary \
  --root /path/to/data/dataset \
  --id-list /path/to/data/all_ids.txt \
  --train-list /path/to/data/train.txt \
  --val-list /path/to/data/val.txt \
  --test-list /path/to/data/test.txt \
  --save-dir runs/binary
```

Useful overrides:

```bash
python scripts/train.py --task topk --epochs 100 --batch-size 2 --lr 1e-5
python scripts/train.py --task binary --sequence-mode light --allow-zero-esm-fallback 1
```

## Prediction

Top-k prediction:

```bash
python scripts/predict.py \
  --config configs/config.example.json \
  --task topk \
  --split test \
  --save-dir runs/topk \
  --out-dir runs/predict_topk
```

Binary prediction:

```bash
python scripts/predict.py \
  --config configs/config.example.json \
  --task binary \
  --split test \
  --save-dir runs/binary \
  --out-dir runs/predict_binary
```

Use an explicit checkpoint when needed:

```bash
python scripts/predict.py \
  --task topk \
  --checkpoint /path/to/best_TOPK.pt \
  --root /path/to/data/dataset \
  --split test \
  --out-dir runs/predict_topk
```

Prediction split options:

```text
train
val
test
all
/path/to/custom_ids.txt
```

## Outputs

Training outputs are written under `save_dir`, usually:

```text
runs/<run_name>/
  checkpoints/
    best_TOPK.pt
    best_AUPRC.pt
    best_TOPK_metrics.tsv
    best_AUPRC_metrics.tsv
  metrics_history_<timestamp>.tsv
```

Prediction outputs are written under `out_dir`. Depending on task and available labels, outputs may include:

```text
metrics.json
metrics.tsv
topk_metrics.tsv
ranking_top_residues.tsv
residue_scores/
case_study_candidates.json
case_study_candidates.md
```

## Task Selection

Use `--task topk` when the goal is residue prioritization, ranking evidence, or top-k case-study analysis.

Use `--task binary` when the goal is residue-level binding-site classification and thresholded predictions.

Both modes share the same model code and data interface. The task switch controls the primary objective, checkpoint family, metrics, and prediction behavior.

## Reproducibility Notes

- Keep all heavy data under `data/`, `resources/`, or another external path.
- Keep ESM weights outside Git.
- Keep checkpoints and generated runs under `runs/`.
- Use `configs/config.example.json` as the single source of project-level defaults.
- Override dataset-specific paths from the command line or by copying the config file locally.

## Common Commands

Check CLI options:

```bash
python scripts/train.py --help
python scripts/predict.py --help
```

Minimal top-k run:

```bash
python scripts/train.py --config configs/config.example.json --task topk
```

Minimal binary run:

```bash
python scripts/train.py --config configs/config.example.json --task binary
```

Run prediction on a custom ID list:

```bash
python scripts/predict.py \
  --task topk \
  --ids /path/to/custom_ids.txt \
  --checkpoint /path/to/best_TOPK.pt \
  --out-dir runs/custom_prediction
```
