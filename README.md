# ARISE-PPI

ARISE-PPI provides training and prediction code for residue-level protein interaction analysis. The GitHub-ready entry points are:

- `scripts/train.py`: train with either `--task topk` or `--task binary`.
- `scripts/predict.py`: predict/export results with either `--task topk` or `--task binary`.

## Install

```bash
pip install -r requirements.txt
```

For editable development:

```bash
pip install -e .
```

## Data Layout

Keep large datasets outside Git, or place them under ignored folders such as `data/`:

```text
data/
  RBP400/
    seq/
    PSSM/
    dssp_rsa_asa/
    labels/
    structures/
  RBP400_full_accessions.txt
  RBP400_split_train.txt
  RBP400_split_val.txt
  RBP400_split_test.txt
resources/
  esm/
    esm2_t33_650M_UR50D.pt
```

## Train

Top-k objective:

```bash
python scripts/train.py --config configs/rbp400_topk.example.json
```

Binary objective:

```bash
python scripts/train.py --config configs/rbp400_binary.example.json
```

Any config value can be overridden from the command line:

```bash
python scripts/train.py --task topk --root /path/to/RBP400 --save-dir runs/my_topk
```

## Predict

Top-k prediction:

```bash
python scripts/predict.py --config configs/rbp400_topk.example.json --split test --out-dir runs/predict_topk
```

Binary prediction:

```bash
python scripts/predict.py --config configs/rbp400_binary.example.json --split test --out-dir runs/predict_binary
```

Use `--checkpoint /path/to/best_TOPK.pt` or `--checkpoint /path/to/best_AUPRC.pt` to select an exact checkpoint. Otherwise, `--task topk` selects the top-k checkpoint family and `--task binary` selects the binary/AUPRC checkpoint family.

## Notes

Large datasets, generated runs, model checkpoints, and ESM weights are intentionally excluded by `.gitignore`. Upload only source code, lightweight configs, and documentation to GitHub.
