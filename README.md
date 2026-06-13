# ARISE-PPI

ARISE-PPI provides training and prediction code for residue-level protein interaction analysis. The GitHub-ready entry points are:

- `scripts/train.py`: train with `--task topk` or `--task binary`.
- `scripts/predict.py`: predict with `--task topk` or `--task binary`.

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

## Train

Top-k objective:

```bash
python scripts/train.py --config configs/config.example.json --task topk
```

Binary objective:

```bash
python scripts/train.py --config configs/config.example.json --task binary
```

Any config value can be overridden from the command line:

```bash
python scripts/train.py --task topk --root /path/to/dataset --save-dir runs/my_topk
```

## Predict

Top-k prediction:

```bash
python scripts/predict.py --config configs/config.example.json --task topk --split test --out-dir runs/predict_topk
```

Binary prediction:

```bash
python scripts/predict.py --config configs/config.example.json --task binary --split test --out-dir runs/predict_binary
```

Use `--checkpoint /path/to/best_TOPK.pt` or `--checkpoint /path/to/best_AUPRC.pt` to select an exact checkpoint. Otherwise, `--task topk` selects the top-k checkpoint family and `--task binary` selects the binary/AUPRC checkpoint family.

## Notes

Large datasets, generated runs, model checkpoints, and ESM weights are intentionally excluded by `.gitignore`. Upload only source code, lightweight configs, and documentation to GitHub.
