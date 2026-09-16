# Data

The full corpus is not distributed with this repository (source publications are not
freely redistributable). Place your own Excel file here, e.g. `data/dati_Pico.xlsx`,
with exactly these two columns:

| Column | Type | Description |
|---|---|---|
| `Text` | string | the sentence extracted from a structured abstract |
| `Category` | int (0/1) | binary target for the PICO element being modelled |

`sample_data.xlsx` in this folder is a small synthetic file with the same schema, so the
three pipelines in `src/` can be run end to end without the real corpus, e.g.:

```bash
python src/train_classification.py --data data/sample_data.xlsx --seed 42
```
