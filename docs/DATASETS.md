# Datasets

All datasets are from the **ETT-small** collection published by [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) and used in the original FAN paper (NeurIPS 2024).

---

## Available Datasets

| File         | Type    | Resolution | Rows   | Features | Period          |
|--------------|---------|------------|--------|----------|-----------------|
| `ETTh1.csv`  | Hourly  | 1h         | 17,420 | 7        | Jul 2016–Jul 2018 |
| `ETTh2.csv`  | Hourly  | 1h         | 17,420 | 7        | Jul 2016–Jul 2018 |
| `ETTm1.csv`  | Minute  | 15min      | 69,680 | 7        | Jul 2016–Jul 2018 |
| `ETTm2.csv`  | Minute  | 15min      | 69,680 | 7        | Jul 2016–Jul 2018 |

## Features (columns)

| Column | Description                         |
|--------|-------------------------------------|
| date   | Timestamp                           |
| HUFL   | High Useful Load                    |
| HULL   | High Useless Load                   |
| MUFL   | Middle Useful Load                  |
| MULL   | Middle Useless Load                 |
| LUFL   | Low Useful Load                     |
| LULL   | Low Useless Load                    |
| OT     | Oil Temperature (target variable)   |

---

## Data Splitting

We follow the **official FAN paper** split:

| Split      | Fraction | Note                                      |
|------------|----------|-------------------------------------------|
| Train      | 70%      | Scaler fitted here only (no data leakage) |
| Validation | 20%      | Used for early stopping / monitoring      |
| Test       | 10%      | Final evaluation only                     |

> The official FAN repo (`wayne155/FAN`) acknowledges that their default implementation scales using **all** the data (train + val + test), which introduces a minor label leak. Our implementation uses `scale_in_train=True` behaviour — the scaler is fitted **strictly on the training split** to ensure clean evaluation.

---

## Download

The datasets are downloaded automatically by running:

```bash
# PowerShell (from project root)
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv" -OutFile "data/ETTh1.csv"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv" -OutFile "data/ETTh2.csv"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv" -OutFile "data/ETTm1.csv"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv" -OutFile "data/ETTm2.csv"
```

All 4 files should be placed in the `data/` directory.
