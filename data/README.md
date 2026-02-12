# Dataset

## Source

This project uses accelerometer data from body-worn sensors (back and thigh placement) for human activity recognition.

**Note:** Raw data files are not included in this repository due to size constraints.

## Download Instructions

1. Obtain the dataset from your course materials or the original data source
2. Place training CSV files in `raw/`
3. Place test CSV files in `raw/test/`

## Expected Format

Each CSV file should contain the following columns:

| Column | Description |
|--------|------------|
| `timestamp` | Recording timestamp |
| `back_x` | Back sensor X-axis acceleration (g) |
| `back_y` | Back sensor Y-axis acceleration (g) |
| `back_z` | Back sensor Z-axis acceleration (g) |
| `thigh_x` | Thigh sensor X-axis acceleration (g) |
| `thigh_y` | Thigh sensor Y-axis acceleration (g) |
| `thigh_z` | Thigh sensor Z-axis acceleration (g) |
| `label` | Activity label (integer) |

## Activity Labels

| Label | Activity |
|------:|---------|
| 1 | Walking |
| 2 | Running |
| 3 | Shuffling |
| 4 | Stairs (Ascending) → merged to 9 |
| 5 | Stairs (Descending) → merged to 9 |
| 6 | Standing |
| 7 | Sitting |
| 8 | Lying Down |
| 13, 14, 130, 140 | Cycling variants → dropped |

## Data Size

- Training: ~5.6M rows
- Test: ~1.1M rows
- After preprocessing: ~123K feature windows
