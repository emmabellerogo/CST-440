# Int8 Evaluation Report

- int8_val_acc: 0.25817
- int8_mean_conf: 0.25393
- macro_f1: 0.10260
- weighted_f1: 0.10595

## Per-Class Metrics

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| like | 0.2582 | 1.0000 | 0.4104 | 158 |
| dislike | 0.0000 | 0.0000 | 0.0000 | 166 |
| peace | 0.0000 | 0.0000 | 0.0000 | 147 |
| background | 0.0000 | 0.0000 | 0.0000 | 141 |

## Prediction Distribution

- like: 612
- dislike: 0
- peace: 0
- background: 0

## Confusion Matrix (Counts)

Rows=true, cols=pred
```
[[158   0   0   0]
 [166   0   0   0]
 [147   0   0   0]
 [141   0   0   0]]
```

## Confusion Matrix (Row-Normalized)

Rows sum to 1.0
```
[[1.000 0.000 0.000 0.000]
 [1.000 0.000 0.000 0.000]
 [1.000 0.000 0.000 0.000]
 [1.000 0.000 0.000 0.000]]
```
