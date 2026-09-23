# Final evaluation summary

Dropout 0.3, three model seeds (42/43/44), test partition. Mean ± std.

## Table 9. Binary detection (gate_bin + classifier head)

n = 1379 (415 malware). Threshold 0.5. All-benign baseline accuracy = 0.6991.

| Metric | Value |
|---|---|
| Accuracy | 0.8542 ± 0.0402 |
| Precision (malware) | 0.7451 ± 0.1094 |
| Recall (malware) | 0.8249 ± 0.0442 |
| F1 (malware) | 0.7765 ± 0.0470 |
| ROC-AUC | 0.9222 ± 0.0278 |

## Table 10. Behaviour-level retrieval (relevant = shares ≥1 tag)

Queries: tagged malware in test. Experts evaluated on covered queries only.

| k | Metric | SAFE | MalConv | Node2Vec | MoE |
|---|---|---|---|---|---|
| 1 | Precision | 0.8058 | 0.7557 | 0.6631 | **0.7513 ± 0.0126** |
|  | nDCG | 0.8058 | 0.7557 | 0.6631 | **0.7513 ± 0.0126** |
| 3 | Precision | 0.7546 | 0.6836 | 0.6762 | **0.7090 ± 0.0028** |
|  | nDCG | 0.7669 | 0.6992 | 0.6739 | **0.7186 ± 0.0037** |
| 5 | Precision | 0.7212 | 0.6361 | 0.6602 | **0.6863 ± 0.0108** |
|  | nDCG | 0.7402 | 0.6624 | 0.6632 | **0.7004 ± 0.0061** |
| 10 | Precision | 0.6797 | 0.5819 | 0.6391 | **0.6510 ± 0.0229** |
|  | nDCG | 0.7050 | 0.6155 | 0.6471 | **0.6711 ± 0.0162** |

Query counts: SAFE 345, MalConv 393, Node2Vec 279, MoE 394

## Table 10b. Matched comparison — MoE on each expert's own queries

Same query set for both columns in each row. Expert = single run; MoE = mean ± std.

| Subset | n | k | Expert P | MoE P | Expert nDCG | MoE nDCG |
|---|---|---|---|---|---|---|
| SAFE | 345 | 1 | 0.8058 | 0.7604 ± 0.0096 | 0.8058 | 0.7604 ± 0.0096 |
| SAFE | 345 | 10 | 0.6797 | 0.6567 ± 0.0312 | 0.7050 | 0.6781 ± 0.0237 |
| MalConv | 393 | 1 | 0.7557 | 0.7523 ± 0.0132 | 0.7557 | 0.7523 ± 0.0132 |
| MalConv | 393 | 10 | 0.5819 | 0.6518 ± 0.0221 | 0.6155 | 0.6720 ± 0.0153 |
| Node2Vec | 279 | 1 | 0.6631 | 0.7611 ± 0.0132 | 0.6631 | 0.7611 ± 0.0132 |
| Node2Vec | 279 | 10 | 0.6391 | 0.6795 ± 0.0471 | 0.6471 | 0.6970 ± 0.0379 |

## Table 11. Mean gate weights by availability pattern (seed 42)

Pattern = (MalConv, SAFE, Node2Vec) availability.

| Pattern | n | gate_bin [Mal, SAFE, N2V] | gate_retr [Mal, SAFE, N2V] |
|---|---|---|---|
| 111 | 372 | 0.000, 0.126, 0.874 | 0.000, 0.644, 0.356 |
| 110 | 149 | 0.339, 0.661, 0.000 | 0.000, 1.000, 0.000 |
| 101 | 348 | 0.000, 0.000, 1.000 | 0.454, 0.000, 0.546 |
| 100 | 509 | 1.000, 0.000, 0.000 | 1.000, 0.000, 0.000 |
| 000 | 1 | 0.000, 0.000, 0.000 | 0.000, 0.000, 0.000 |

## Figure 5(c) data. Worked example, SAFE missing (seed 42)

| | MalConv | SAFE | Node2Vec |
|---|---|---|---|
| mask | 1.000 | 0.000 | 1.000 |
| gate_bin raw | 0.000 | 0.000 | 1.000 |
| gate_bin renormalised | 0.000 | 0.000 | 1.000 |
| gate_retr raw | 0.000 | 0.500 | 0.500 |
| gate_retr renormalised | 0.000 | 0.000 | 1.000 |
