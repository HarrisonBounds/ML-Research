# ML-Research - Malware Classification with Generative Adversarial Networks

## Baseline Classification

### Trojan

|                | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| Trojan-Emotet  | 0.69      | 0.68   | 0.69     | 372     |
| Trojan-Reconyc | 0.69      | 0.69   | 0.69     | 303     |
| Trojan-Refroso | 0.77      | 0.81   | 0.79     | 391     |
| Trojan-Scar    | 0.68      | 0.68   | 0.68     | 405     |
| Trojan-Zeus    | 0.81      | 0.78   | 0.79     | 427     |
| accuracy       |           |        | 0.73     | 1898    |
| macro avg      | 0.73      | 0.73   | 0.73     | 1898    |
| weighted avg   | 0.73      | 0.73   | 0.73     | 1898    |

### Ransomware

|                 | precision | recall | f1-score | support |
|-----------------|-----------|--------|----------|---------|
| Ransomware-Ako  | 0.49      | 0.54   | 0.51     | 379     |
| Ransomware-Conti| 0.56      | 0.59   | 0.57     | 398     |
| Ransomware-Maze | 0.62      | 0.54   | 0.58     | 402     |
| Ransomware-Pysa | 0.61      | 0.51   | 0.56     | 338     |
| Ransomware-Shade| 0.62      | 0.68   | 0.65     | 442     |
| accuracy        |           |        | 0.58     | 1959    |
| macro avg       | 0.58      | 0.57   | 0.57     | 1959    |
| weighted avg    | 0.58      | 0.58   | 0.58     | 1959    |

### Spyware

|                      | precision | recall | f1-score | support |
|----------------------|-----------|--------|----------|---------|
| Spyware-180solutions | 0.70      | 0.61   | 0.65     | 399     |
| Spyware-CWS          | 0.52      | 0.56   | 0.54     | 375     |
| Spyware-Gator        | 0.73      | 0.85   | 0.78     | 467     |
| Spyware-TIBS         | 0.87      | 0.79   | 0.83     | 270     |
| Spyware-Transponder  | 0.57      | 0.52   | 0.54     | 493     |
| accuracy             |           |        | 0.66     | 2004    |
| macro avg            | 0.68      | 0.67   | 0.67     | 2004    |
| weighted avg         | 0.66      | 0.66   | 0.66     | 2004    |

## GAN Classification

### CTGAN
- Overall Quality Score: 69.91%
- Column Shapes: 84.15%
- Column Pair Trends: 55.66%



|                  | precision | recall | f1-score | support |
|------------------|-----------|--------|----------|---------|
| Spyware-180solutions |   0.36    |  0.05  |   0.09   |   399   |
| Spyware-CWS          |   0.41    |  0.19  |   0.26   |   375   |
| Spyware-Gator        |   0.31    |  0.99  |   0.47   |   467   |
| Spyware-TIBS         |   0.82    |  0.38  |   0.52   |   270   |
| Spyware-Transponder  |   0.46    |  0.14  |   0.22   |   493   |
| accuracy             |           |        |          |  0.36   |
| macro avg            |   0.47    |  0.35  |   0.31   |  2004   |
| weighted avg         |   0.44    |  0.36  |   0.30   |  2004   |


