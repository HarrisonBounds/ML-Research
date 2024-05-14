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

### Using CTGAN 
### Train with Synthetic, Test with Real

__Spyware__
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
| accuracy             |           |        |   0.36   |  2004   |
| macro avg            |   0.47    |  0.35  |   0.31   |  2004   |
| weighted avg         |   0.44    |  0.36  |   0.30   |  2004   |


__Trojan__
- Overall Quality Score: 69.97%
- Column Shapes: 84.34%
- Column Pair Trends: 55.6%

|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| Trojan-Emotet |   0.28    |  0.19  |   0.23   |   372   |
| Trojan-Reconyc|   0.25    |  0.22  |   0.23   |   303   |
| Trojan-Refroso|   0.30    |  0.80  |   0.43   |   391   |
| Trojan-Scar   |   0.26    |  0.10  |   0.15   |   405   |
| Trojan-Zeus   |   0.38    |  0.14  |   0.20   |   427   |
| accuracy      |           |        |   0.29   |  1898   |
| macro avg     |   0.29    |  0.29  |   0.25   |  1898   |
| weighted avg  |   0.30    |  0.29  |   0.25   |  1898   |

__Ransomware__
- Overall Quality Score: 69.05%
- Column Shapes: 81.8%
- Column Pair Trends: 56.36%

|                 | precision | recall | f1-score | support |
|-----------------|-----------|--------|----------|---------|
| Ransomware-Ako  |    0.45   |  0.29  |   0.35   |   379   |
| Ransomware-Conti|    0.13   |  0.03  |   0.04   |   398   |
| Ransomware-Maze |    0.32   |  0.03  |   0.05   |   402   |
| Ransomware-Pysa |    0.37   |  0.03  |   0.06   |   338   |
| Ransomware-Shade|    0.25   |  0.90  |   0.39   |   442   |
| accuracy        |           |        |   0.27   |  1959   |
| macro avg       |    0.30   |  0.25  |   0.18   |  1959   |
| weighted avg    |    0.30   |  0.27  |   0.19   |  1959   |

### Train with Synthetic + Real, Test with Real

__Spyware__

|                     | precision | recall | f1-score | support |
|---------------------|-----------|--------|----------|---------|
| Spyware-180solutions|   0.67    |  0.56  |   0.61   |   399   |
|      Spyware-CWS    |   0.51    |  0.55  |   0.53   |   375   |
|    Spyware-Gator    |   0.69    |  0.86  |   0.77   |   467   |
|     Spyware-TIBS    |   0.85    |  0.76  |   0.80   |   270   |
| Spyware-Transponder |   0.57    |  0.51  |   0.54   |   493   |
|      accuracy       |           |        |   0.64   |   2004  |
|      macro avg      |   0.66    |  0.65  |   0.65   |   2004  |
|   weighted avg      |   0.64    |  0.64  |   0.64   |   2004  |

__Trojan__








