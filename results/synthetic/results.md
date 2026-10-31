## Results (k = 5, hit rule: centre)

| run | frames | persons | recall@5 | precision@5 | false alarms / frame | FA / frame (excl. other objects) | steady-state recall@5 | steady-state FA / frame | fps |
|---|---|---|---|---|---|---|---|---|---|
| full | 360 | 739 | 92.4% | 51.7% | 1.77 | 1.77 | 99.3% | 1.90 | 2.67 |
| no_temporal | 360 | 739 | 89.2% | 40.2% | 2.73 | 2.73 | 88.7% | 2.80 | 2.64 |
| no_glint | 360 | 739 | 92.4% | 51.2% | 1.81 | 1.81 | 99.3% | 1.94 | 2.75 |
| no_morphology | 360 | 739 | 92.4% | 45.7% | 2.50 | 2.50 | 99.3% | 2.68 | 2.65 |

Steady-state numbers leave out the first min_persistence - 1 frames of each sequence, during which the temporal stage cannot yet have promoted a track.

### Recall@5 by target size (sqrt of box area)

| run | <16 px (n=618) | 16-32 px (n=121) |
|---|---|---|
| full | 92.2% | 93.4% |
| no_temporal | 87.2% | 99.2% |
| no_glint | 92.2% | 93.4% |
| no_morphology | 92.2% | 93.4% |

### Recall@5 by altitude

| run | <60 m (n=117) | 60-100 m (n=393) | 100-140 m (n=109) | >=140 m (n=120) |
|---|---|---|---|---|
| full | 93.2% | 92.9% | 89.0% | 93.3% |
| no_temporal | 99.1% | 90.6% | 72.5% | 90.0% |
| no_glint | 93.2% | 92.9% | 89.0% | 93.3% |
| no_morphology | 93.2% | 92.9% | 89.0% | 93.3% |

### Recall vs shortlist length

| run | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 45.3% | 85.7% | 92.3% | 92.4% | 92.4% | 92.4% | 92.4% | 92.4% | 92.4% | 92.4% |
| no_temporal | 29.0% | 57.1% | 72.4% | 82.7% | 89.2% | 91.3% | 94.6% | 95.8% | 96.6% | 97.0% |
| no_glint | 45.3% | 85.7% | 92.3% | 92.4% | 92.4% | 92.4% | 92.4% | 92.4% | 92.4% | 92.4% |
| no_morphology | 45.3% | 85.3% | 92.3% | 92.4% | 92.4% | 92.4% | 92.4% | 92.4% | 92.4% | 92.4% |
