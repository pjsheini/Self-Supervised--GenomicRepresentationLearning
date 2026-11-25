Read binning: 3997 reads from 6 genomes, DNABERT-S protocol (threshold from a labelled split, greedy KMedoid, Hungarian matching).

| Method | bins with F1>0.9 | F1>0.7 | F1>0.5 | bins found | ARI | reads binned | time |
|---|---|---|---|---|---|---|---|
| 3-mer profile (l1) | 0 | 0 | 1 | 12 | 0.058 | 0.984 | 0.2s |
| 4-mer profile (l1) / TNF | 0 | 0 | 1 | 13 | 0.093 | 0.987 | 0.2s |
| canonical 4-mer profile | 0 | 0 | 1 | 11 | 0.109 | 0.982 | 0.2s |
| canonical 6-mer profile | 0 | 0 | 1 | 17 | 0.097 | 0.994 | 2.5s |
| SSGRL (this repo) | 0 | 0 | 0 | 8 | 0.082 | 0.995 | 9.8s |
| canonical 6-mer, 150bp tiles | 0 | 0 | 1 | 17 | 0.097 | 0.994 | 2.5s |
