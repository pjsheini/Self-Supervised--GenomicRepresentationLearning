Read binning: 1196 reads from 6 genomes, DNABERT-S protocol (threshold from a labelled split, greedy KMedoid, Hungarian matching).

| Method | bins with F1>0.9 | F1>0.7 | F1>0.5 | bins found | ARI | reads binned | time |
|---|---|---|---|---|---|---|---|
| 3-mer profile (l1) | 1 | 1 | 2 | 8 | 0.326 | 0.97 | 0.9s |
| 4-mer profile (l1) / TNF | 1 | 1 | 3 | 9 | 0.376 | 0.979 | 0.9s |
| canonical 4-mer profile | 1 | 3 | 6 | 6 | 0.564 | 0.977 | 0.8s |
| SSGRL (this repo) | 1 | 1 | 3 | 5 | 0.383 | 1.0 | 78.4s |
| canonical 6-mer profile | 0 | 2 | 6 | 10 | 0.449 | 0.972 | 1.4s |
| canonical 6-mer, 150bp tiles | 0 | 2 | 5 | 10 | 0.433 | 0.969 | 28.5s |
