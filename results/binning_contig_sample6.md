Read binning: 1209 reads from 6 genomes, DNABERT-S protocol (threshold from a labelled split, greedy KMedoid, Hungarian matching).

| Method | bins with F1>0.9 | F1>0.7 | F1>0.5 | bins found | ARI | reads binned | time |
|---|---|---|---|---|---|---|---|
| 3-mer profile (l1) | 1 | 2 | 4 | 7 | 0.226 | 0.983 | 0.9s |
| 4-mer profile (l1) / TNF | 1 | 2 | 4 | 8 | 0.249 | 0.978 | 1.0s |
| canonical 4-mer profile | 1 | 4 | 6 | 9 | 0.454 | 0.998 | 0.8s |
| canonical 6-mer profile | 1 | 4 | 5 | 10 | 0.476 | 0.97 | 1.3s |
| SSGRL (this repo) | 1 | 2 | 3 | 4 | 0.244 | 0.997 | 84.4s |
| canonical 6-mer, 150bp tiles | 1 | 4 | 5 | 10 | 0.472 | 0.974 | 26.5s |
