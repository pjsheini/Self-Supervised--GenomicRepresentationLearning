Retrieval of 1305 unique held-out reads from noisy copies.

| Method | recall@1 | recall@5 | recall@10 | subs only | RC only | crop only | time |
|---|---|---|---|---|---|---|---|
| canonical 6-mer profile | 0.730 | 0.852 | 0.880 | 0.935 | 0.997 | 0.706 | 0.4s |
| minimap2 (preset sr) | 0.726 | 0.805 | 0.813 | 0.868 | 0.971 | 0.717 | 0.2s |
| Mash (k=21, s=1000) | 0.695 | 0.836 | 0.874 | 0.835 | 1.000 | 0.723 | 5.2s |
| canonical 4-mer profile | 0.663 | 0.808 | 0.844 | 0.878 | 0.995 | 0.659 | 0.1s |
| SSGRL (this repo) | 0.569 | 0.753 | 0.799 | 0.752 | 1.000 | 0.597 | 3.3s |
| 4-mer profile (l1) / TNF | 0.373 | 0.465 | 0.507 | 0.918 | 0.031 | 0.682 | 0.0s |
| 4-mer counts (cosine) | 0.373 | 0.464 | 0.507 | 0.918 | 0.031 | 0.684 | 0.0s |
| 3-mer profile (l1) | 0.344 | 0.448 | 0.477 | 0.870 | 0.033 | 0.634 | 0.1s |
