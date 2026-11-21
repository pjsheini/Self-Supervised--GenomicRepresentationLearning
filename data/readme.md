Input data for SSGRL.

- `bat2100k.unmapped.fa` – 100,000 unmapped 150 bp reads (training / eval database)
- `RaTG13.fasta` – bat coronavirus RaTG13 genome (MN996532.1)
- `RmYN01genome_NMDC60013004-01.fasta` – RmYN02 bat coronavirus genome
- `sim_RaTG13.fq` – 500 reads simulated from RaTG13 with
  `python -m ssgrl.simulate data/RaTG13.fasta -n 500 -o data/sim_RaTG13.fq`

`*.npy` files here are caches written by `ssgrl.encoding.load_reads` and are git-ignored.
