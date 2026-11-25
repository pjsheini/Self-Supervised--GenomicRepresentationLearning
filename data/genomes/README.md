Reference genomes for the synthetic metagenome used by the read-binning
benchmark, taken from [nf-core/test-datasets](https://github.com/nf-core/test-datasets)
(`modules` branch, `data/genomics/`). They stand in for CAMI2, which is not
reachable from every environment.

| File | Source path in nf-core/test-datasets | Size |
|---|---|---|
| bacteroides_fragilis.fa | prokaryotes/bacteroides_fragilis/genome/genome.fna.gz | 5.2 Mbp |
| candidatus_portiera_aleyrodidarum.fa | prokaryotes/candidatus_portiera_aleyrodidarum/genome/genome.fasta | 358 kbp |
| escherichia_coli.fa | prokaryotes/escherichia_coli/genome/genome.fa.gz | 48 kbp |
| haemophilus_influenzae.fa | prokaryotes/haemophilus_influenzae/genome/genome.fna.gz | 1.9 Mbp |
| saccharomyces_cerevisiae.fa | eukaryotes/saccharomyces_cerevisiae/genome/genome.fa | 230 kbp |
| sarscov2.fa | sarscov2/genome/genome.fasta | 30 kbp |

Rebuild the metagenome with:

    python -m ssgrl.simulate_metagenome data/genomes/*.fa -o data/metagenome_sim -n 1200 -l 5000
