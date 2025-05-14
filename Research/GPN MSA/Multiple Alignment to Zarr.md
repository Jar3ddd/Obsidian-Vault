**Overview:**
For use of the GPN-MSA model for our prediction, we will first need to convert multiple alignments to a readable zarr format. To convert this file type, below is an annotated original copy for reference and the outline for the new Oryza Sativa Jap. alignments. 

---

**Annotated Original Copy**: Utilizes Snakemake workflow by defining rules to build outputs from inputs provided. #Snakemake #Snakemakenotes

**Defining global constants**
```
from gpn.data import load_fasta
import pandas as pd
import numpy as np

CHROMS = [str(i) for i in range(1, 20)] + ['X', 'Y']
SPECIES_RENAMING = {
    "GCF_003668045v3": "GCF_003668045",
}
CHUNK_SIZE = 512
SHARD_SIZE = 1_048_576
```
- SPECIES_RENAMING: Rules for naming specie IDs in the phlogenetic tree
- CHUNK_SIZE/SHARD_SIZE: Controls how much data is read and written (direct influence of training performance)

**Rules for Snakemake**
```
rule all:
	input:
		"results/msa/mm39_multiz35way.zarr",
```

**Rule 2: download_maf**
```
rule download_maf:
    output:
        "results/maf/{ref}_{msa}/{chrom}.maf",
    shell:
        "wget -O - https://hgdownload.soe.ucsc.edu/goldenPath/{wildcards.ref}/{wildcards.msa}/maf/chr{wildcards.chrom}.maf.gz | gunzip -c > {output}"
```
- Downloads and decompresses .maf alignment for a specific chromosome
- {ref} is the reference genome (e.g. mm39 for mouse) and {msa} is the alignemnt set (e.g., multiz35way)

**Rule 3: download_genome_chr**
```
rule download_genome_chr:
    output:
        temp("results/genome_chr/{ref}.fa"),
    shell:
        "wget -O - https://hgdownload.soe.ucsc.edu/goldenPath/{wildcards.ref}/bigZips/{wildcards.ref}.fa.gz | gunzip -c > {output}"
```
- Downloads the entire genome FASTA for the reference species
- Marked as temp to delete after use in the pipline

**Rule 4: extract_chrom
```
rule extract_chrom:
    input:
        "results/genome_chr/{ref}.fa",
    output:
        "results/genome_chr_chrom/{ref}/{chrom}.fa",
    shell:
        "faOneRecord {input} chr{wildcards.chrom} > {output}"
```
- Extracts a single chromosome's sequence from the reference FASTA using UCSC's faOneRecord

**Rule 5: maf2fasta**
```
rule maf2fasta:
    input:
        "results/genome_chr_chrom/{ref}/{chrom}.fa",
        "results/maf/{ref}_{msa}/{chrom}.maf",
    output:
        "results/maf_fasta/{ref}_{msa}/{chrom}.fa",
    shell:
        "maf2fasta {input} fasta > {output}"
```
- Converts .maf multiple alignments into aligned FASTA format (for all species in the alignment)

**Rule 6: download_tree**
```
rule download_tree:
    output:
        "results/tree/{ref}_{msa}.nh",
    params:
        lambda wildcards: wildcards.msa.replace("multiz", "")
    shell:
        "wget -O {output} https://hgdownload.soe.ucsc.edu/goldenPath/{wildcards.ref}/{wildcards.msa}/{wildcards.ref}.{params}.nh"
```
- Downloads the Newick-format tree used to compute species distance (important for alignment ordering and weighting)

**Rule 7: extract_species**
```
rule extract_species:
    input:
        "results/tree/{ref}_{msa}.nh",
    output:
        "results/species/{ref}_{msa}.txt",
    run:
        from Bio import Phylo
        tree = Phylo.read(input[0], "newick")
        species = pd.DataFrame(index=[node.name for node in tree.get_terminals()])
        species["dist"] = species.index.map(lambda s: tree.distance(wildcards.ref, s))
        species = species.sort_values("dist")
        species.index = species.index.map(lambda s: SPECIES_RENAMING.get(s, s))
        species.to_csv(output[0], header=False, columns=[])
```
- Parses previous tree and ranks species by phylogenetic distance to the reference
- Applies renaming by trimming version tags
- Outputs an ordered list of species

**Rule 8: make_msa_chrom**
```
rule make_msa_chrom:
    input:
        "results/maf_fasta/{msa}/{chrom}.fa",
        "results/species/{msa}.txt",
    output:
        #temp("results/msa_chrom/{msa}/{chrom}.npy"),
        "results/msa_chrom/{msa}/{chrom}.npy",
    run:
        MSA = load_fasta(input[0])
        # the ref should be in first position
        species = pd.read_csv(input[1], header=None).values.ravel()
        MSA = MSA[species]
        MSA = np.vstack(MSA.apply(
            lambda seq: np.frombuffer(seq.upper().encode("ascii"), dtype="S1")
        ))
        # let's only keep non-gaps in reference
        MSA = MSA[:, MSA[0]!=b'-']
        MSA = MSA.T
        vocab = np.frombuffer("ACGT-".encode('ascii'), dtype="S1")
        # decision: consider all "N" and similar as "-"
        # might not be the best, some aligners have a distinction
        # between N, or unaligned, and gap
        MSA[~np.isin(MSA, vocab)] = b"-"
        np.save(output[0], MSA)
```
- Converts aligned FASTA to a NumPy array
- Retains only non-gap columns in the reference
- Ensures on valid characters (A, C, G, T, -) are used, where unknowns replaced with "-"
- Saves array as Numpy object

**Rule 9: merge_msa_chroms**
```
rule merge_msa_chroms:
    input:
        expand("results/msa_chrom/{{msa}}/{chrom}.npy", chrom=CHROMS),
    output:
        directory("results/msa/{msa}.zarr"),
    threads: workflow.cores
    run:
        import zarr

        z = zarr.open_group(output[0], mode='w')
        for chrom, path in zip(CHROMS, input):
            print(chrom)
            data = np.load(path).view(np.uint8)
            z_chrom = z.create_array(
                name=chrom, shape=data.shape, dtype=data.dtype,
                chunks=(CHUNK_SIZE, data.shape[1]),
                shards=(SHARD_SIZE, data.shape[1]),
            )
            z_chrom[:] = data
```
- Final merging: Loads all NumPy alignment arrays and writes them to a single Zarr group
- Each chromosome is stored as a seperate compressed array within the .zarr store
- Uses .view(np.uint8) to ensure compact storage and compatibility with model input

