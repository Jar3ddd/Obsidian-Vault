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

---

**Plant Version** (without Snakemake)
To start I propose a simple maf to zarr conversion to view the accuracy and plausibility of using the smaller nine species alignment file from [Plant Transcriptional Regulatory Map]([PlantRegMap: Plant Regulation Data and Analysis Platform @ CBI, PKU](https://plantregmap.gao-lab.org/download.php#alignment-conservation)). The current version presented and outlined is without the Snakemake workflow and only uses Python for converting files for GPN-MSA training. 

---
Function convert_maf_to_zarr(maf_file, species_list, output_path): (psuedocode)
1. Create output directory and initialize Zarr store
2. Initialize sequence buffer for each species
3. For each alignment block in the MAF
	1. Parse species sequence mappings
	2. Skip block if reference in missing
	3. Identify non-gap indices in reference
	4. For each species in species_list:
		1. If present, extract valid sequence bases at ungapped positions
		2. Else, use gap filled default
		3. Replace uncertain bases with 'N'
		4. Append to that species' buffer
4. For each species:
	1. Concatenate its sequence chunks into one string
	2. Break into fixed length windows
	3. Convert NumPy array and encode as uint8
	4. Store to Zarr as compressed version
5. Save files
---
**Libraries and dependencies**
==Changed zarr version from 3.0.1 >>> 2.X.X (causing issues when calling packages)===
```
import argparse
import os
from Bio import AlignIO
import numpy as np
import zarr
from numcodecs.blosc import Blosc
from tqdm import tqdm
```
- AlignIO: Reads MAF-format multiple sequence alignments block by block (blocks have varying sizes and species)
- zarr: Stores alarged aligned sequences in compressed and chunked format (allows for random access to individual windows of alignment)
- np: Used for converting character arrays to numeric (uint8) format for ML compatability
- tdqm: Progress bar visual for reading large files (helped diagnosed hanging errors)

**parse_species_list()**
```
def parse_species_list(species_file):
	"""Read species names from a file, one per line."""
	with open(species_file) as f:
		return [line.strip() for line in f if line.strip()]
```
Specifies which species to include in the alignment. This differs from the Snakemake workflow as this order is currently not supported biologically (phylogenetically closest) but rather uses the extracted species from the order presented in the alignment file (get_species_list.py)
- GPN-MSA will expect consistent species ordering for chromosomes

**Setup: (maf_to_zarr)**
```
os.makedirs(zarr_out, exist_ok=True)
root = zarr.open_group(zarr_out, mode='w')
seq_buffers = {sp: [] for sp in species_list}
ref_sp = species_list[0]  # First species is assumed reference
```
- Start with creating the Zarr directory
- Initialize one sequence buffer per species
- Choose the first species as reference (Oryza_sativa_subsp_jap)

Reading the MAF block by block: To start the MAF has alignment blocks, each covering differing regions and also containing varying species and lengths. We will iterate through all blocks to build genome-wide alignment
```
	print(f"Parsing MAF file: {maf_path}")
	with open(maf_path) as handle:
		for block in tqdm(AlignIO.parse(handle, "maf")):
			...
```

Build dictionary of aligned sequences
```
recs = {rec.id.split('.')[0]: str(rec.seq) for rec in block}
ref_seq = recs.get(ref_sp)
if ref_seq is None:
	continue  # skip blocks missing reference
```
- rec.id.split('.'): Extracts species name from the MAF (not nicely named but separates chromosome and region e.g., oryza_sativa_subsp_jap/ ***.chr1.1000***)
- Skip the block if the reference species is missing.

Remove gaps
```
keep_idx = [i for i, c in enumerate(ref_seq) if c != '-']
```
- To keep parallel with the Snakemake and creation from the original GPN-MSA, we remove positions that are not aligned in the reference such that there are no gaps.

Filter and normalize sequences
```
for sp in species_list:
	seq = recs.get(sp, '-' * len(ref_seq))
	filtered = ''.join(
		seq[i] if seq[i] in 'ACGT' else 'N'
		for i in keep_idx
	)
	seq_buffers[sp].append(filtered)
```
- For species not present in an alignment block, we will fill with gaps ('-'). While for aligned species only keep characters aligned to the reference. For uncertain nucleotides we replace with 'N' and keep valid ones such as A, C, T, and G.

Concatenate and Chunk
```
for sp in species_list:
	joined = ''.join(seq_buffers[sp]).upper()
	total_len = len(joined)
	n_windows = total_len // window_size
```
- Combine all per-block sequences into a single genome scale string per species. Additionally, bor the single string, break the sequence into fixed sized non-overlapping windows for expected input dimension (128bp)

Converting to NumPy and Storing
```
arr = np.array(
	[list(joined[i*window_size:(i+1)*window_size]) for i in range(n_windows)],
	dtype='S1'
)
data = arr.view(np.uint8)
```
Start with converting strings into byte arrays (specified by uint8) #uint8
```
ds = root.create_dataset(
	name=sp,
	shape=data.shape,
	dtype=data.dtype,
	chunks=(chunk_size, data.shape[1]),
	compressor=zarr.Blosc(cname='zstd', clevel=5)
)
ds[:] = data
```
- Per species we create one dataset inside the Zarr archive. (Zarr supposedly nice for compression which in turn improves performance and storage. Likewise, Blosc can be changed but suggested from use cases as a start.)

