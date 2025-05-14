### Descriptions of files:
##### **mo_nom** -
==get_redf5_cwms_per_pattern_v1.1R==: ***Extracts Contribution Weight Matricies (CWMs)*** from MoDisco HDF5 output, assigns consistent motifs names, and exports results in a Jaspar format.

This script extracts **motifs** (EPMs/patterns) from deep learning-derived MoDISCo '.hdf5' files using contribution scores, producing weighted ***CWMs***. Each motif is
- Named systematically based on species, model, cluster and strand
- Weighted by seqlet support
- Saved in '.jaspar' format for dowstream analysis (e.g. BLAMM, clustering, importance visualization)
**Inputs**:

**Outputs**:
File: redf4_epmXXXS0_cwm-motifs.jaspar
Format: JASPAR-styel
Content: CWM matricies (scaled, one per motif and strand)

**Conceptual**:
A contribution weight matrix (CWM) captures how important each base (A, C, T, G) at each position is to the models decision. Given a matrix, $$\mathbf{M} \in \mathcal{R}^{4 \times L} $$the scaled motif matrix $$ \mathbf{M}_{ij}^{scaled} = \left| \frac{M_{ij}}{max_{ij} | M_{ij} |} \right| \times \textrm{Seqlet Count} $$
This metric preserves:
- Directionality (fwd/rev)
- Relative base importance
- Support level (number of seqlets)

Outline:
1. Open HDF5 and read metacluster
```
h5file <- H5Fopen(file.path(
  dirpath_in,
  FILE1), "H5F_ACC_RDONLY")
metacluster_group <- h5read(h5file,
                            "metacluster_idx_to_submetacluster_results")
```
2.  Extract CWM Matricies
	1. Loops through both metacluster_0 and metacluster_1:
		1. Forward: task0_contrib_scores/fwd
		2. Reverse: task0_contrib_scores/rev
	2. Save into lists:
		1. matriciesF0, matriciesF1 (forward)
		2. matriciesR0, matriciesR1 (reverse)
3. Assign motif names
4. Normalize and scale motifs
```
for (i in seq_along(motifs)) {
  m0 <- motifs[i]
  m1 <- motifs[[i]]
  name <- names(m0)[1]
  seq_count <- sub(".*_([0-9]+)$", "\\1", name)
  nfcwm <- abs(m1)
  nfcwm <- round(as.numeric(seq_count)*(nfcwm/max(nfcwm)))
  motifs[[i]] <- abs(nfcwm)
}
```
5. Export JASPAR Format

==mo_compare_JASPAR2020_v1.0R==: For the extract motifs (CWMs) , compare with known plant transcription factor motifs in the **JASPAR2020** database and identifies the most similar match.

This script performs **motif similarity analysis** between experimentally discovered motifs (e.g., from CNN saliency scores) and known plant transcription factor binding motifs from the **JAPSAR2020** database. Each discovered motif is:
- Compared to JASPAR2020 motifs using **Smith-Waterman (SW)** similarity
- Match with its **most similar** known TF motif
- Reported along with similarity score, p-value, and E-Value

**Inputs**:

**Outputs**:
File: redf5_XXXS0_cwm-motifs.jaspar_comparison_JASPAR2020.csv
Format: Tab delimited table
Columns:
1. subj: Discovered EPM
2. targ: Matching JASPAR TF name
3. ID: Jaspar motif ID
4. scor: Similiarity score (Smith-Waterman)
5. pval: P-value
6. eval: E-val

All motifs are converted to a Position Weight Matrix (PWM) format using the 'universalmotif' and 'TFBSTools' conversion functions. The Smith-Waterman score (SW) is used to compare motifs positionally by computing the best local alignmnet between two PWMs. Formally, the algorithm is a dynamic programming method that find the highest scoring local matches between two matrices M1 and M2, allowing for mismatches and shifts
**Advantages**
- Captures local similiarites
- Allows partial matches
- Nice for comparing motifs with variable lengths

Outline:
1. Motifs from jaspar_file are stores in 'pwm_uni0'
2. Motifs from JASPAR2020 (plant group only) are stored in the 'jaspar_motifs_plants0'
3. All motifs are combined into a list: 'omni_list'
4. For each extracted motif, run:
```
c<-compare_motifs(cwm1, method = "SW")
```
This compares motif *i* against ever motif in the list (including JASPAR)
5. From the returned scores
	1. Remove self-comparisons (i.e., comparisons with other EPMs)
	2. Keep only the **most similar** JASPAR motif


##### **mo_clu**
==mo_cluster_v2.7.R==: Clusters extracted motifs (CWMs) using similarity metrics (e.g., Smith-Waterman), generated phlogenetic trees, computes motif similarity matrices, and categorizes motifs by uniqueness.

This script identifies **similar motif clusters** among learned expression-predictive (EPMs), performs **hierarchal clustering**, and builds **Newick (NWK)** trees for visualizations. It also detects motifs with and without highly similar counterparts based on similarity quantities.

**Inputs**:

**Outputs**:
- *_summary.txt
- *-Sandelin-Wassermann.nwk
- *-Smith-Waterman.nwk
- *_matrix-SW.csv
- *_epms_with/without_highly_similar_counterparts

Motif similarity is computed using:
- Smith-Waterman (SW) local alignment of position matricies
- Resulting in a matrix of similarity scores S(i, j)
The upper 5% quantile is used as a threshold for identifying "highly similar" motifs.

Outline:
1. Loading data
```
cwm1 <- read_jaspar(File1)
```
	1. Adds number of seqlets (nsites) and total IC + consensus sequence to motif names
	2. Converts to PWM and PCM formats
2. Generate Sequence Logos
	1. For each motif, saves a .png logo using 'ggseqlogo':
```
ggsave(file_path, plot = seq_logo, width = 4831, height = 592, units = "px")
```
3. Summarize Motifs
```
sum<-as.data.frame(summarise_motifs(pcm))
write.csv(sum, file = paste0(dirpath_out,FILE1,"summary.txt"))
```
4. Pariwise Similarity Matrix
```
c<-compare_motifs(cwm1, method = "SW")
```
	1. Generates a similarity matrix S
	2. Uses quantile() to determine 5th and 95th percentiles
5. Subset Motifs by label
	1. Removes motifs labeled with 'string_to_remove1' or 'string_to_remove2' to compare subsets.
6. Identify Highly Similar Motifs
	1. A motif is considered to have a highly similar counterpart if:
$$ S(i, j) > Q_{0.95}$$
The script uses this to classify motifs into:
- epms_with_highly_similar_counterparts
- epms_withtout_highly_similar_counterparts
7. Building clustering trees
	1. Method 1: SW-based matrix -> distance matrix -> hierarchical clustering
	2. Method 2: clusteringMotifs() using Smith-Waterman
8. Visualization Tree
```
```


##### **mo_ran**
==rdf5_get_seql_per_patternV2.1.R==: Extracts genomic location of seqlets (short unprocessed subsequences of patterns) for each motif pattern from the MoDISco HDF5 file output. Final output consist of table with motif assignments, genomic coordinates, and strand information.

The script parses the seqlet-to-pattern assignments in MoDISco HDF5 file and returns a table with:
- The sequence ID from which the seqlet is extracted
- Start and end positions
- Whether the seqlet is a reverse complement
- The metacluster and pattern that the seqlet supports

**Input**:

**Output**:
- rdf5_seqlet_patterXXXX_XXX.txt

Outline:
1. Load HDF5 files from model output
```
h5file <- H5Fopen(file.path(
  dirpath_in,
  FILE1), "H5F_ACC_RDONLY")
h5ls(h5file)
metacluster_group <- h5read(h5file, "metacluster_idx_to_submetacluster_results")
# loop through the metaclusters 0 and 1
for (i in c(0, 1)) {
  metacluster <- metacluster_group[[paste0("metacluster_", i)]]
}
```
2. Iterate through metacluster patterns
```
# From metacluster 0
  seqletls <- metacluster_group[["metacluster_0"]][["seqlets_to_patterns_result"]][["patterns"]][[pattern_name]][["seqlets_and_alnmts"]][["seqlets"]]
# From metacluster 1
  seqletls <- metacluster_group[["metacluster_1"]][["seqlets_to_patterns_result"]][["patterns"]][[pattern_name]][["seqlets_and_alnmts"]][["seqlets"]]
```
3.  Final formatting and table creation
```
seqlet_mc01 <- rbind.data.frame(seqlets_all_mc0, seqlets_all_mc1)

df <- seqlet_mc01 %>%
  mutate(example = NA, start = NA, end = NA, rc = NA) %>%
  separate(col = seqlets, into = c("example", "start", "end", "rc"), sep = "[,]")

df$example <- gsub("example:", "", df$example)
df$start <- gsub("start:", "", df$start)
df$end <- gsub("end:", "", df$end)
df$rc <- gsub("rc:", "", df$rc)
```
It is important to note that a seqlet is not the final or processed motif (EPM/pattern). Rather it is a short DNA region (10-20bp) that activates filters in the trained CNN mode which are extracted and aligned during MoDISco processing. Additionally, these sequences support a pattern within particular metacluster (low and high expression seqlets). In other words, these are the local regions with high important scores and also related to the PWM and CWM construction.
For a particular patter, let $$S = \{s_1, \ldots, s_n\}$$ be the seqlets supporting it. Then for each seqlet $s_i$ there is information referencing genomic coordinates $[ a_i, b_i ]$ with strand flags $r_i \in \{forward, reverse\}$. 

==epm_occurence_ranges_TSS-TTS.16.R==: For evaluating where each motif (EPM) most frequently occurs in the gene space by analyzing the distribution of seqlet positions relative to the TSS and TTS. It summarize these distributions with both summary statistics and visualizations

The script will take the previous motif-seqlet mapping table ('==rdf5_get_seql_per_patternV2.1.R==') and computes:
- Histograms of motif occurrence positions
- Summary statistics (mean, median, mode, IQR, SD, CV, ect.)
- CSV tables for TSS and TTS disributions
- Visualizations for EPM positional frequency

**Input**:

**Output**:
- X_TSS_motif_ranges_q1q9.csv - Summary stats. for upstream region
- X_TTS_motif_ranges_q1q9.csv - Summary stats. for downstream region
- epm_X.png - Histogram plots of seqlet density per motif


##### **mo_proj**
==occ_filter_v1.2JaredsEdit.R==: Filteres mapped motif occurrences (from BLAMM) to retain only those within 1,500bp of a genes's transcription start site (TSS) or transcription termination start (TTS). Afterwards it then annotates each retained motif hit with gene metadata (ID, strand, TSS/TTS distance).

The script connects motif hits (significant mappings from BLAMM) to nearby genes using a GFF annotation file. Specifically, it:
- Filters motifs based on distance to TSS or TTS positions
- Appends gene context
- Outputs a finalized table

==mo_feat-filter_JaredsEdit.R==: After filtering for EPMs within range, we further refine motifs matches based on orientation, motif length, and spatial consistency with expression associated positional ranges. Outputs filtered tabled and BED files for genome visualization.

This script integrates the motif-gene proximity data (from occ_filter) with motif positional statistics (from TSS/TTS range analysis) and gene annotation to:
- Filter and label motifs based on strand, region, and expected genomic positioning
- Apply optional quantile or min/max based filters for biological significance 
- Exports gene-associated motif hits in both csv and BED formats

**Inputs**:

**Outputs**:

Outline:
1. Loading data from motif-gene matices in 'occ_filter' with motif position and gene annotations.
```
occ_filter_output <- "occOry_filter_Root"  # <-- Output from previous script
file1 <- "OryS0_Root-TSS_motif_ranges_q1q9.csv"        # TSS motif range summary
file2 <- "OryS0_Root-TTS_motif_ranges_q1q9.csv"        # TTS motif range summary
file3 <- "Oryza_sativa.IRGSP-1.0.60.gff3" # GFF3 annotation file
```
2. Preprocessing annotations and determine parameters for filtering. If set can take only genes, CDS, transcription related features. Additionally, for features create a loc column for identification
```
weight_region <- "yes"   # yes or no
word_size <- 14           # minimum motif match length
Filter_motif_orient <- "none" # forward, reverse, none
Filter_annot_type   <- "gene" # gene, CDS, mRNA, UTR, none
```
3. For filtering motifs, we retain motifs within 1,500bp of TSS or TTS sites. Moreover, can filter based on orientation and size of motifs with word size specification. Finally, statistical filtering is applied based on quantile or min/max filters
	1. q10, q90: 10th and 90th percentile of observed motif position (range based filtering)
	2. min, max: Absolue range of motif position
	3. dist_transc_border: Min of distance from TSS or TTS
```
## Applying min-max filter
upstream_mima <- upstream_merged %>%
  filter(dist_transc_border >= min & dist_transc_border <= max)
downstream_mima <- downstream_merged %>%
  filter(dist_transc_border >= min & dist_transc_border <= max)

## Applying quantile filters
upstream_q1q9 <- upstream_merged %>%
  filter(dist_transc_border >= q10 & dist_transc_border <= q90)
downstream_q1q9 <- downstream_merged %>%
  filter(dist_transc_border <= q10 & dist_transc_border <= q90)
```
Final BED file are exported to may be used in Genomic visualization tracks (USCS, Ensembl, ect.)
