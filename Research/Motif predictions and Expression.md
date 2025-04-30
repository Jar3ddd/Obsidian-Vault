# Predicting cis-Regulatory Code within Rice using Convolution Neural Networks

## Authors  
- **Jared Richardson**    
  *Math, University of Texas - Arlington*    
  Email: jared.richardson@mavs.uta.edu  

- **Dr. Jeremy Edwards**    
  *DBNRRC, USDA - ARS*    
  Email: jeremy.edwards@usda.gov  

- **Dr. Jianzong Su**    
  *Math, University of Texas - Arlington*    
  Email: su@uta.edu  

---

## Abstract  
With the increasing availability of genomic data and advancements in deep learning, we aim to explore the relationship between genotype motifs and gene expression regulation in rice (Oryza sativa). Our approach involves developing convolutional neural network (CNN) models to predict gene expression levels based on DNA sequence information from gene flanking regions, which play a critical role in gene transcription regulation. Using both single-species (Oryza sativa japonica) and multi-species models, we identify key regulatory motifs that influence gene expression. These identified motifs will be analyzed across rice variants in the Germplasm Collection and used for allele mining to identify superior haplotypes  to uncover potential associations with agronomically important traits. This model will also be employed for the enhanced genomic selection methods incorporating gene expression predictions to accelerate rice breeding.

---

## Introduction

### Transcription Factors and cis-Regulatory Elements  
The process of gene expression allows for the cellular products to be made, inevitably influencing an organism's phenotype. The first step in this complex process is transcription. Transcription involves the action of turning DNA into readable mRNA, of which will be transported and translated into amino acids making proteins. 
	
These proteins can serve many functions within a cell and can have duties ranging from transportation, structural, and altering chemical processes. Additionally, of these functions there is another more specially known as transcription factors. These proteins bind to specific DNA sequences, or motifs, and directly affect the turnover of mRNA from a gene. These motifs, known as transcription factor binding sites (TFBS) reside in the non-coding portion of DNA and typically are located proximal to the gene they influence. Depending on the type of proteins that bind to these motifs, several actions can be executed that can promote, silence, repress, and many more, all of which influence the rate of transcription.

With the Dale Bumpers National Rice Research Center's (DBNRRC) extensive germplasm and gene bank, the discovery and analysis of these motifs will be useful in elucidating the link between these novel motifs and target genes. Understanding the regulatory elements and transcription factors controlling gene expression in rice is vital for improving traits such as yield, drought tolerance, and disease resistance. Rice, as a staple food for over half the world’s population, faces increasing demands amid global challenges such as climate change. Identifying transcription factor binding sites and cis-regulatory elements across the rice genome, particularly in the germplasm collection, will provide insight into how gene expression is regulated in response to environmental stimuli and developmental processes.

The purpose of this research is to uncover novel regulatory motifs that can serve as targets for crop improvement. By predicting gene expression from DNA sequences using deep learning models like convolutional neural networks (CNNs), we aim to link specific motifs with gene expression profiles and phenotypic traits. This approach, when applied to rice germplasm, will help identify new alleles that could be beneficial for breeding programs, ultimately accelerating the development of rice varieties that are more resilient to stress and have improved agronomic traits. Furthermore, incorporating these gene expression predictions into genomic selection strategies could enhance the accuracy and efficiency of selecting desirable traits in breeding populations..

### Convolutional Neural Networks (CNNs)  
Convolutional Neural Networks (CNNs) are a type of deep learning architecture which have become renowned for their use in computer vision, natural language processing, and recommender systems. One unique capability of the CNN is its property of processing data with matrix-like topologies, such as images, genomic sequences, and frequency waves.

The driving force behind CNN architectures are the use of its convolutional layers. These layers apply a series of learnable filters to the input data. These kernels (filters) convolve (slide) over the input data to produce feature maps, allowing the detection of features such as edges, textures and patterns. The convolution operation, applies a dot-product, to capture local dependencies by analyzing relationships between neighboring elements.

In conjunction with the convolutional layers, activation functions introduce non-linearity into the model and is critical for the network's ability to learn and represent nuanced patters in the data. Furthermore, to decrease computational load and prevent overfitting, additional layers such as pooling and dropout layers are employed to reduce the dimensions of feature maps and to help with training by reducing data size and neuron connectedness. Finally, outputs are produced through fully connected layers. These layers facilitate the final classification by connecting every neuron in one layer to every neuron in the subsequent layer.

In the context to our research, CNNs can be leverage to decode the cis-regulatory elements in many species. By training the CNNs on genomic sequences, we can capture the frequency and spatial dependencies, allowing for the characterization and identification of motifs. 

---

## Methodology

### Transcription Counts for Classification  
To train the various CNN and random forest models, we utilized data from the the Sequence Read Archive and used calculated transcription counts for classification. The counts were produced using single end reads, available across several different Bioprojects using Kallisto (0.51.0). From other studies \cite{peleke24}, empirical evidence had shown that the model interpretability and computation complexity was balanced and acceptable for flanking sequences of 1500 nt lengths (1000 nt promoter - 500nt 5'UTR - 500 nt 3'UTR - 1000nt terminator sequences).

The trimmed reads were aligned to cDNA using kallisto with the following settings: {**-b 100**} to perform 100 bootstrap samples, and {**--single -l 200 -s 20**} for single-end read processing with an average fragment length of 200 bases and a standard deviation of 20 bases. The resulting Kallisto outputs provided transcript-level abundance estimates. Afterwards, to normalize the data we applied a log transform. These normalized counts were aggregated to produce a single TPM value for each gene, which served as the input for model training.{Will need to add transformed data plots}

### Data Encoding  
To train our convolutional neural network (CNN) on sets of genomic sequences, we employed the one-hot encoding method. One-hot encoding is a common technique to convert categorical data into numerical data [2]. For use in genomics, we convert nucleotide sequences into a numerical format that can be utilized by the CNN architecture. For this approach [1], each nucleotide (A, C, T, G) is represented as a binary vector.
\newline
Each nucleotide is encoded as follows:
    Adenine (A) is encoded as [1, 0, 0, 0]
    Cytosine (C) is encoded as [0, 1, 0, 0]
    Guanine (G) is encoded as [0, 0, 1, 0]
    Thymine (T) is encoded as [0, 0, 0, 1]

This method of encoding ensures that each nucleotide is distinctly represented, allowing the CNN to process the input sequences effectively. The resulting one-hot encoding sequence for any particular DNA strand of length \(n \) is an \(n \times 4\) matrix, where \(n \) is the number of nucleotides.

### CNN Architecture  
As mentioned, the driving force behind the convolutional deep learning architecture is the use of convolutional layers. Mathematically, the convolution operation for a single filter _f_ on an input matrix **X** can be represented as:

$$ (Y)_{ij} = (X * f)_{ij} = \sum_{m}\sum_{n} X_{i+m, j+n} \cdot f_{m, n} $$

where ∗∗ denotes the convolution operation, and Y is the resulting feature map. 

For our convolutional network, we initially use the model designed by Pelek et al. [1]([Deep learning the cis-regulatory code for gene expression in selected model plants | Nature Communications](https://www.nature.com/articles/s41467-024-47744-0#data-availability)). The model utilizes three 1D convolutional blocks (two convolution layers) with:
- Two convolutional layers: 
- Filters:
	- **Block 1 and 3**: 64 filters
	- **Block 2**: 128 filters
	- **Block 3**: 64 filters

Each block has a kernel size of 8. Every block includes a dropout layer to prevent overfitting, with a dropout rate of 25%. Additionally, each block employs a rectified linear activation function defined as:

$$  
\text{ReLU}(x) = \max(0, x)  
$$

To reduce the dimensions of the feature maps and improve computation time, the blocks use max-pooling layers. These layers work by taking the maximum value from the computed convolution (the resulting matrix). The max-pooling operation is represented as:

$$  
(Y)_{ij} = \max \{X_{i+k, j+l}\}  
$$

where k and l define the pooling window size. For the model described by Pelek, the pooling window size is 8.

After the convolution and pooling layers, the feature maps are transformed into a single vector, which is passed through a final fully connected layer. In these layers, all nodes are connected with subsequent layers. These fully connected layers are responsible for combining the features extracted from the convolution and making a final prediction.

The fully connected layers can be represented as:

$$  
Y = W \cdot X + b  
$$

where W is the weight matrix, X is the input vector, b is the bias vector, and Y is the output.

Similarly to the convolution blocks, the fully connected layers employ dropout after each layer. This technique helps prevent overfitting by randomly deactivating nodes, such that their input becomes zero during predictions.


---

## Data Types  
The data used in our analysis comes from several data bases and includes genome sequences and annotations for the species A. thaliana, S. lycopersicum, S. bicolor, Z. mays, and O. sativa (japonica). These reference genomes and annotations were used for the extraction of gene flanking regions and estimation of transcript counts and taken from  Ensembl plants database v52 ([plants.ensembl.org](plants.ensembl.org)) [GCA_000001735.1]([Arabidopsis_thaliana - Ensembl Genomes 60](https://plants.ensembl.org/Arabidopsis_thaliana/Info/Index)), [GCA_000188115.3]([Solanum_lycopersicum - Ensembl Genomes 60](https://plants.ensembl.org/Solanum_lycopersicum/Info/Index)), [GCA_000003195.3]([Sorghum_bicolor - Ensembl Genomes 60](https://plants.ensembl.org/Sorghum_bicolor/Info/Index)), and [GCA_902167145.1]([Zea_mays - Ensembl Genomes 60](https://plants.ensembl.org/Zea_mays/Info/Index)).

Additionally, transcriptomic single end short-read data was downloaded from the National Center for Biotechnology Information (NCBI) Sequence Read Archive (SRA) database for leaf and root data from Bioprojects to determine transcript profiles [PRJEB32665]([Arabidopsis tissue atlas (ID 600640) - BioProject - NCBI](https://www.ncbi.nlm.nih.gov/bioproject/PRJEB32665)), [SRP010775]([SRP010775 : Study : SRA Archive : NCBI](https://trace.ncbi.nlm.nih.gov/Traces/?view=study&acc=SRP010775)), [PRJNA171684]([Zea mays (ID 171684) - BioProject - NCBI](https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA171684)), [PRJEB22168]([Sorghum RNA-seq data of comparative transcriptome ... (ID 449035) - BioProject - NCBI](https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJEB22168)), [PRJNA237342]([Arabidopsis thaliana (ID 237342) - BioProject - NCBI](https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA237342)), [PRJNA640858]([ID 640858 - BioProject - NCBI](https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA640858)), [PRJNA217523]([Sorghum bicolor (ID 217523) - BioProject - NCBI](https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA217523)), and [PRJNA271595]([Conserved Molecular Program for Root Development i... (ID 271595) - BioProject - NCBI](https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA271595)).

---

## Results

### Model Comparisons  
Species model accuracy (w/ Rice): ![[Leaf_Training_Results 1.xlsx]]
# Moca-Blue Processing
---
The [moca-blue]([GitHub - NAMlab/moca_blue](https://github.com/NAMlab/moca_blue)) pipeline, as described is a tool-box from Simon Z. for the analysis of DNA motifs. There are six working directories that can be utilized for the analysis and includes clustering, importance scores, assigning nomenclature (based on model design), motif distribution, motif matching, and filtering. The list below includes said directories and the files to be run in order:
- mo_nom: Extracts motifs from TF-MoDisco hdf5 files. Assigns nomenclature for future interpretability and produces weblogos for motif visualization
	- get_rdf5_cwms_per_pattern_v1.1.R
	- mo_compare_JASPAR2020_v1.0.R
- mo_clu: Analyze and edit motif files stores in jaspar format. The purpose is to generate dendrograms or trees based on similarity matrices between motifs
	- mo_cluster_v2.7.R
	- mo_tree_viz.SZ.v1.0.R
- mo_imp: (Slightly redundant, scripts made personally before reimplemented model). Simply extracts and visualizes saliency maps and importance scores.
	- rdf5_get_epm_contrib_scores.v1.1.R
	- mo_imp_scores.v1.1.R
	- mo_imp_depth_v0.7.R
- mo_ran: For optimizing the search for motifs in the genome. Extracts positionally preferred ranges for each motif in a hdf5 file. Creates distributions and meta data associated with preferred positions.
	- rdf5_get_seql_per_patternV2.1.R
	- epm_occurence_ranges_TSS-TTS.1.6.R
	- meta_motif_ranges_characteristics_TSS-TTS.1.4.R
- BLAMM (mapping motifs to genes): BLAMM maps motifs to reference genome. This links the matches between epm patterns and acutal DNA sequences. 
- mo_proj: After searching (using BLAMM), the results are further filter based on specified criteria. Suggested to split occurrences using script from BLAMM pipeline.
	- occ_filter_v1.1.R
	- mo_feat-filter.v3.4.R
	- mo_feature_tester.v1.0.R
	- mo_predictabilityV1.5.R
	- mo_check_mapping-performance_V1.7.R
	- mo_genotype_variance.v1.4.R

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
#### Running Changes to Moca-blue
Moca-blue as it is currently provided in the [NAMLab Github]([moca_blue/README.md at main · NAMlab/moca_blue · GitHub](https://github.com/NAMlab/moca_blue/blob/main/README.md)) ; however, is currently incomplete and does not use files known from the original paper. In order to complete the motif analysis it was required to modify some of these scripts in order to run on a new species. Below are altered scripts and the modifications made.
**Occ_filter_v1.2_JaredsEdit.R**:
The first altered script included the occ_filter.R script. This script is initially run after completing mapping with BLAMM. In the original script it is assumed that from the occurrences output (epms with associated genes) includes metadata such as loc IDs; however, doesn't. To circumvent this issue, I now assign these missing data using a reference gene model file (.gff). Before filtering motifs that are outside the 1500bp feature region (gene, CDS, intron, ect.), I complete preprocessing on the reference file by removing Mt. and Pt. genes. Additionally, I filter information within the reference to make descriptions and identifiers easy to read and work with.
```
gff_file <- "Oryza_sativa.IRGSP-1.0.60.gff3"
gene_df <- read.table(paste0(refpath, "/", gff_file),
                      header = FALSE, sep = "\t", quote = "",
                      stringsAsFactors = FALSE, comment.char = "#") 

gene_df <- gene_df %>%
  #filter(V3 == "gene") %>%
  select(chr = V1, feature_start = V4, feature_end = V5, feature_strand = V7, type = V3, info = V9) %>%
  mutate_at(c("chr"), as.character) %>%
  #filter(!chr %in% c("Pt", "Mt"))
  filter(grepl("^\\d+$", chr)) %>%
  filter(type != "chromosome")

# Extract gene_id from the info column
#gene_df$gene_id <- sub(".*ID=gene:*([^;]+).*", "\\1", gene_df$info)
gene_df$gene_id <- sub(".*ID=gene:([^;]+).*", "\\1", gene_df$info)
#gene_df$gene_id <- sub(".*ID=transcript:([^;]+).*", "\\1", gene_df$gene_id)
gene_df <- gene_df %>% filter(!grepl("transcript:|CDS:", gene_id))

gene_df <- gene_df %>%
  select(-info) %>%
  mutate(across(c(feature_start, feature_end), as.numeric))
```

Additionally, due to lack of metadata from BLAMM and ease of implementation, I've added the "GenomicRanges" library to easily assign and manipulate the transcription start and termination sites. This allows for the comparison of genomic regions and simplifies comparisons to single algebraic comparisons.
```
gene_df <- gene_df %>%
  mutate(TSS = ifelse(feature_strand == "+", feature_start, feature_end),
         TTS = ifelse(feature_strand == "-", feature_end, feature_start))

# Create GRanges objects for the gene boundaries
gr_TSS <- GRanges(seqnames = gene_df$chr,
                  ranges = IRanges(start = gene_df$TSS, width = 1),
                  strand = gene_df$feature_strand)
gr_TTS <- GRanges(seqnames = gene_df$chr,
                  ranges = IRanges(start = gene_df$TTS, width = 1),
                  strand = gene_df$feature_strand)
gr_genes <- GRanges(seqnames = gene_df$chr,
                    ranges = IRanges(start = gene_df$feature_start, end = gene_df$feature_end),
                    strand = gene_df$feature_strand)

common_seqs <- unique(gene_df$chr)
```
To complete the filtering (for this step motifs that are within 1500bp of TSS and TTS), after computing TSS and TTS positions I then construct full feature intervals and get common chromsome annotations and only keep motifs on a valid chromosome "common_idx".
```
  # Identify rows where the chromosome is among those in the gene_df
  common_idx <- which(occurrence_df$chr %in% common_seqs)
  
  if (length(common_idx) > 0) {
    # Create GRanges object for the motif occurrences on common chromosomes
    gr_motifs_common <- GRanges(seqnames = occurrence_df$chr[common_idx],
                                ranges = IRanges(start = occurrence_df$mstart[common_idx],
                                                 end = occurrence_df$mend[common_idx]),
                                strand = occurrence_df$strand[common_idx])
    seqlevels(gr_motifs_common) <- common_seqs
    
    # Calculate distance to the nearest TSS
    nearest_idx_TSS <- nearest(gr_motifs_common, gr_TSS)
    dTSS <- rep(NA, length(gr_motifs_common))
    valid_idx <- !is.na(nearest_idx_TSS)
    if (any(valid_idx)) {
      dTSS[valid_idx] <- distance(gr_motifs_common[valid_idx], gr_TSS[nearest_idx_TSS[valid_idx]])
    }
    
    # Calculate distance to the nearest TTS
    nearest_idx_TTS <- nearest(gr_motifs_common, gr_TTS)
    dTTS <- rep(NA, length(gr_motifs_common))
    valid_idx2 <- !is.na(nearest_idx_TTS)
    if (any(valid_idx2)) {
      dTTS[valid_idx2] <- distance(gr_motifs_common[valid_idx2], gr_TTS[nearest_idx_TTS[valid_idx2]])
    }
    
    # Assign the calculated distances back to the full-length vectors
    dist_to_TSS[common_idx] <- dTSS
    dist_to_TTS[common_idx] <- dTTS
    
    # Find the nearest gene for each motif occurrence to retrieve gene boundaries
    nearest_idx_gene <- nearest(gr_motifs_common, gr_genes)
    if (length(nearest_idx_gene) > 0) {
      # Save gene boundaries and gene_id into separate vectors
      feature_start_vec[common_idx] <- gene_df$feature_start[nearest_idx_gene]
      feature_end_vec[common_idx]   <- gene_df$feature_end[nearest_idx_gene]
      gene_id_vec[common_idx]    <- gene_df$gene_id[nearest_idx_gene]
      
      # Construct the loc field as "chr:feature_start-feature_end"
      loc_vec[common_idx] <- paste0(as.character(seqnames(gr_genes)[nearest_idx_gene]), ":",
                                    gene_df$feature_start[nearest_idx_gene], "-",
                                    gene_df$feature_end[nearest_idx_gene])
    }
  }
  
  # Add new columns to the occurrence data frame
  occurrence_df$dist_to_TSS <- dist_to_TSS
  occurrence_df$dist_to_TTS <- dist_to_TTS
  occurrence_df$loc         <- loc_vec
  occurrence_df$feature_start  <- feature_start_vec
  occurrence_df$feature_end    <- feature_end_vec
  occurrence_df$gene_id     <- gene_id_vec
  
  # Filter to keep only motif occurrences within 1500 bp of either TSS or TTS
  filtered_occ <- occurrence_df %>% 
    filter((!is.na(dist_to_TSS) & dist_to_TSS <= 1500) |
             (!is.na(dist_to_TTS) & dist_to_TTS <= 1500))
```

## Citations  
1. Peleke, F.F., Zumkeller, S.M., Gültas, M., et al. *Deep learning the cis-regulatory code for gene expression in selected model plants*. Nat Commun 15, 3488 (2024). DOI: 10.1038/s41467-024-47744-0 - [Reference]([Deep learning the cis-regulatory code for gene expression in selected model plants | Nature Communications](https://www.nature.com/articles/s41467-024-47744-0)) [GitHub Scripts]([GitHub - NAMlab/DeepCRE: Deep learning the cis-regulatory code for gene expression in selected model plants](https://github.com/NAMlab/DeepCRE/tree/main))

## Including Rice Data

### 1. Procurement of Rice Data

- **Gene Reference File**:    
  For the Oryza sativa japonica dataset, we used the version 60 gene reference file from Ensembl Plants.    
  **Source Link**: [Ensembl Plants Oryza sativa]([https://plants.ensembl.org/Oryza_sativa_japonica/Info/Index)](https://plants.ensembl.org/Oryza_sativa_japonica/Info/Index) "https://plants.ensembl.org/oryza_sativa_japonica/info/index)")    
  **File Details**:    
    - **Version**: GTF v60    
    - **Location**: Saved locally in the project directory under `jxr550\Oryza_sativa.IRGSP-1.0.60.gtf`.  

- **Expression Data**:    
  TPM (Transcripts Per Million) counts were obtained from the Expression Atlas. The dataset provided gene expression data for rice leaf tissues before and after flowering, including multiple experimental runs.    
  **Source Link**: [Expression Atlas](https://nam12.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.ebi.ac.uk%2Fgxa%2Fplant%2Fexperiments&data=05%7C02%7Cjared.richardson%40uta.edu%7C0b140ba6db5942ed5b2a08dccd26fd60%7C5cdc5b43d7be4caa8173729e3b0a62d9%7C0%7C0%7C638610814565484124%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=KKzvi15KCunF9q9YzeouqPT3tCQ7NgS9vXEiLVslSFQ%3D&reserved=0): [Current Training Data]([Experiment < Expression Atlas < EMBL-EBI](https://www.ebi.ac.uk/gxa/experiments/E-GEOD-56463/Results))

### 2. Preprocessing and Analysis

#### **Reference GTF File Usage**:  
The reference GTF file served as the master list for expected genes. It ensured that all gene IDs in our analysis conformed to the standardized nomenclature (`OsXXgXXXXX`). This was particularly important for resolving discrepancies in naming conventions between the Expression Atlas and other datasets.

#### **Expression Atlas TPM Counts**:  
- **Key Observation**: The Expression Atlas dataset excluded genes with zero expression, leading to missing gene counts in our initial analysis.  
- **Solution**: Missing genes identified in the GTF file but absent from the Expression Atlas data were imputed with a count of zero. This replacement maintained uniform input dimensions for training and ensured all genes were accounted for.

### 3. Analysis

A comparison, using R, was conducted between the reference GTF file and the Expression Atlas data revealed several discrepancies:  
- **Missing Genes**: Genes such as `Os03g0146900` were absent in the Expression Atlas data, despite similar neighboring genes (e.g., `Os03g0146000` and `Os03g0141000`) being present.  
- **Resolution**: The missing genes were explicitly added with zero TPM values to avoid `KeyError` issues during training.
Additionally, to verify the metadata and to confirm missing entries were caused by that of zero-expression, Kallisto runs using the same reference genes confirmed that genes causing KeyErrors were a result of zero transcript expression.
### 4. Integration into CNN Training

The corrected dataset, combining the reference GTF file and TPM counts from the Expression Atlas (with missing genes replaced by zero), was successfully used for training the CNN. This step resolved all `KeyError` issues and ensured consistency between the rice data and other model plant datasets.

### 5. File Organization and References  
- **Reference GTF File**: `jxr5507/Oryza_sativa.IRGSP-1.0.60.gtf`  
- **Expression Atlas Data**: `jxr5507/expresson_atlas_data2_leaf_trim`  
- **Processed Data**: `jxr5507/expression_atlas_leaf_counts_complete`

