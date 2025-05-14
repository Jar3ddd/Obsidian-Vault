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

**mo_feat-filter_JaredsEdit**
After filtering for motifs near annotated features, we would like to further refine our list for biological relevance and add even more scrutinous statistical filtering. This script reduces motif occurrence data relative to gene annotations to:
- Filter motifs by distance (added as check from original fitering)
- Annotate motifs as upstream/downstream
- Apply region-weight filters using motif summary statistics
- Output filtered table and also create '.bed' file for visualization

In addition to statistical filtering, this also adds interpretability to the model outputs by providing:
- GO enrichment
- IGV/USCS genome annotations
- Motif enrichment summaries

Most changes applied are simply pipeline and data handling measures to account for differences in content from previous moca_blue steps and BLAMM outputs.
```
merg_df <- merg_df %>%
  mutate(motif_length = mend - mstart + 1) %>%
  filter(motif_length == word_size)

merg_dfB <- merg_df %>%
  mutate(dist_transc_border = pmin(dist_to_TSS, dist_to_TTS, na.rm = TRUE)) %>%
  mutate(region = ifelse(dist_to_TSS <= dist_to_TTS, "upstream", "downstream")) %>%
  mutate(region = ifelse(strand.x == "-", ifelse(region == "upstream", "downstream", "upstream"), region))
