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
#CNN
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

## Results: For Eight Rice Tissues
### Model Comparisons Across Eight Rice Tissues

In this study, we focused on eight tissues of *Oryza sativa* (japonica) - anther, endosperm, inflorescence, leaf, panicle, pistil, root, and seed - rather than comparing across different plant species.  For each tissue, a convolutional neural network (CNN) was trained using leave one out chromosome (of 12).  Expression labels were placed into three quantiles; however, all training and evaluation steps excluded class 2 genes (intermediate expression), so the models learned to differentiate only low‑expression (class 0) and high‑expression (class 1) genes.  Predicted probabilities ≥ 0.5 were treated as high expression (1), and values < 0.5 were labelled low expression (0).

#### Training Results

The networks were trained using only class 0 and class 1 genes.  Training runs consistently produced high accuracy and AUROC values, indicating that the models generalize well across different chromosome partitions.  These metrics are reported below.

After training, each tissue‑specific model was applied to a held‑out chromosome from the rice genome.  Only genes labelled low or high expression (classes 0 or 1) were included, and predicted probabilities ≥ 0.5 were labelled high expression.  The table below summarizes prediction performance.  Accuracy ranges from 0.80 to 0.89; AUROC values span 0.88–0.95; and AUPR values range from 0.88–0.96.  F1‑scores are balanced between high and low expression classes, and confusion matrices show no obvious bias toward either class.

| Tissue            | Accuracy | AUROC |  AUPR | F1 (high) | F1 (low) | True Low (TN) | False High (FP) | False Low (FN) | True High (TP) |
| ----------------- | -------: | ----: | ----: | --------: | -------: | ------------: | --------------: | -------------: | -------------: |
| **Anther**        |    0.861 | 0.931 | 0.948 |     0.878 |    0.838 |         5 391 |             811 |          1 276 |          7 513 |
| **Endosperm**     |    0.850 | 0.920 | 0.916 |     0.849 |    0.850 |         8 492 |           1 527 |          1 465 |          8 425 |
| **Inflorescence** |    0.878 | 0.943 | 0.957 |     0.896 |    0.852 |         5 122 |             799 |            977 |          7 630 |
| **Leaf**          |    0.802 | 0.880 | 0.883 |     0.804 |    0.801 |         5 674 |           1 288 |          1 534 |          5 777 |
| **Panicle**       |    0.882 | 0.948 | 0.964 |     0.904 |    0.849 |         4 594 |             667 |            966 |          7 656 |
| **Pistil**        |    0.892 | 0.951 | 0.961 |     0.907 |    0.871 |         5 336 |             688 |            898 |          7 719 |
| **Root**          |    0.812 | 0.890 | 0.912 |     0.829 |    0.791 |         4 505 |             979 |          1 408 |          5 772 |
| **Seed**          |    0.885 | 0.949 | 0.964 |     0.904 |    0.858 |         4 893 |             678 |            941 |          7 612 |

The following bar charts depict accuracy, AUROC and AUPR for predictions across tissues:

**Include bar plots for the three results**


### Example of probability distribution

To visualize how the model separates low‑ and high‑expression genes, the histogram below shows predicted high‑expression probabilities for the Leaf tissue model.  True high‑expression genes (class 1) cluster near 1, while true low‑expression genes (class 0) cluster near 0.  The overlap area corresponds to the misclassified cases.

![[pred_prob_distribution_leaf_updated.png]]



## Discussion

When focusing solely on rice tissues, the CNN models demonstrate consistently high performance on the held‑out data.  Prediction metrics show balanced precision and recall for both expression classes and no systematic bias toward either high or low expression.  The **Pistil** and **Panicle** models achieve slightly higher AUROC and AUPR than others, whereas **Leaf** and **Root** are slightly more challenging; nevertheless, all tissues exhibit strong discriminative ability.




### Moca Blue Analysis

**Rice Model EPM results**
To analyze predicted motifs and interpret the regulatory elements caputured by the CNN model, we will utilize the MOCA-Blue pipeline originally developed by S. Zumkeller. This framewrk enables the post processing of model derived patters through the use of contribution weight matricies (CWMs), matrix representations of nucleotide level importance derived from saliency maps. CWMs allow us to characterize the expression predictive motifs (EPMs) by highlighting the contribution of each nucleotide position to the model's predictions

For Oryza sativa (jap.) there were a total of 64 and 63 EPMs discovered for leaves and roots, respectively. These CWMs formed the basis of subsequent analysis, including motif annotations via JASPAR2020 database, unsupervised clusting of EPMs based on similarity, and genomic localization using BLAMM for positional enrichment relative to gene features such as the TSS and TTS.

Examples of weblogos derived from the CWMs. Included are motifs that contained higher total contribution weights from each nucleotide.
![[epm_Ory_S0_Leaf_p0m18F_91_17_SSCGCCGCGGCCGC.png]]
![[epm_Ory_S0_Leaf_p0m02F_1028_17.6_NCCTCCTCCTCCNC.png]]

Histograms depicting the positional frequency of EPMs across input sequences revlealed a strong enrichment near transcription start sites (TSS) and transcription termination sites (TTS). This trend closely mirrors findings from prior analysis of the model species Arabidopsis thaliana, Zea Mays, Sorghum bicolor, and Solanum lycopersicum, suggesting a conserved regulatory architecture across plant species. The observed clustering of EPMs around TSS and TTS positions support the biological relevance of the motifs confirms that the CNN model has learned features associated with promoter and terminator regions. 
##### Leaf Models
To assess the biological relevance of the expression predictive motifs (EPMs) identified by our CNN models, we performed a transcription factor (TF) motif comparison using the JASPAR2020 database. Each EPM was queried against the database to identify potiential matches to known plant TF binding profiles. These comparisons were conducted for the five species: Oryza sativa (jap.), Zea mays, Arabidopsis thaliana, Sorghum bicolor, and Solanum lycopersicum.
Using a R and R-studio (==add created scripts to a supplement or methods section==), we generated a binary comparison matrix indicating the presence or absence of JASPAR2020 matched TFs across the five species. This allowed for the assessment of both species specific and shared regulatory elements, providing some additional insight into the conservation and divergence of transcriptional control elements inferred by the CNN based model.
![[venndiagram_TF_components.pdf]]
No transcription factors (TFs) were universally conserved across all fives species. However, several subsets of TFs were common to multiple species, indicative of partially conserved regulatory elements that may reflect core transcriptional controls shared broadly among plants (==list TF sets==). Despite this partial conservation, the substantial number of species exclusive TFs highlights that our single species CNN model effectively captures specie specific regulatory motifs. Notably, among these species specific TFs were AHL12, ATHB34, ERF109, ERF7, IDD4, OBPI1, Os05g047200, and TCXI3. Moreover, certain TFs, including TCXI3, TB1, and ERF008 matched more frequently across multiple EPMs., potentially signifying broader roles.

No TFs were seen in all species; however there were several sets that were conserved across multiple species (==May list specific sets, i.e. between species==) . Though these particular sets suggest shared regulatory motifs and additionally may represent core regulatory elements in plant gene expression; the large class of separated TFs concludes the single species models are learning species specific TFs. Several of these included AHL12, ATHB34, ERF109, ERF7, IDD4, OBP1, Os05g0497200, and TCXI3. Several EPMs were found to match more often, including TCXI3, TB1 and ERF008.
Among the transcription factors (TFs) identified uniquely to rice, notable examples included Os05g0497200 and IDD4. Os05g0497200 ([MSF1]([MULTI-FLORET SPIKELET1, which encodes an AP2/ERF protein, determines spikelet meristem fate and sterile lemma identity in rice - PubMed](https://pubmed.ncbi.nlm.nih.gov/23629832/))) is a rice-specific TF previously reported to be involved in rice development processes and responses to environmental stressors and effects the multi-floret spikelete1 (==will need to read more on the publication==). Similarly, IDD4 (Indeterminate Domain 4) belongs to the indeterminate domain (IDD) trascription factor family, recognized for regulating expression patterns under abiotic stresses and hormone responses such as grain size, flowering time, and overall yield potential in rice ([Zhang et. al. 2020]([New insight into comprehensive analysis of INDETERMINATE DOMAIN (IDD) gene family in rice - PubMed](https://pubmed.ncbi.nlm.nih.gov/32912488/))).
Also, several TFs, including TCXI3, TB1, and ERF008 were frequently matched across multiple predicted patterns, highlighting their broad regulatory influence.

- contribution score: measure similarity between EPM and JASPAR motif
- p-values: hypothesis to discover statistically significant matches
- tf annotations: group TF by function instead of name (find if theme in regulation function)


---
### Prediction on mutated sequences
To support future efforts in allele mining and variant prioritization, we evaluated how sequence variants affected predicted gene expression by running our CNN model on mutated version of rice genes. This approach allows for the assessment of regulatory consequences of naturally occurring genetic variation, potentially identifying alleles with expression modifying potential.
Utilizing the variant call format (VCF) files from ... , which contain single nucleotide variants (SNVs) and small indels for Oryza sativa (jap.). The genomic coordinates of transcription start sites (TSS) and transcription termination sites (TTS) were first extracted from the Ensemble GTF annotation file and added to a BED file using the "extract_regions.py" script found in the variant_files directory in Atlas. Using the BED file with position coordinates, we obtained flanking sequences for each gene from the rice reference genome FASTA using *bedtools getfasta*. Because the positions of interest were predefined (i.e., TSS and TTS), no dynamic scanning was needed only standardized padding and sequence concatenation. Once, the sequences for each flank of every gene is generated by bcftools from the BED file, the mutations are applied using a Python script "apply_mutations.py".It parses each sequence and overlays the mutations from the VCF where it overlaps. The final output is FASTA file where the reference sequences have been mutated to reflect the genotypic differences found from the VCF. These mutated sequences, now only need to be appended and padded with '20 Ns' per the expected model's desired shape. To perform this step, all that is required is the running of the "join_sequences.py". For the final step, the CNN model is executed via an earlier script meant to predict on raw genotype sequences, "get_preds_V2.py", located in the "seq_predictions" directory. This script accepts any FASTA file and returns the predicted gene expression values, given the correct sequence shapes (i.e., 3020bp).

==I will like to add the incorporation of INDEL mutations, adding to the types of mutations that can be predicted on. Also, the execution of several scripts is cumbersome and is not optimal, so for usability I will try to explore Snakemake workflows (or similar).==


- Also try and include importance scores across validation sequences (or visual) -> currently have total contribution >>> look to mo_imp for base level contributions
- For the clustering algorithm script (creates the dendrograms) compare PWMs of rice with the clusters defined in paper (the 2CWY+, 2CT_, ect...)

### Analysis of QT12 Gene
To also analyze the expression changes we will also look at the impact of variations in the QT12 gene, a determinant of grain quality and yield thermotolerance in rice (Liu et. al., 2025). 

The QT12 gene encodes a protein implicated in regulating endosperm storage substance homeostasis through unfolded protein response (UPR). The study by Li et. al. (2025) identified a natural variation in QT12 regulatory sequences particularly within the promoter region, leading to differential expression under heat stress. This variation consists of a key SNP (single nucleotide polymorphism) affecting the interaction of transcription factors such as NF-Y complexes, particularly NF-YA8, NF-YB9, and NF-YC10. This single variation significantly alters the grain's thermal tolerance and subsequently effects both yield and grain quality

#### Methods:
First, we must retrieve genomic sequences for Oryza Sativa japonica. The original CNN model is trained and predicts on 1kb upstream (promoter), the complete 5' UTR, 3' UTR, and 1kb downstream terminator regions. ==Using (*still finding nice resource for variant and sequence information*), we will need to retrieve both the wild type and mutant regulatory sequences. To ensure that the SNP is properly Identified (as  in Li et. al., (2025)) these sequences will be aligned and the location of the variation will be compared to literature and proximity to the "CCAAT" box.== Initially, after obtaining these sequences, I will use the default CNN pipeline to encode the sequences and pad them accordingly. 

To add to interpretation to the CNN predictions and help improve understanding of the regulatory mechanisms disrupted or activated by the natural variant, we will utilize DeepLIFT. This already completed in the moca_blue pipeline, will allow us to calculate the nucleotide level importance scores, highlighting the regulatory EPMS that impact predicted expression levels the most.
Subsequently, using TF-modisco we will identify the EPMs within the CNN outputs. Using this we can comparatively assess the location of the discovered motifs and their known SNP location in the QT12 promoter that Li et. al. (2025) identified. 

==Future - using the JASAPR (or other) to map EPMs to known transcription factors. Will be looking for TF binding sites corresponding to the NF-Y family found in the QT12 paper. Would also be neat to link predictions to phenotypic outcomes, though not sure with current pipeline if possible but look into hypothesis testing (also ask Dr. Edwards too).==

---

### SHAP Analysis: Genome-wide Variant Importance

#### Overview:
In concurrence with the mutated sequence prediction, in R we used a SHAP-based script to help identify variants that drive expression differences for eight rice tissues (leaf, root, panicle, inflorescence, seed, endosperm, anther, and pistil). The script utilized calculated per-variant SHAP-$\delta$ (per base difference between the SHAP contribution for a reference sequence and mutated sequence) from the prediction step and computed zero-shot scores from Plant Caduceus. The summary and plot files combine all chromosomes and tissues. Below is a summary of the results:

##### Variant importance and Zero-shot Scores
To remove the extra "noise" we only considered the top 0.1% of variants by $\left| \text{SHAP-}\delta \right|$  on each chromosome. Across all chromosomes there was similarity in the magnitudes of importance ($\left| SHAP \right|$ threshold ~ 0.045) suggesting that a comparable absolute SHAP values distinguishes the most influential across tissues. (==may also be nice to include the other metrics, but are they useful? Sign agreement, ect...==). However, despite this agreement there was weak correlation between absolute sizes of SHAP-$\delta$  and ZSS with an average Pearson correlation ~0.018. Some chromosomes (e.g., Chr9) show slightly positive correlation while others show negative correlation (e.g., Chr3). Therefore, we conclude that there is a lack of relationship and hence a variant with a large SHAP contribution does not necessarily have a large ZSS and vice versa.
##### Rare-allele enrichment among high-impact variants
Similar to Dr. Buckler et. al () we wished to assess the enrichment of rare variants among those with large SHAP contributions. To conduct this we computed the average minor allele frequency (MAF) across binned portions of the entire SHAP values. Additionally, we provided precision curves depicting the proportion of the top-N variants whose MAF was below the threshold hold of 0.01. 
Overall, the line graphs for each chromosome portray that as the SHAP values increase the average minor allele frequency decreases (with the exception of Chr3). In other words, variants that high impact on determining gene expression had lower population frequency. These results agree with the ZSS for deleteriousness from Plant Caduceus which demonstrated that mutations with functional effect had 3-fold lower minor allele frequencies. 

##### Relationship between SHAP and zero-shot scores
We initially tried to find some relationship or correlation between SHAP and zero-shot scores with little success, indicating that the models are inherently capturing different things. First, scatter plots of $\left| \text{SHAP-}\delta \right|$ versus $\left| \text{ZSS} \right|$ for the top 0.1% of variants reveal a wide dispersion with only a slight negative tend for Chr9. Additionally, many variants with high ZSS have moderate contributions and vice versa.
Likewise, we turned to ZSS enrichment across SHAP bins. For the top 0.1% of variants, grouping variants into 0.01-wide $\left| SHAP \right|$ bins shows that roughly 40-50% exceed $\left| ZSS \right| > 1$ in every bin, without a monotonic increase, further illustrating the weak relationship between SHAP magnitude and ZSS.
Despite this lack of clear relationship, the connection between SHAP-ZSS varies by chromosome.
- ==may include those extra metrics from previous highlight==

##### Summary
Taken together, the SHAP analysis across eight rice tissues (leaf, root, panicle, inflorescence, seed, endosperm, anther, and pistil) reveals a consistent and biologically meaningful pattern in how genomic variants influence predicted gene expression. Across all tissues, variants with the largest SHAP contributions tend to be rare, with mean minor allele frequency (MAF) decreasing steadily from low to high SHAP percentiles. This rare-allele enrichment implies that tissue specific expression variation is largely driven by a shared pool of low-frequency regulatory variants rather than by common alleles.
- line plot for MAF v SHAP percentile

The relationship between SHAP importance and experimentally derived zero-shot scores (ZSS) is more nuanced. On average, the sign of the SHAP contribution agrees with the direction of the ZSS in roughly two-thirds of high impact variants, suggesting that SHAP caputres the same general direction of regulatory effect seen in empirical data. However, the magnitudes of the two measures only correlate weakly as shown in the scatter plot of $\left| \text{SHAP-}\delta \right|$ and $\left| ZSS \right|$ 
- scatter plot
Many variants with large SHAP effects exhibit modest ZSS values, and vice versa, indicating that the sign information in more reliable than the magnitude when interpreting variant influence. The color coding of the scatter plot also reinforces the enrichment of rare variants among those with large SHAP values, with lighter-colored (low frequency) points concentrated in the upper range of SHAP importance.
Chromosome level comparisons further highlight heterogeneity in how SHAP and ZSS relate to each other. Chromosome 4, for instance, shows he strongest sign agreement, while chromosome 5 exhibits bot the weakest agreement and the highest mean $\left| ZSS \right|$. Such differences could potentially reflect the local genomic structure differences in gene density, linkage disequilibrium, or distribution of tissue specific regulatory elements that show how sequence variation translates into expression effects.
### Candidate Analysis
To further refine the results, we begun looking at top 1% values of SHAP values



---
### Prediction with Experimental Conditions
To improve the power and versatility of the model and its post processing we initially introduced several tissue models including the standard leaf and root among several other (8 total). Since some expression is dependent on factors such as experimental condition, we further extend our model to train on such transcription data. Using the ==Tenor=== database, we retrieved several tissues counts (RPK) for many experimental condtions. 













---
## Citations  #references 
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

gabriel aguular -- englis hfor defence