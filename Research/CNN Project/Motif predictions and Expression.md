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

## Results
### Model Comparisons  
Species model accuracy (w/ Rice): Leaf Models
![[CNN_Leaf_Model#sheet1|0-10:0-4]]

---

Species model accuracy (w/ Rice): Root Models
![[CNN_Root_Model#sheet1|0-5:0-4]]



**Model Averages**
![[averagesPlot.pdf]]
Summary Table:

|     | averages | model_type | specie           |
| --- | -------- | ---------- | ---------------- |
| 1   | 0.820962 | Leaf       | Arabidopsis      |
| 2   | 0.851567 | Root       | Solanum          |
| 3   | 0.805955 | Leaf       | S. Bicolor       |
| 4   | 0.822908 | Root       | Z. Mays          |
| 5   | 0.809666 | Leaf       | O. Sativa (jap.) |
| 6   | 0.809666 | Root       | Arabidopsis      |
| 7   | 0.82276  | Leaf       | Solanum          |
| 8   | 0.856558 | Root       | S. Bicolor       |
| 9   | 0.797546 | Leaf       | Z. Mays          |
| 10  | 0.812109 | Root       | O. Sativa (jap.) |



### Moca Blue Analysis

**Rice Model EPM results**
- Also try and include importance scores across validation sequences (or visual)
- Match epms with JASPAR -> Get EPMs for rice and begin comparing with those for other speice -> rice specific motifs

- For the clustering algorithm script (creates the dendrograms) compare PWMs of rice with the clusters defined in paper (the 2CWY+, 2CT_, ect...)
- ==Include EPM clusters occurrence relative to start and stop sites==



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

