This is a reminder for producing mutations for example genes
## QT12
The CNN model predicts using 1500bp flanking regions. Specifically, it will need the 5' (TSS) and 3' (TTS) regions including 1kb upstream/downstream with .5kb downstream/upstream inside the gene for the two flanks, respectively.
Hence I will need the regions:
5' Flank: (3,718,814 - 1000) = 3,717,814 - 3,719,314 = (3,718,814 + 500)
3' Flank: (3,720,267 - 500) = 3,719,767 - 3,721,267 = (3,720,267 + 1000)

I use samtools to get the corresponding sequence, (i.e. samtools faidx ref.fasta 12:X-X > flank.fasta)

In seq_retrieval I stitch and pad them.

I then will write a script to parse vcf and apply the mutations individually. Then save as a single fasta to then use with the seq_prediction

(06/26) The model doesn't seem to pick up on the differences in expression caused by mutations. I will try with inflorescence and panicle transcript data. Also, I will try adding potentially missing genes from the GTF file, in case they were dropped in the original transcript computation.

(06/30) New data added and missing have been included with R. No changes in prediction as probabilities are consistent across mutations from Dr. Edwards vcf file
**Structural Variants in Paper**:
In the CNN paper, section "CNN models identify perturbations in gene expression associated with structural variance among sub-species and varieties of tomato" contains analysis on if the CNN can predict difference in gene expression caused by genetic variants (in tomato). The model used a **MSR** CNN model on the original four model species. 
- 15 tomato genotypes procured with known structural variants (Alonge et al. 2020)
- Included SNPs, as well as INDELS
The method seems similar to our pipeline for QT12. That is, for each gene extract the regions from flanks, introduce mutations from VCF, and then rerun sequences to predict expression probability.
For future work (I think was hinted in Wed. meeting)
- Compare expression predictions
- Genes with variance > .005 -> mark as differentially expressed
- Matched genes with expression predicitons and inserted mutations
Afterwards, use BLAMM to map EMPs to each of the genotype and mark as conserved or mutated (present for all genotypes, disrupted due to mutation), i.e., say for example EPM1 is in QT12 wild variety is disrupted by mutation. In other words, want to compare if EPM gained or lost due to mutations.

To start assessing what the model is seeing or if there are any issues I will try and look at the saliency map. This map will allows to see how the model scores base positions and their importance for prediction. Within **seq_prediction** will will write a small script to visualize the added saliency column in the **seq_retrieval** (dated version).
## qSH1
1:36445456-36449951
- Symbol: qSH1,RIL1
- MSU: [LOC_Os01g62920](http://rice.uga.edu/cgi-bin/ORF_infopage.cgi?orf=LOC_Os01g62920)
- RAPdb: [Os01g0848400](https://rapdb.dna.affrc.go.jp/locus/?name=Os01g0848400)
**The SNP is located 12kb away from TSS site. Specific location coordinates 1:36433021**

5' Flank: (36445456 - 1000) = 36,444,456 - 36,445,956 (36445456 + 500)
3' Flank: (36449951 - 500) = 36449451 - 36450951 (36449951 + 1000)

## OsSPL14 (IPA1)
8:25274449-25278696
- Symbol: OsSPL14,IPA1,WFP
- MSU: [LOC_Os08g39890](http://rice.uga.edu/cgi-bin/ORF_infopage.cgi?orf=LOC_Os08g39890)
- RAPdb: [Os08g0509600](https://rapdb.dna.affrc.go.jp/locus/?name=Os08g0509600)
**Mutation supposedly in third exon or coding region of OsSPL14). Associated with OsmiR156 binding site. "Transcriptional activator that binds to the SBP-box DNA core binding motif 5'-GTAC-3'." (UniProt). Also, look at, "The domesticated high-expression promoter allele (WFP) differs from Nipponbare by multiple SNPs within a ~2.6 kb region upstream of the transcription start site (TSS)."

One key distinguishing SNP is used in the SPL14‑04SNP marker, located approximately at Chr8:25,850,000 (based on alignment around Os08g0509600), although precise coordinates vary by ~±500 bp across varieties .

No single definitive SNP has been fixed—rather, the causal variation is a haplotype across ~2.6 kb upstream, rather than a precise single nucleotide position. **

5' Flank: (25274449 - 1000) = 25273449 - 25274949 (25274449 + 500)
3' Flank: (25278696 - 500) = 25278196 - 25279696 (25278696 + 1000)


## OsTB1
3:28428504-28430462
- Symbol: OsTB1,FC1,SCM3,MP3
- MSU: [LOC_Os03g49880](http://rice.uga.edu/cgi-bin/ORF_infopage.cgi?orf=LOC_Os03g49880)
- RAPdb: [Os03g0706500](https://rapdb.dna.affrc.go.jp/locus/?name=Os03g0706500)
**Mutation is a deletion within coding region (look for CDS coordinates).**

5' Flank: (28428504 - 1000) = 28427504 - 28429004 (28428504 + 500)
3' Flank: (28430462 - 500) = 28429962 - 28431462 (28430462 + 1000)


## OsMADS1
3:6052750-6061369
- Symbol: OsLG3b,OsMADS1,LHS1,AFO
- MSU: [LOC_Os03g11614](http://rice.uga.edu/cgi-bin/ORF_infopage.cgi?orf=LOC_Os03g11614)
- RAPdb: [Os03g0215400](https://rapdb.dna.affrc.go.jp/locus/?name=Os03g0215400)
==Cannot find Liu paper from PNAS. Papers found under differing Liu for varying journals. Ask for clarification. Also, may still work taking haplotypes from region to see if any are important across any of the journals.==

## Wx
6:1765622-1770656
- Symbol: Wx,qFC6
- MSU: [LOC_Os06g04200](http://rice.uga.edu/cgi-bin/ORF_infopage.cgi?orf=LOC_Os06g04200
- RAPdb: [Os06g0133000](https://rapdb.dna.affrc.go.jp/locus/?name=Os06g0133000)
**Look for splice mutations in first intron. Additionally, provided "Multiple promoter InDels/SNPs (such as ~23 bp insertions or deletions) modulate expression and amylose content, located roughly between Chr6:1,765,000–1,766,000, upstream of exon 1 in the Wx gene region ."

5' Flank: (1765622 - 1000) = 1764622 - 1766122 (1765622 + 500)
3' Flank: (1770656 - 500) = 1770156 - 1771656 (1770656 + 1000)


## Rc
7:6062889-6069317
- Symbol: Rc,qSD7-1,qPC7
- MSU: [LOC_Os07g11020](http://rice.uga.edu/cgi-bin/ORF_infopage.cgi?orf=LOC_Os07g11020)
- RAPdb: [Os07g0211500](https://rapdb.dna.affrc.go.jp/locus/?name=Os07g0211500)

5' Flank: (6062889 - 1000) = 6,061,889 -  6,063,389 = (6062889 + 500)
3' FLank: (6069317 - 500) = 6,068,817 - 6,070,317 = (6069317 + 1000)
# Other genes to check:
**Some extra from a Chat search. RC also from Dr. Edwards rec.**
!Rc/Os07g11020
Hd3a/Os06g0632
OsLG1/Os04g52370
GS3/Os03g0407400
GW5/qSW5/Os05g09520



### ZSS and SHAP comparisons for gene identification
Here I'll try and outline the workflow of the "zeroshot_v_shap_scatter.R" script and the analysis conducted. 

**Overall**: Input per tissue and chromosome CSVs that contain zeroshot score from PlantCaduecues, shap delta (CNN effect size for variant, e.g., hypo - actual), and variant meta data including position, reference, alternative, minor/major allele frequencies and output a single combined talbe "all_df".

Step 1: Unify the results
- read_one(tissue, chr) builds a file path from root directory, the tissue subfolder, and the naming pattern of files. If it exists adds two columns detailing tissue and corresponding chromosome
- Loop over all tissues and each chromosome and bind rows into on big table labeled "all_df"

Step 2: Adding features to aggregated dataframe
- add absolute values for
	- abs_zss - magnitude of zero-shot scores
	- abs_shap_delta - magnitude of CNN effects
	- shap_sign/zss_sign - direction of values
	- sign_agreement - binary, see if directions of signs are matching
- ==Are SHAP and ZSS pointing the same way variant by variant?==
- ==Are strong effects driven by magnitude rather than sign?==

Step 3: ZSS magnitude bins
Now for the absolute value of zeroshot scores, they are binned into different buckets
- What to gain?
	- How results distribute across effect size bands (are most variants small or are they many with large >10 values)
	- Future filtering about strength of zeroshot score

Step 4: Tissue specific SHAP
For each tissue, compute the 99.9th percentile of abs_shap_delta. Now we have a 0.1% cutoff within each tissue, to allow fair comparisons between tissues
Now we can also label the entries as being in the top 0.1% or not
- What to gain?
	- Which variants are extreme according to the CNN per tissue
	- Highlight the most influential SHAP values

Step 5: Orient ZSS sigh to match SHAP
For each tissue and chromosome, compute the sign of the Spearman correlation between shap_delta and zss. 