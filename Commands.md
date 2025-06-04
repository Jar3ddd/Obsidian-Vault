
#Ceres
Getting a compute node: salloc -n 1 -p ceres -t 4:00:00

---
#Atlas
View queued jobs (with full name): #squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me

**Modules**
#bedtools Find overlapping features between genomic datasets (.bed, .gff, and .vcf). Compares intervals between files and returns pairs that overall. 
- bedtools intersect -a features.bed -b variances.vcf.gz -wa -wb > output_file.tsv
- Get original sequences: bedtools getfasta -fi fileName.fasta -bed gene_flanks.bed -s -name -fo (>) output_fileName.fasta
- bedtools getfasta -fi 
#samtools Create indexed reference file (from fasta). Also can be obtained from Ensembl (.fai)
- samtools faidx reference_fileName.fasta
#bedfiles Create .bed file for easy read
- awk -F'\t' 'BEGIN{OFS="\t"} $3 == "gene" {match($9, /ID=([^;]+)/, a); print $1, ($4-1500<0?0:$4-1500), $5+1500, a[1]}' annotation_file.gff3 > gene_flanks.bed
- bedtools getfasta -fi (Input fasta .fa) -bed (ranges to extract from -fi) -name -fo (name of file)
#bcftools 
- Index the variant information: bcftools index variant_callFile.vcf.gz



---
#WindowsSubsystem
Starting a Linux job: wsl --distribution
Search home dir.: find ~ -type f -iname "*example*"