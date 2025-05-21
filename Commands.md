
#Ceres
Getting a compute node: salloc -n 1 -p ceres -t 4:00:00

---
#Atlas
View queued jobs (with full name): #squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me

**Modules**
#bedtools Find overlapping features between genomic datasets (.bed, .gff, and .vcf). Compares intervals between files and returns pairs that overall. 
- bedtools intersect -a features.bed -b variances.vcf.gz -wa -wb > output_file.tsv
