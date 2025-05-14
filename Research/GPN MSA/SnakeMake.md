
**Overview:**
Just some comments and notes on the SnakeMake workflow as the pipeline is new to me.
#Snakemake #Snakemakenotes
---
Snakemake appears as an alternative to GNU MakeFile and NextFlow workflows. Written similar to Python syntax but specialized by defining rules that describe how to build and read files.

1. Rules: Each rule will describe a single step within the pipline. It includes several parameters such as
- input
- outputs
- shell/run

2. Wildcards: Very similar to Python variable names but allow rules to generalize such as alternating file names or moving across chromosomes (e.g., {chrom}, {ref}, ect.)

3. Directed Acyclic Graphs (DAG): Unique to the Snakemake workflow, Snakemake will construct a dependency graph to learn the order of execution based on file relationships

4. Execution Control:
- Only runs the steps needed to make final target
- Uses timestamps to skip already compled subroutines
- Can parallelize and run on clusters (will be beneficial for Atlas and Slurm scheduler)
