The following is meant as a guide for prediction on raw sequences, retrieved and processed from the Ensembl database. This pipeline will serve for predicting on unseen sequences of currently trained single species CNN models. Additionally, all sequences are processed during retrieval and meet the criteria in terms of content and shape for maximal model performance. Likewise, during prediction, genes located on the same chromosome are grouped together for predictions on the appropriately trained models. The following guide is meant for use on the SCINet's HPC Atlas, though can be ran using any system that can support Tensorflow v.2.10.0.

#### Using Trained Models: A Short Guideline
1. Install necessary dependencies: To run the scripts, please use the already prepared Python environment backup. The environment will contain necessary packages, dependencies, and utility used in the original DeepCRE training methodology.
	1. Environment Location: /project/dbnrrc/jared/env_backups/tensorflowV2_backup.yml
	2. Use: "conda env create --prefix {desired location of env} --file tensorflowV2_backup.yml"
2. After installing dependencies: Moving the saved models is **not** necessary; however, if you wish to do predictions on personal or differing project directory, notes on how to do so are below.
	1. Moving models to desired working directory:
		1. All trained models will be stored in a singular directory (i.e., /project/dbnrrc/jared/research/DeepCRE-CUDA/model_reimplemented/deepCRE_reimplemented/src/deepCRE/saved_models)
			*Note: if previous models are not found please refer to 90daydata!*
		2. Saved Models denote the trained network where the number in the naming convention denotes the validation chromosomes used. For proper use (if testing with personal script), depending on what chromosome the gene is on, is what model should be utilized.
3. Utilizing sequence retrieval and sequences prediction: Once models have been identified, you may then begin the sequence retrieval processed and start predicting! 
	1. Sequence Retrieval - seq_retrieval_v2.0.py
		1. Sequence retrieval supports both single and multiple gene IDs. For the desired prediction the model will expect a particular shape. These shapes are defined by the batches (no. of sequences), length (length of sequences), and features (one-hot encoded nucleotides). Sequences are comprised of 1500bp from surrounding flanking regions and includes 1000bp (500 + 500) of either side of the coding region and 2000bp (1000 + 1000) of the promoter terminator regions. Finally, between these two 20 appended "N"s are added, resulting in sequences of length 3020.
		2. Once gene IDs are provided to the list found within the script, the raw reads will be saved in the same directory as a FASTA file titled "retrieved_flanks.fa"
	2. Sequence Prediction - get_preds_v2.0.py
		1. I recommend running predictions via terminal; however, in case of multitasking or large count of gene IDs, the script can be ran sing SLURM scheduler. 
		2. To run predictions on the newly acquired sequences reads. The new prediction script requires a three parameters be set during running. During execution, please specify FASTA (file from sequence retrieval), model directory (saved models location), and output (desired location of prediction outputs). **Additionally, please ensure that model prefix/suffix conventions are correct!**. During training models have additional information encoded into the name which includes not only chromosome number but date of training as well as data type. These are apparent from the trained models but need to be adjusted. All changes can be made or specified using the parser as shown below (--fasta, --mode_dir, --output, --model_prefix/suffix)
			==Example: python3 get_preds_v2.0.py  --fasta ../genome –-output FILE_OUTPUT.csv==

Current directory structure is provided below on where scripts can be found as well as how to script defines its default parameters:
└── jared/
    └── research/
        └── DeepCRE-CUDA/
            └── model_reimplemented/
                └── deepCRE_reimplemented/
                    └── src/
                        └── deepCRE/
                            ├── seq_prediction/
                            │   ├── get_preds_v2.0.py
                            │   └── predictionOUTPUT.csv
                            ├── saved_models/
                            └── seq_retrieval/
                                ├── seq_retrieval_v2.0.py
                                └── retrieved_flanks.fa
