
## Results
### Model Comparisons  
Species model accuracy (w/ Rice): Leaf Models
![[CNN_Leaf_Model#sheet1|0-10:0-4]]

---

Species model accuracy (w/ Rice): Root Models
![[CNN_Root_Model#sheet1|0-5:0-4]]



**Model Averages**:
Model performance was assessed by comparing the average classification accuracy across five plant species Arabidopsis thaliana, Solanum lycopersicum, Sorghum bicolor, Zea Mays, and Oryza sativa japonica. For each species, leave-one-out chromosome models were trained used either leaf derived and root derived expression data. Below are the summarized mean accuracy for each combination of species and model type.

![[averagesPlot.pdf]]
**Summary Table**

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
Additionally, from training after evaluating models on the twelve Oryza sat. (jap.) chromosomes, our model achieved a a mean accuracy of 80%, with test-fold accuracies ranging from 0.765 to 0.830. The corresponding auROC values also varied between 0.838 and 0.901, and auPR between 0.841 and 0.900. Fold three provided the best performance (loss = .0401, accuracy = 0.830, auROC = 0.901, auPR = 0.841). Overall, the standard deviation between the chromosome models indicates that the rice model generalized nicely, while also sustaining high discriminative power (auROC > 0.84) and predictive accuracy (~0.80-0.82) regardless of specific training and validation chromosomes.

| test | loss        | accuracy    | auROC       | auPR        |
|------|-------------|-------------|-------------|-------------|
| 1    | 0.42757535  | 0.801787615 | 0.886884809 | 0.88533169  |
| 2    | 0.435768336 | 0.813443065 | 0.882233381 | 0.869498551 |
| 3    | 0.401403308 | 0.830393493 | 0.901426613 | 0.900353849 |
| 4    | 0.426205754 | 0.807817578 | 0.885587633 | 0.890048265 |
| 5    | 0.419906467 | 0.813543618 | 0.89200604  | 0.881481051 |
| 6    | 0.421861082 | 0.813636363 | 0.890125573 | 0.880331099 |
| 7    | 0.452201724 | 0.785580516 | 0.874365211 | 0.87488687  |
| 8    | 0.433509231 | 0.807851255 | 0.887381732 | 0.889001369 |
| 9    | 0.444549352 | 0.795336783 | 0.878305495 | 0.88780582  |
| 10   | 0.476812869 | 0.767647088 | 0.858840883 | 0.857477069 |
| 11   | 0.48156473  | 0.768488765 | 0.851252556 | 0.859113455 |
| 12   | 0.49644208  | 0.765027344 | 0.838234186 | 0.841009974 |
==add root table too==

**Prediction Results**
To evaluate the Oryza sativa (jap.) leaf and root model's self prediction performance, we applied it to the same genome it was trained on and calculated the prediction accuracy using genes with defined expression labels (excluding those with class 2, which were skipped during training). Among the 14,273 valid genes, the model achieve the following:
- **Accuracy**: 80.2%
- **auROC**: 0.880
- **F1-score:**
	- High expression (class 1): 0.804
	- Low expression (class 2): 0.801
Confusion matrix (Leaf):

|            | Predicted: Low | Predicted: High |
| ---------- | -------------- | --------------- |
| True: Low  | 5,674          | 1,288           |
| True: High | 1,534          | 5,777           |
These results indicated balanced precision and recall across both classes. The high auROC of 0.88 confirms the model's ability to rank high and low expression genes, while the F1-score suggests comparable performance for either. The model does not display any major bias towards predicting either class. (==could add reasoning and importance==)

Furthermore, the root model slightly outperformed the leaf model in both auROC (0.890 v. 0.880) and F1-score for the high expression class (0.829 v. 0.804). It also maintains consistent balanced precision recall tradeoff with higher precision for high expression genes (0.855) and higher recall for low expression genes (0.821). This further confirms that the model has been tuned for deticting biologically relevant expression patterns in both root and leaf tissues. 
- **Accuracy**: 81.2%
- **auROC**: 0.890
- **F1-score**:
	- High expression (class 1): 0.829
	- Low expression (class 0): 0.791

|            | Predicted: Low | Predicted: High |
| ---------- | -------------- | --------------- |
| True: Low  | 4,505          | 979             |
| True: High | 1,408          | 5,772           |


In addition to doing self prediction, we also computed prediction labels for the four other model species. To conduct this, we used (**reference R script**) to find the total count of high or low expressed genes. By taking the average of true high expression genes of the total, we were able to compare proportions by labeling the predictions using the predicted probabilities. For predicted probabilities $\leq .5$ predicted labels were given zero (0) whereas the remaining were denoted one (1). To reduce inflating error we also removed "mild" expression or true labels equal to two (2). (==May rewrite json so that only the corresponding chromomes are used from rice. (05/27) all chromosomes are being used, potentially leading to worse results==). Interestingly, the rice model was more accurate when predicting on other monocots such as Sorghum Bicolor and Zea Mays. 

|     | species | count | mean_true | mean_pred_prob | std_pred_prob | accuracy |
| --- | ------- | ----- | --------- | -------------- | ------------- | -------- |
| 1   | ara     | 10788 | 0.595569  | 0.368455       | 0.160186      | 0.50723  |
| 2   | sbic    | 14519 | 0.550313  | 0.461955       | 0.311205      | 0.785522 |
| 3   | sol     | 13269 | 0.578491  | 0.387227       | 0.165406      | 0.564097 |
| 4   | zea     | 15621 | 0.603547  | 0.416534       | 0.287085      | 0.700211 |


---
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