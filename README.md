# deep_learning_for_healthcare_final_project

### Citation to the original paper:
Barbieri, S., Kemp, J., Perez-Concha, O. et al. Benchmarking Deep Learning Architectures for Predicting Readmission to the ICU and Describing Patients-at-Risk. Sci Rep 10, 1111 (2020). https://doi.org/10.1038/s41598-020-58053-z


### Citation to the original repo:
Barbieri, S., Kemp, J., Perez-Concha, O. et al. Benchmarking Deep Learning Architectures for Predicting Readmission to the ICU and Describing Patients-at-Risk.(2020). Github Repository. https://github.com/sebbarb/time_aware_attention


## Dependencies:
torch, pandas, numpy, pickle, sklearn, scipy.stats, tqdm, pdb, os


## Data Download Instructions:
https://eicu-crd.mit.edu/gettingstarted/access/. Follow instructions here to gain access.
https://physionet.org/content/mimiciii/1.4/. Dataset can be accessed here.


## Preprocessing Code:
Change the directories in dl4h_final_project_preprocess.py (lines 8-9) to desired locations on your local machine. Run the file.

## Training Code
Change (uncomment desired model and comment others) to desired model in hyperparameters.py (lines 30-42). Run **train.py.**
For example, if you wish to run the RNN+ODE model, comment line 36 in the code, and then uncomment line 34.

## Evaluation Code
Change (uncomment desired model and comment others) to desired model in hyperparameters.py, same as above. **Run test.py.**

## Table of Results
| **Model** | **Average Precision** | **AUROC** | **F1** | **PPV** | **NPV** | **Sensitivity** | **Specificity** | **Time** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|**Logistic Regression** |0.257 [0.248,0.265] |0.663 [0.66,0.667] |0.3 [0.296,0.304] |0.987 [0.969,1.005] |0.883 [0.882,0.885] |0.596 [0.586,0.607] |0.667 [0.656,0.678] |0.244 |
|**Attention** |0.294 [0.286,0.303] |0.72 [0.717,0.723] |0.344 [0.34,0.349] |0.943 [0.91,0.976] |0.883 [0.881,0.884] |0.653 [0.643,0.663] |0.683 [0.674,0.692] |30.857 [26.985,34.728] std: 17.288 |
|**ODE+RNN** |0.321 [0.313,0.329] |0.743 [0.741,0.746] |0.373 [0.369,0.378] |0.989 [0.974,1.004] |0.883 [0.882,0.884] |0.69 [0.678,0.701] |0.69 [0.678,0.701] |855.108 [827.372,882.843] std: 123.851 |
|**ODE+Attention** |0.291 [0.282,0.299] |0.706 [0.703,0.71] |0.335 [0.33,0.34] |0.954 [0.927,0.981] |0.883 [0.882,0.885] |0.64 [0.627,0.654] |0.67 [0.656,0.684] |858.453 [855.227,861.68] std: 14.41 |
|**ODE+RNN+Attention** |0.302 [0.294,0.311] |0.729 [0.726,0.731] |0.357 [0.352,0.361] |0.97 [0.945,0.994] |0.883 [0.882,0.884] |0.729 [0.723,0.735] |0.643 [0.639,0.648] |641.879 [639.578,644.18] std: 10.274 |
|**MCE+Attention** |0.275 [0.266,0.284] |0.689 [0.686,0.692] |0.317 [0.313,0.322] |0.958 [0.931,0.986] |0.883 [0.882,0.884] |0.683 [0.67,0.697] |0.615 [0.601,0.628] |18.054 [17.785,18.322] std: 1.2 |
|**MCE+RNN** |0.302 [0.294,0.31] |0.729 [0.726,0.732] |0.362 [0.357,0.367] |0.974 [0.954,0.994] |0.883 [0.882,0.884] |0.667 [0.655,0.68] |0.692 [0.679,0.705] |297.868 [292.659,303.077] std: 23.26 |
|**MCE+RNN+Attention** |0.315 [0.306,0.324] |0.732 [0.729,0.735] |0.362 [0.358,0.367] |1.0 [nan,nan] |0.884 [0.883,0.885] |0.683 [0.671,0.694] |0.687 [0.675,0.698] |313.617 [307.684,319.549] std: 26.491 |
|**RNN (ODE time decay)+Attention** |0.32 [0.311,0.329] |0.74 [0.738,0.743] |0.363 [0.358,0.368] |0.923 [0.893,0.953] |0.884 [0.882,0.885] |0.684 [0.669,0.7] |0.684 [0.667,0.701] |946.473 [945.496,947.449] std: 4.36 |
|**RNN (ODE time decay)** |0.313 [0.306,0.32] |0.743 [0.74,0.746] |0.372 [0.367,0.377] |0.968 [0.953,0.983] |0.882 [0.881,0.883] |0.696 [0.687,0.705] |0.692 [0.683,0.7] |1494.197 [1455.534,1532.86] std: 172.645 |
|**RNN (exp time decay)+Attention** |0.307 [0.3,0.315] |0.742 [0.739,0.745] |0.373 [0.368,0.377] |0.972 [0.951,0.993] |0.881 [0.88,0.882] |0.646 [0.635,0.656] |0.728 [0.718,0.737] |917.87 [916.992,918.747] std: 3.919 |
|**RNN (exp time decay)** |0.292 [0.286,0.299] |0.734 [0.731,0.737] |0.367 [0.362,0.371] |0.981 [0.968,0.995] |0.881 [0.88,0.883] |0.725 [0.716,0.735] |0.652 [0.644,0.661] |911.051 [904.413,917.689] std: 29.642 |
|**RNN (concat Δtime)+Attention** |0.299 [0.292,0.307] |0.738 [0.735,0.741] |0.356 [0.352,0.361] |0.929 [0.903,0.956] |0.882 [0.881,0.883] |0.709 [0.697,0.721] |0.652 [0.64,0.663] |171.286 [170.583,171.988] std: 3.138 |
|**RNN (concat Δtime)** |0.3 [0.293,0.308] |0.734 [0.732,0.737] |0.363 [0.359,0.368] |0.984 [0.968,1.0] |0.881 [0.88,0.883] |0.65 [0.64,0.66] |0.718 [0.706,0.729] |164.282 [163.412,165.151] std: 3.884 |
*unlike the original paper, we included times in our results to help those who are interested in reproducing the results have an idea of how long each model will take to run*
