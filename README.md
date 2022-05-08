# deep_learning_for_healthcare_final_project

Citation to the original paper:
Barbieri, S., Kemp, J., Perez-Concha, O. et al. Benchmarking Deep Learning Architectures for Predicting Readmission to the ICU and Describing Patients-at-Risk. Sci Rep 10, 1111 (2020). \url{https://doi.org/10.1038/s41598-020-58053-z}


Citation to the original repo:
Barbieri, S., Kemp, J., Perez-Concha, O. et al. Benchmarking Deep Learning Architectures for Predicting Readmission to the ICU and Describing Patients-at-Risk.(2020). Github Repository. \url{https://github.com/sebbarb/time_aware_attention}


Dependencies:
torch, pandas, numpy, pickle, sklearn, scipy.stats, tqdm, pdb, os


Data Download Instructions:
https://eicu-crd.mit.edu/gettingstarted/access/. Follow instructions here to gain access.
https://physionet.org/content/mimiciii/1.4/. Dataset can be accessed here.


Preprocessing Code:
Change the directories in dl4h_final_project_preprocess.py to desired locations. Run file.

Training Code


Evaluation Code


Table of Results
| Model | Average Precision | AUROC | F1 | PPV | NPV | Sensitivity | Specificity | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|Logistic Regression |0.257 [0.248,0.265] |0.663 [0.66,0.667] |0.3 [0.296,0.304] |0.987 [0.969,1.005] |0.883 [0.882,0.885] |0.596 [0.586,0.607] |0.667 [0.656,0.678] |-- |
