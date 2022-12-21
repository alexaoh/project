# Explainable AI: A Comparison of Generative On-Manifold Methods for Counterfactual Explanations

This repository contains the code that was developed while working on the specialization project (TMA4500) with the aforementioned title. This README is intended to clarifiy what each directory and file in this repository contains. The repository can be represented with the following tree-structure: 


```bash
├── bash_scripts
│   ├── gen_uncond.sh
│   └── run_exp.sh
├── classifiers
│   ├── ANN_experiment3.h5
│   └── ANN_experiment4.h5
├── code
│   ├── adult_data_binarized.R: 
│   ├── adult_data_categ.R
│   ├── check_generation_distr.R
│   ├── check_trees.R
│   ├── compare_counterfact_MCCE_mod_MCCE.R
│   ├── experiment1.R
│   ├── experiment2.R
│   ├── experiment3.R
│   ├── experiment5.R
│   ├── feasibility.R
│   ├── fit_ANN_exp3.R
│   ├── fit_ANN_exp4.R
│   ├── investigate_data.R
│   ├── MCCE.R
│   ├── ModMCCE.R
│   ├── play_with_data.R
│   ├── random_forest.R
│   ├── test_sampling_from_end_node.R
│   ├── tests.R
│   ├── utilities.R
│   ├── VAE_custom_layer.R
│   └── VAE.R
├── data
│   ├── exp3_data
│   └── exp4_data
├── original_data
│   ├── adult.data
│   ├── adult.names
│   └── adult.test
├── README.md
├── results
│   ├── D_hs
│   └── Hs
└── resultsVAE
    └── D_hs

```

A short description of each of the directories is given in the following list: 

* *bash_scripts*: contains two shell scripts that were used for running some R scripts on an external computer. 
* *classifiers*: contains the two neural networks that were trained for Experiments 3 and 5, as well as Experiments 4 and 6, respectively. The ".h5" file is a Keras format that allows saving and loading pre-trained models, i.e. models with a certain architecture that have been fitted previously. 
* *code*: this contains all the R scripts that were used to produce the results in the Experiments. The most important scripts are:
    * adult_data_binarized.R: loads and pre-processes data in *original_data*. Saves the binarized data set to the hard drive. 
    * adult_data_categ.R: loads and pre-processes data in *original_data*. Saves the categorical data set to the hard drive. 
    * compare_counterfact_MCCE_mod_MCCE.R: used to make Tables 4.15 and 4.16. 
    * experiment*.R: these files contain code for each of the experiments in the report. Notice that "experiment3.R" was used for both Experiment 3 and 4, with different data. Similarly, "experiment5.R" was used for both Experiment 5 and 6. 
    * fit_ANN_exp3.R: fits the classifier for Experiments 3 and 5, and saves it to the hard drive. 
    * fit_ANN_exp4.R: fits the classifier for Experiment 4 and 6, and saves it to the hard drive. 
    * investigate_data.R: code to make density plots, mosaic plots and Q-Q plots. 
    * MCCE.R: almost complete general version of MCCE. This should be able to tackle more classifiers (not only neural nets), as well as different data sets. Thus, the idea was to use this same script for Experiment 3 and 4 (and other usecases later). Still a work in progress. 
    * ModMCCE.R: almost complete general version of Modifed MCCE. Not complete. 
    * utilities.R: contains some functions that were used in many of the other scripts. 
    
    The rest of the files are left in the directory, not because they necessarily were essential in producing the final results, but because they might come in handy in the future. 
* *data*: contains binarized and categorical data saved in .RData files. Also contains data splits and normalization constants from Experiments 3 to 6. 
* *original_data*: contains the Adult data files that were downloaded from UCI Machine Learning Repository. 
* *results*: contains results from the experiments concerning MCCE. Also contains $H$, $D_h$ and the final counterfactuals from Experiments 3 and 4, as well as results from Experiment 1. 
* *resultsVAE*: contains results from the experiments concerning Modified MCCE. Also contains $D_h$ and the final counterfactuals from Experiments 5 and 6. 
