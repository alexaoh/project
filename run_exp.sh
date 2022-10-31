#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

# Experiments for binarized Adult data. Experiment 1 in the article. 
nice Rscript $DIR/code/experiment3.R ANN 100 10000 TRUE TRUE
#nice Rscript $DIR/code/MCCEalgo1.R randomForest 100 10000 TRUE TRUE
#nice Rscript $DIR/code/MCCEalgo1.R logreg 100 10000 TRUE TRUE

# Experiments for (categorized) Adult data. Experiment 2 in the article. 
#nice Rscript $DIR/code/MCCEalgo1.R randomForest 100 10000 TRUE FALSE
#nice Rscript $DIR/code/MCCEalgo1.R ANN 100 10000 TRUE FALSE 
#nice Rscript $DIR/code/MCCEalgo1.R logreg 100 10000 TRUE FALSE

# Experiments for binarized Adult data, with random forest. Experiment 3 in the article. 
#nice Rscript $DIR/code/MCCEalgo1.R randomForest 100 10000 TRUE TRUE 
