#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

# Experiments for binarized Adult data. 
nice Rscript $DIR/code/experiment3.R ANN 100 10000 TRUE TRUE

# [Work in progress] Developed a more general file that works for more classifiers. 
#nice Rscript $DIR/code/MCCEalgo1.R randomForest 100 10000 TRUE TRUE
#nice Rscript $DIR/code/MCCEalgo1.R logreg 100 10000 TRUE TRUE

# Experiments for categorical Adult data. 
nice Rscript $DIR/code/experiment3.R ANN 100 10000 TRUE FALSE

# [Work in progress] Developed a more general file that works for more classifiers. 
#nice Rscript $DIR/code/MCCEalgo1.R randomForest 100 10000 TRUE FALSE
#nice Rscript $DIR/code/MCCEalgo1.R logreg 100 10000 TRUE FALSE
