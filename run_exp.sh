#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

# Experiments for binarized Adult data. Experiment 1 in the article. 
nice Rscript $DIR/code/MCCEalgo1.R ANN 100 10000 T T
nice Rscript $DIR/code/MCCEalgo1.R logreg 100 10000 T T

# Experiments for (categorized) Adult data. Experiment 2 in the article. 
