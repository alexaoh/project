#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

nice Rscript $DIR/code/test_non_conditional_sampling_from_trees.R bin
nice Rscript $DIR/code/test_non_conditional_sampling_from_trees.R cat
