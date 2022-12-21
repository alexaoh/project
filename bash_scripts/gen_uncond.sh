#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

nice Rscript $DIR/code/experiment1.R bin
nice Rscript $DIR/code/experiment1.R cat
