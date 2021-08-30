#!/bin/bash
FILES="posterior_line_chart/posts/*.npy"
for f in $FILES
do
    echo "Processing $f file..."
    # take action on each file. $f store current file name
    python local/plot_posteriors.py --input "$f" --units data/lang/10k_units.txt
done
