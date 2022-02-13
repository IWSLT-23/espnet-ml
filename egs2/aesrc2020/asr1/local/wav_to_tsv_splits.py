#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import numpy as np
import math

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--wav", required=True, type=str)
    parser.add_argument("--splits", required=True, type=int)
    parser.add_argument("--tsvdir", required=True, type=str)
    return parser

args = get_parser().parse_args()

def main(args):
    with open(args.wav, "r") as f:
        lines = f.readlines()
        n = len(lines) // args.splits
        print(n)
        tsv = {x:['.'] for x in range(args.splits+1)}
        for i, l in enumerate(lines):
            split = i // n
            uttid, wav = l.strip().split()
            tsv[split].append(wav+"\t"+"1")

    for split in tsv.keys():
        with open(args.tsvdir+"/"+str(split)+".tsv", "w") as f:
            f.write("\n".join(tsv[split]))

if __name__ == "__main__":
    main(args)

