#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from jiwer import wer

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--ref", required=True, type=str)
    parser.add_argument("--hyp", required=True, type=str)
    return parser

args = get_parser().parse_args()

import numpy as np

if __name__ == "__main__":
    with open(args.ref, "r", encoding="utf-8") as f:
        ref_lines = f.readlines()
    with open(args.hyp, "r", encoding="utf-8") as f:
        hyp_lines = f.readlines()

    for r, h, in zip(ref_lines, hyp_lines):
        try:
            r, uttid = r.strip().rsplit(' ', 1)
            h, _ = h.strip().rsplit(' ', 1)
        except:
            continue
        mer = wer(r, h)
        if mer == 0:
            print(uttid[1:-1])
        # import pdb; pdb.set_trace()
