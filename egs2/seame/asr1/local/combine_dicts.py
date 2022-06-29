#!/usr/bin/env python3
# -*- encoding: utf8 -*-

import os
import argparse

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_src", type=str)
    parser.add_argument("--man_src", type=str)
    parser.add_argument("--dst", type=str)
    args = parser.parse_args()


    en_lines = [x.strip() for x in open(args.en_src+"tokens.txt", "r").readlines()][2:-1]
    man_lines = ["▁" + x.strip() for x in open(args.man_src+"tokens.txt", "r").readlines()][2:-1]
    # slice blk, unk, and sos

    lines = en_lines + man_lines
    with open(args.dst, "w") as f:
        f.write("bpe_nlsyms=\""+",".join(man_lines)+"\"\n")
        f.write("man_nbpe="+str(len(man_lines) + 3))