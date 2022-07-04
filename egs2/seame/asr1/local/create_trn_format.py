#!/usr/bin/env python3
# -*- encoding: utf8 -*-

import os
import argparse

alph = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
alph_lower = [x.lower() for x in alph]
num = ["0","1","2","3","4","5","6","7","8","9"]
eng_set = alph + alph_lower + num

def lid(c):
    if c[0] in eng_set:
        return "<en>"
    else:
        return "<zh>"

def get_masks(txt):
    en_only = []
    zh_only = []
    lid_only = []
    all_nolid = []
    for i, c in enumerate(txt.split()):
        if c == "<en>" or c == "<zh>":
            lid_only.append(c)
            continue
        elif lid(c) == "<en>":
            en_only.append(c)
        elif c != "<noise>":
            zh_only.append(c)
        all_nolid.append(c)
    return " ".join(en_only), " ".join(zh_only), " ".join(lid_only), " ".join(all_nolid)
        

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp", type=str)
    parser.add_argument("--ref", type=str)
    args = parser.parse_args()

    hyp_lines = [get_masks(x.strip()) for x in open(args.hyp, "r").readlines()]
    ref_lines = [get_masks(x.strip()) for x in open(args.ref, "r").readlines()]

    assert len(hyp_lines) == len(ref_lines)

    with open(args.hyp+".en", "w") as h_en, \
        open(args.hyp+".zh", "w") as h_zh, \
        open(args.hyp+".lid", "w") as h_lid, \
        open(args.hyp+".all", "w") as h_all, \
        open(args.ref+".en", "w") as r_en, \
        open(args.ref+".zh", "w") as r_zh, \
        open(args.ref+".lid", "w") as r_lid, \
        open(args.ref+".all", "w") as r_all:
        for i in range(len(hyp_lines)):
            hyp = hyp_lines[i]
            ref = ref_lines[i]

            h_en.write(hyp[0] + "\t(0_"+str(i)+")\n")
            h_zh.write(hyp[1] + "\t(0_"+str(i)+")\n")
            h_lid.write(hyp[2] + "\t(0_"+str(i)+")\n")
            h_all.write(hyp[3] + "\t(0_"+str(i)+")\n")

            r_en.write(ref[0] + "\t(0_"+str(i)+")\n")
            r_zh.write(ref[1] + "\t(0_"+str(i)+")\n")
            r_lid.write(ref[2] + "\t(0_"+str(i)+")\n")
            r_all.write(ref[3] + "\t(0_"+str(i)+")\n")
