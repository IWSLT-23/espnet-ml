#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from jiwer import wer
from statistics import mean

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--hyp", required=True, type=str)
    parser.add_argument("--ref", required=True, type=str)
    return parser

args = get_parser().parse_args()

if __name__ == "__main__":
    zh_cnt = 0
    en_cnt = 0
    cs_points_cnt = {}
    uttid_to_cspoints = {}
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for li in lines:
            cs_points = 0
            toks = li.strip().split()[1:]
            for t in toks:
                if int(t) == 11237 or int(t) == 11238:
                    cs_points = cs_points + 1
                elif int(t) >= 5000:
                    zh_cnt += 1
                elif int(t) < 5000:
                    en_cnt += 1
                else:
                    import pdb;pdb.set_trace()
            cs_points = cs_points - 1
            if cs_points in cs_points_cnt.keys():
                cs_points_cnt[cs_points] = cs_points_cnt[cs_points] + 1
            else: 
                cs_points_cnt[cs_points] = 1
            uttid = li.strip().split()[0]
            uttid_to_cspoints[uttid] = cs_points

        total = zh_cnt + en_cnt

        print("Zh:", zh_cnt)
        print(zh_cnt/total)
        print("En:", en_cnt)
        print(en_cnt/total)

        print(cs_points_cnt)

    with open(args.ref, "r", encoding="utf-8") as f:
        ref_lines = f.readlines()
    with open(args.hyp, "r", encoding="utf-8") as f:
        hyp_lines = f.readlines()

    mer_dict = {}
    for c in cs_points_cnt.keys():
        mer_dict[c] = []

    for r, h, in zip(ref_lines, hyp_lines):
        try:
            r, uttid = r.strip().rsplit(' ', 1)
            h, _ = h.strip().rsplit(' ', 1)
        except:
            continue

        mer = wer(r, h)
        uttid = uttid[1:-1].split("-", 1)[-1]

        mer_dict[uttid_to_cspoints[uttid]].append(mer)

    # for c in mer_dict.keys():
    #     print(c, mean(mer_dict[c]))
    print("1", mean(mer_dict[1]))
    print("2", mean(mer_dict[2]))
    print("3+", mean(mer_dict[3] + mer_dict[4] + mer_dict[5]))