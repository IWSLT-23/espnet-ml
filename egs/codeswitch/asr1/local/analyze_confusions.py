#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--units", required=True, type=str)
    return parser

args = get_parser().parse_args()

dictionary = {}
with open(args.units, encoding="utf-8") as f:
    for l in f:
        try:
            symbol, val = l.strip().split()
        except:
            symbol = ""
            val = l.strip()
        if int(val) < 5000:
            dictionary[symbol.lower()] = 'en'
        else:
            dictionary[symbol] = 'zh'

if __name__ == "__main__":
    total = 0
    cross = 0
    zh_cnt = 0
    en_cnt = 0
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()[31:]  #double check that this is start of confusions
        for i, li in enumerate(lines):
            if "->" not in li:  #signals end of confusions
                break

            try:
                _, cnt, _, ref, _, hyp = li.strip().split()
            except:
                print(li)
                continue

            #count total confusions and count number of cross lang confusions
            total += int(cnt)
            if ref == "<blank>":
                continue
            if dictionary[ref] != dictionary[hyp]:
                cross += int(cnt)
                # print(li)
            elif dictionary[ref] == 'zh':
                zh_cnt += int(cnt)
            elif dictionary[ref] == 'en':
                en_cnt += int(cnt)

        print("Cross:", cross)
        print("Total:", total)
        print(cross/total)
        print("Zh:", zh_cnt)
        print(zh_cnt/total)
        print("En:", en_cnt)
        print(en_cnt/total)