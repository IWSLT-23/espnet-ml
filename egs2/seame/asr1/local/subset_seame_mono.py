#!/usr/bin/env python3
# -*- encoding: utf8 -*-

import os
import argparse

cnt = 0

def is_mono(txt):
    for i, c in enumerate(txt.split()):
        if i > 0 and (c == "<zh>" or c == "<en>"):
            return False
    global cnt
    cnt += 1
    return True

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dst", type=str)
    args = parser.parse_args()


    lines = [x.strip().split(" ", 1) for x in open(args.src+"/text", "r").readlines()]
    new_lines = [id + ' ' + txt + "\n" for id, txt in lines if is_mono(txt)]
    mono_ids = [x.split(" ")[0] for x in new_lines]
    
    wavs = {id:wav for id,wav in [x.strip().split(" ", 1) for x in open(args.src+"/wav.scp", "r").readlines()]}
    # new_wav_lines = [id + ' ' + txt + "\n" for id, txt in lines if id in wav_lines]
    
    with open(args.dst+"/text", "w") as f:
        f.writelines(new_lines)

    with open(args.dst+"/wav.scp", "w") as f:
        for id in mono_ids:
            f.write(id + " " + wavs[id] + "\n")

    durs = {id:dur for id,dur in [x.strip().split(" ", 1) for x in open(args.src+"/utt2dur", "r").readlines()]}

    with open(args.dst+"/utt2dur", "w") as f:
        for id in mono_ids:
            f.write(id + " " + durs[id] + "\n")

    print(cnt)