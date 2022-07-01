#!/usr/bin/env python3
# -*- encoding: utf8 -*-

import os
import argparse

#sometimes noise token is not segmented in the transcript

def fix(id, txt):
    return id + " " + " ".join(" <noise> ".join(txt.split("<noise>")).split()) + "\n"


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    args = parser.parse_args()


    lines = [x.strip().split(" ", 1) for x in open(args.src, "r").readlines()]
    new_lines = [fix(id, txt) for id, txt in lines]
    
    with open(args.src+"2", "w") as f:
        f.writelines(new_lines)
