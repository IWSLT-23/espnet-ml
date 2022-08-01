#!/usr/bin/env python3
# -*- encoding: utf8 -*-

import os
import argparse
import numpy as np

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dst", type=str)
    parser.add_argument("--nj", type=int)
    args = parser.parse_args()

    all_posts = {}
    for i in range(1, args.nj+1):
        split_posts = np.load(args.src+"/output."+str(i)+"/post.npz", allow_pickle=False)
        for k in split_posts.files:
            all_posts[k] = split_posts[k]

    print(len(all_posts.keys()))
    np.savez(args.dst, **all_posts)