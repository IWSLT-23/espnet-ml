import argparse
import codecs
import json
import logging
import copy
from itertools import groupby

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add token gt using dict
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    data = {"utts":{}}
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            uttid, tokstr = l.strip().split(" ", 1)
            # shift by 1 to make it compatible with 0 for blk
            tokints = [int(x) + 1 for x in tokstr.split()]
            toks = [x[0] for x in groupby(tokints)]
            temp_dict = {}
            temp_dict["tokenid"] = " ".join([str(x) for x in toks])
            temp_dict["shape"] = [len(toks), 501]
            data["utts"][uttid] = {"output":[temp_dict]}

    with open(args.input+".json", "wb") as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False, sort_keys=True).encode("utf-8"))