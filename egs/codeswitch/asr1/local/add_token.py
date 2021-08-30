import argparse
import codecs
import json
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add token gt using dict
    parser.add_argument("--input", required=True)
    parser.add_argument("--dict", required=True)
    args = parser.parse_args()

    units = {}
    with open(args.dict, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            u, idx = l.strip().split(' ', 1)
            units[idx] = u

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    for uttid in data["utts"]:
        tokids = data["utts"][uttid]["output"][0]["tokenid"].split()
        toks = []
        for id in tokids:
            if id == '0':
                continue
            else:
                toks.append(units[id])
        toks = " ".join(toks)
        data["utts"][uttid]["output"][0]["token"] = toks

    with open(args.input, "wb") as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False, sort_keys=True).encode("utf-8"))