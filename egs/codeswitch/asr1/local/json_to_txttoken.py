import argparse
import codecs
import json
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add token gt using dict
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(args.output, "w", encoding="utf-8") as f:
        for uttid in data["utts"]:
            toks = data["utts"][uttid]["output"][0]["token"]
            f.write(toks+"\n")