import argparse
import codecs
import json
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--uttlist", required=True)
    args = parser.parse_args()

    utts = []
    with open(args.uttlist, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            uttid = l.strip().split('-', 1)[1]
            utts.append(uttid)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    data_subset = {"utts":{}}

    count = 0
    for uttid in utts:
        data_subset["utts"][uttid] = data["utts"][uttid]
        count += 1

    with open(args.input+".perfect", "wb") as f:
        f.write(json.dumps(data_subset, indent=4, ensure_ascii=False, sort_keys=True).encode("utf-8"))

    print(count)