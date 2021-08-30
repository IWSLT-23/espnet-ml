import argparse
import codecs
import json
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    textid_dict = {}

    for uttid in data["utts"]:
        textid_dict[uttid] = data["utts"][uttid]["output"][0]["tokenid"]

    with open(args.text, "r", encoding="utf-8") as f1, open(args.text+"id", "w", encoding="utf-8") as f2:
        lines = f1.readlines()
        for li in lines:
            uttid = li.split()[0]
            f2.write(uttid + " " + textid_dict[uttid] + "\n")