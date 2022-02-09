import argparse
import codecs
import json
import logging
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add token gt using dict
    parser.add_argument("--input", required=True)
    parser.add_argument("--inter1", required=True)
    parser.add_argument("--inter2", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(args.inter1, "r", encoding="utf-8") as f:
        inter1 = json.load(f)

    with open(args.inter2, "r", encoding="utf-8") as f:
        inter2 = json.load(f)

    for u in data["utts"]:
        temp_dict = copy.copy(inter1["utts"][u]["output"][0])
        temp_dict["name"] = "target2"
        data["utts"][u]["output"].append(temp_dict)

        temp_dict = copy.copy(inter2["utts"][u]["output"][0])
        temp_dict["name"] = "target3"
        data["utts"][u]["output"].append(temp_dict)

    with open(args.input+".hier"+args.tag, "wb") as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False, sort_keys=True).encode("utf-8"))