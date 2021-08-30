import argparse
import codecs
import json
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add token gt using dict
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    for uttid in data["utts"]:
        
        if "K190" in uttid:
            lid = "en"
        else:
            tmp = uttid.split("_")
            if len(tmp) == 4:
                lid = "zh"
            elif len(tmp) == 6:
                lid = "cs"
            else: 
                import pdb;pdb.set_trace()
                
        data["utts"][uttid]["category"] = lid

    with open(args.input+".cat", "wb") as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False, sort_keys=True).encode("utf-8"))