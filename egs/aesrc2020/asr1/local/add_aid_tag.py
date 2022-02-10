import argparse
import codecs
import json
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add token gt using dict
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    aid_tokens = {'<US>':0, '<CHN>':1, '<PT>':2, '<ES>':3, '<JPN>':4, '<IND>':5, '<KR>':6, '<UK>':7, '<RU>':8, '<CA>':9}

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    for uttid in data["utts"]:
        utt_tok =  uttid.split("-")
        if utt_tok[0][:2] == 'sp':
            cat = utt_tok[1]
        else:
            cat = utt_tok[0]

        cat = "<"+cat+">"

        data["utts"][uttid]["output"][0]["token"] = cat + " " + data["utts"][uttid]["output"][0]["token"]
        aid_tokenid = str(data["utts"][uttid]["output"][0]["shape"][-1] + aid_tokens[cat] - 1)
        data["utts"][uttid]["output"][0]["tokenid"] = aid_tokenid + " " + data["utts"][uttid]["output"][0]["tokenid"]
        
        shape = data["utts"][uttid]["output"][0]["shape"]
        shape[0] += 1
        shape[1] += len(aid_tokens)

    with open(args.input+".aid", "wb") as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False, sort_keys=True).encode("utf-8"))