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
        new_vocab_size = (len(aid_tokens) * (data["utts"][uttid]["output"][0]["shape"][-1] - 2)) + 2
        shift_amt = aid_tokens[cat] * (data["utts"][uttid]["output"][0]["shape"][-1] - 2)

        data["utts"][uttid]["output"][0]["tokenid"] = " ".join(str(int(t) + shift_amt) for t in data["utts"][uttid]["output"][0]["tokenid"].split())
        
        shape = data["utts"][uttid]["output"][0]["shape"]
        shape[1] = new_vocab_size

    with open(args.input+".acc", "wb") as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False, sort_keys=True).encode("utf-8"))