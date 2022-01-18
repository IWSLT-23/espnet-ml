#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import copy
import multiprocessing as mp
from transformers import AutoTokenizer

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--input-json", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    return parser

args = get_parser().parse_args()
tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
vocab = tokenizer.vocab
V = len(vocab.keys())
unk_id = tokenizer.unk_token_id

with open(args.input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

def main(args, token_dict, token_id_dict, shape_dict):
    for u in token_dict.keys():
        temp_dict = copy.copy(data["utts"][u]["output"][0])
        temp_dict["name"] = "target2"
        temp_dict["token"] = token_dict[u]
        temp_dict["tokenid"] = token_id_dict[u]
        temp_dict["shape"] = shape_dict[u]
        data["utts"][u]["output"].append(temp_dict)

    with open(args.output_json, "wb") as json_file:
        json_file.write(
                json.dumps(
                    data, indent=4, ensure_ascii=False, sort_keys=True
                ).encode("utf_8")
        )


def get_bert_seq(utt):
    output = data["utts"][utt]["output"][0]
    bert_token = [s[1:] if s[0] == '▁' else '##'+s for s in output['token'].split()]
    #try:
    bert_tokenid = [str(vocab[t]) if t in vocab else str(unk_id) for t in bert_token]
    #except:
    #    import pdb;pdb.set_trace()
    shape = (len(bert_token), V)
    return (utt, " ".join(bert_token), " ".join(bert_tokenid), shape)

if __name__ == "__main__":
    with mp.Pool(processes = 20) as p:
        results = p.map(get_bert_seq, data["utts"].keys())
    #for u in data["utts"].keys():
    #    get_bert_seq(u)

    token_dict = {}
    token_id_dict = {}
    shape_dict = {}
    for result in results:
        utt = result[0]
        token_dict[utt] = result[1]
        token_id_dict[utt] = result[2]
        shape_dict[utt] = result[3]

    main(args,token_dict, token_id_dict, shape_dict)
