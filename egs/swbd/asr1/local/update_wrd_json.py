#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import copy
import multiprocessing as mp

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--input-json", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    parser.add_argument("--units", required=True, type=str)
    parser.add_argument("--text-file", required=True, type=str)
    return parser

dictionary={}
args = get_parser().parse_args()


def main(args, token_dict, token_id_dict, shape_dict):

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
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


def get_wrd_seq(l):
    if len(l.strip().split(" ")) == 1:
        utt = l.strip().split(" ",1)[0]
        text= ""
    else:
        utt, text = l.strip().split(" ",1)

    wrds = ["▁"+w for w in text.split()]
    wrd_ids = [dictionary[w] if w in dictionary.keys() else str(-1) for w in wrds]
    shape = [len(wrds), len(dictionary) + 1]    #0, which is eps is included in list; +1 for eos
    return (utt, " ".join(wrds), " ".join(wrd_ids), shape)

if __name__ == "__main__":
    # Get dictionary
    with open(args.units, encoding="utf-8") as f:
        for l in f:
            val, symbol = l.strip().split(" ")
            dictionary[symbol] = val
    shape_dict = {}
    token_dict = {}
    token_id_dict = {}
    text_lines = []
    with open(args.text_file,encoding="utf-8") as f:
        for l in f:
            text_lines.append(l)

    with mp.Pool(processes = 20) as p:
        results = p.map(get_wrd_seq, text_lines)

    for result in results:
        utt = result[0]
        token_dict[utt] = result[1]
        token_id_dict[utt] = result[2]
        shape_dict[utt] = result[3]

    main(args,token_dict, token_id_dict, shape_dict)
