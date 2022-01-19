#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import copy
import multiprocessing as mp
from transformers import AutoTokenizer
import sentencepiece as spm

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--input-json", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    parser.add_argument("--spm", required=True, type=str)
    return parser

args = get_parser().parse_args()
tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
vocab = tokenizer.vocab
V = len(vocab.keys())
unk_id = tokenizer.unk_token_id

sp_model = spm.SentencePieceProcessor()
sp_model.load(args.spm)


with open(args.input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

def main(args):
    for u in data["utts"].keys():
        bertid, spm2wrd, bert2wrd = get_info(u)
        temp_dict = {"name": "target2bert", "tokenid": bertid}
        data["utts"][u]["output"].append(temp_dict)
        temp_dict = {"name": "spm2wrd", "tokenid": spm2wrd}
        data["utts"][u]["output"].append(temp_dict)
        temp_dict = {"name": "bert2wrd", "tokenid": bert2wrd}
        data["utts"][u]["output"].append(temp_dict)

    with open(args.output_json, "wb") as json_file:
        json_file.write(
                json.dumps(
                    data, indent=4, ensure_ascii=False, sort_keys=True
                ).encode("utf_8")
        )


def get_info(utt):
    txt = data["utts"][utt]["output"][-1]["text"]
    token = data["utts"][utt]["output"][-1]["token"]
    wrds = txt.split()
    spm_toks = sp_model.encode_as_pieces(txt)
    if " ".join(spm_toks) != token:
        import pdb;pdb.set_trace()

    wrd_toks = []
    spm2wrd = []
    bert2wrd = []
    bertid = []
    for i, w in enumerate(wrds):
        #get spm tokenization
        tmp = sp_model.encode_as_pieces(w)
        wrd_toks = wrd_toks + tmp
        spm2wrd = spm2wrd + [i for x in tmp]

        #get bert tokenization
        bert_toks = tokenizer.encode(w)
        bert_toks = bert_toks[1:-1]
        bertid = bertid + bert_toks
        bert2wrd = bert2wrd + [i for x in bert_toks]


    if spm_toks != wrd_toks:
        import pdb;pdb.set_trace()


    return " ".join([str(x) for x in bertid]), " ".join([str(x) for x in spm2wrd]), " ".join([str(x) for x in bert2wrd])

if __name__ == "__main__":
    main(args)
