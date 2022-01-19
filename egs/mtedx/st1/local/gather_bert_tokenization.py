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
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--ogtokens", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    return parser

args = get_parser().parse_args()
tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
vocab = tokenizer.vocab
V = len(vocab.keys())
unk_id = tokenizer.unk_token_id

tokens = {}

def main(args):
    with open(args.input, "r") as fin, open(args.output, "w") as fout, open(args.output+".noid", "w") as fout2:
        lines = fin.readlines()
        for l in lines:
            utt, txt, = l.strip().split(" ", 1)
            bert_tokenids = tokenizer.encode(txt)
            bert_tokens = tokenizer.convert_ids_to_tokens(bert_tokenids)
            
            fout2.write(" ".join(bert_tokens[1:-1]) + "\n")
            
            bert_tokens = ["▁"+t if t[0] != '#' else t[2:] for t in bert_tokens[1:-1]]

            fout.write(utt + " " + " ".join(bert_tokens) + "\n")

            for t in bert_tokens:
                if t in tokens:
                    tokens[t] += 1
                else:
                    tokens[t] = 1

    top_tokens = [x[0] for x in sorted(tokens.items(), key=lambda item: -item[1])[:1000]]
    with open(args.output+".top_tokens", "w") as f:
        for t in top_tokens:
            f.write(t + "\n")

    single_char_toks = set()
    with open(args.ogtokens, "r") as f:
        lines = f.readlines()
        for l in lines:
            t = l.strip().split()[0]
            if len(t) == 1 or (len(t)==2 and t[0]=="▁"):
                if t not in top_tokens:
                    single_char_toks.add(t)

    import pdb;pdb.set_trace()
    

if __name__ == "__main__":
    main(args)
