# get tokenids from previous dataprep
import argparse
import codecs
import json
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--text")
parser.add_argument("--src")
parser.add_argument("--dst")
parser.add_argument("--output_dim", required=True, type=int)
parser.add_argument("--nbpe", default=5000, type=int)
args = parser.parse_args()

def add_lid(toks):
    toks = [int(x) for x in toks.split()]
    new_txt = ''
    prev_lid = args.output_dim - 2 if toks[0] > args.nbpe else args.output_dim - 1
    new_toks = [str(prev_lid)]
    for t in toks:
        # <ch> is -2 and <en> is -1
        curr_lid = args.output_dim - 2 if t > args.nbpe else args.output_dim - 1
        if curr_lid != prev_lid:
            new_toks.append(str(curr_lid))
            prev_lid = curr_lid
        new_toks.append(str(t))
    new_txt = ' '.join(new_toks)
    return new_txt

if __name__ == "__main__":


    utts = []
    with open(args.text, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            uttid, txt = l.split(' ', 1)
            utts.append(uttid)

    tokenid_dict = {}
    with open(args.src, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            try:
                uttid, tokenid = l.strip().split(' ', 1)
            except:
                print(l)
                import pdb;pdb.set_trace()
            tokenid_dict[uttid] = tokenid

    with open(args.dst, "w", encoding="utf-8") as f:
        for u in utts:
            if u not in tokenid_dict.keys():
                import pdb; pdb.set_trace()
            toks_lid = add_lid(tokenid_dict[u])
            f.write(u+" "+toks_lid+"\n")