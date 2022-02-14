import argparse
import codecs
import json
import logging
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    aid_tokens = ['US', 'CHN', 'PT', 'ES', 'JPN', 'IND', 'KR', 'UK', 'RU', 'CA']

    total = 0
    correct = 0
    correctbyaid = {x:0 for x in aid_tokens}
    totalbyaid = {x:0 for x in aid_tokens}
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            toks = l.strip().split()
            hyp = toks[0][1:-1]
            ref = toks[-1].split('_')[0][1:]
            if hyp == ref:
                correct += 1
                correctbyaid[ref] += 1
            total += 1
            totalbyaid[ref] += 1

    print("Total acc", correct/total)
    for aid in aid_tokens:
        print(aid, correctbyaid[aid] / totalbyaid[aid])