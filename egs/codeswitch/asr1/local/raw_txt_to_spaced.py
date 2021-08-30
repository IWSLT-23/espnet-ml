import argparse
import codecs
import json
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add token gt using dict
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    alph = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

    with open(args.input, "r", encoding="utf-8") as f_in, open(args.input+'.spa', "w", encoding="utf-8") as f_out:
        lines = f_in.readlines()
        for li in lines:
            uttid, txt = li.strip().split(' ', 1)
            new_txt = ""
            for c in txt:
                if c in alph or c == ' ':
                    new_txt += c
                else:
                    new_txt += ' ' + c + ' '
            new_txt = ' '.join(new_txt.split())
            f_out.write(uttid + ' ' + new_txt + '\n')