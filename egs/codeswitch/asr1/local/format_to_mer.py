import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="modify hyp and ref to be word based for en")
    parser.add_argument("--hyp", required=True, type=str)
    parser.add_argument("--ref", required=True, type=str)
    return parser

args = get_parser().parse_args()

def reformat(filename):
    alph = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    with open(filename, "r", encoding="utf-8") as f1, open(filename+".wrd", "w", encoding="utf-8") as f2:
        lines = f1.readlines()
        for li in lines:
            try:
                txt, id = li.strip().rsplit(' ', 1)
            except:
                f2.write(li)
                continue
            new_txt = ''
            toks = txt.split()
            for t in toks:
                if t[0] in alph:
                    new_txt += t
                else:
                    new_txt += ' ' + t
            new_txt = new_txt.strip()
            f2.write(new_txt + ' '+ id + '\n')

def main(args):
    reformat(args.ref)
    reformat(args.hyp)


if __name__ == "__main__":
    main(args)