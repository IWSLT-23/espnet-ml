import argparse
import sys
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rcParams

# rcParams.update({'figure.autolayout': True})

from espnet2.utils.types import str2triple_str

from  matplotlib.colors import LinearSegmentedColormap


def get_parser():
    parser = argparse.ArgumentParser(
        description="Calculate the diagonality of the self-attention weights."
    )
    parser.add_argument(
        "--bi",
        type=str,
        help="path to results",
    )
    parser.add_argument(
        "--en",
        type=str,
        help="path to results",
    )
    parser.add_argument(
        "--zh",
        type=str,
        help="path to results"
    )
    parser.add_argument(
        "--dst_bi",
        type=str,
        help="path to results"
    )
    parser.add_argument(
        "--dst_en",
        type=str,
        help="path to results"
    )
    parser.add_argument(
        "--dst_zh",
        type=str,
        help="path to results"
    )
    parser.add_argument(
        "--tok",
        type=str,
        help="path to results"
    )
    return parser

def get_b_nb(x):
    b = x[:,:,0][0]
    nb = x[:,:,1:].max(axis=-1)[0]
    nb_tok = x[:,:,1:].argmax(axis=-1)[0] + 1
    return b, nb, nb_tok

def plot(posts, name, tok_dict, dst):
    b, nb, nb_tok = get_b_nb(posts)
    label = [tok_dict[i].strip() for i in nb_tok]
    for i in range(len(nb)):
        # if nb[i] <= b[i]:
        if nb[i] <= 0.2:
            label[i]=""
    
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.rcParams["figure.figsize"] = (18.0,3.0)
    x = [i+1 for i in range(posts.shape[1])]

    if name == "bi":
        c = 'purple'
    elif name == "zh":
        c = 'red'
    else:
        c = 'blue'
    plt.errorbar(x, nb, linestyle='-', color=c, label=name+" Non-Blank")
    plt.errorbar(x, b, linestyle=':', color='gray', label=name+" Blank")

    plt.legend(loc='lower center', ncol=2, fontsize=14)
    plt.title(name+' CTC Posteriors', fontsize=19)
    plt.xlabel('Frames', fontsize=19)
    plt.ylabel("CTC Posterior", fontsize=19)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    prev = ""
    for i in range(len(nb)):
        if label[i] != prev:
            plt.annotate(label[i], # this is the text
                        (x[i],nb[i]), # these are the coordinates to position the label
            )
            prev = label[i]

    plt.show()
    plt.savefig(dst)

    plt.clf() 

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    bi = np.load(args.bi, allow_pickle=True)
    en = np.load(args.en, allow_pickle=True)
    zh = np.load(args.zh, allow_pickle=True)

    tok_dict = {i:tok for i,tok in enumerate(open(args.tok).readlines())}
    tok_dict[len(tok_dict.keys()) - 1] = "<msk>"

    plot(bi, "bi", tok_dict, args.dst_bi)
    plot(en, "en", tok_dict, args.dst_en)
    plot(zh, "zh", tok_dict, args.dst_zh)