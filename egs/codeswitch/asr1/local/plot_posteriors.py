import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--units", required=True, type=str)
    return parser

args = get_parser().parse_args()

dictionary = {}
with open(args.units, encoding="utf-8") as f:
    for l in f:
        try:
            symbol, val = l.strip().split()
        except:
            symbol = ""
            val = l.strip()
        if int(val) < 5000:
            dictionary[int(val)] = symbol
        else:
            dictionary[int(val)] = symbol

posts = np.load(args.input)

zh_post = posts[0]

en_post = posts[1]

x = [i for i in range(zh_post.shape[0])]

zh_blank_y = zh_post[:,0]
zh_nonb_y = zh_post[:,1:].max(axis=1)
zh_labels = zh_post.argmax(axis=1)

ChineseFont2 = FontProperties('SimHei')

fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 4), dpi=80)
# fig = plt.figure(figsize=(20, 2), dpi=80)
# plt.xlabel('Frame #')
# plt.ylabel('Posterior Value')
# plt.title('Mandarin CTC Posteriors')
ax1.plot(x, zh_blank_y, color='red', linewidth=1.0, linestyle='--')
ax1.plot(x, zh_nonb_y, color='red', linewidth=1.0)
prev = 0
for i, l in enumerate(zh_labels):
    if l != 0 and l != prev:
       ax1.annotate(dictionary[l], xy=(x[i],zh_nonb_y[i]-0.1), fontproperties = ChineseFont2, fontsize=20)
    prev = l


# fig.savefig('posterior_line_chart/zh.png', dpi=fig.dpi)

en_blank_y = en_post[:,0]
en_nonb_y = en_post[:,1:].max(axis=1)
en_labels = en_post.argmax(axis=1)

# fig = plt.figure(figsize=(20, 2), dpi=80)
# plt.xlabel('Frame #')
# plt.ylabel('Posterior Value')
# plt.title('English CTC Posteriors')
ax2.plot(x, en_blank_y, color='blue', linewidth=1.0, linestyle='--')
ax2.plot(x, en_nonb_y, color='blue', linewidth=1.0)
prev = 0
for i, l in enumerate(en_labels):
    if l != 0 and l != prev:
        ax2.annotate(dictionary[l], xy=(x[i],en_nonb_y[i]-0.1), fontsize=18)
    prev = l

fig.savefig(args.input+'.png', dpi=fig.dpi)


# # Plot on single chart and remove blanks
# fig = plt.figure(figsize=(20, 2), dpi=80)
# plt.plot(x, zh_nonb_y, color='red', linewidth=1.0)

# prev=0
# for i, l in enumerate(zh_labels):
#     if l != 0  and l != prev:
#         plt.annotate(dictionary[l], xy=(x[i],zh_nonb_y[i]), fontproperties = ChineseFont2)
#     prev = l

# plt.plot(x, en_nonb_y, color='blue', linewidth=1.0)
# prev=0
# for i, l in enumerate(en_labels):
#     if l != 0 and l != prev:
#         plt.annotate(dictionary[l], xy=(x[i],en_nonb_y[i]))
#     prev = l

# fig.savefig('posterior_line_chart/tmp2.png', dpi=fig.dpi)


# # Plot on single chart and remove blanks and LIDs
# zh_nonb_y = zh_post[:,1:-3].max(axis=1)
# en_nonb_y = en_post[:,1:-3].max(axis=1)

# fig = plt.figure(figsize=(20, 2), dpi=80)
# plt.plot(x, zh_nonb_y, color='red', linewidth=1.0)

# prev=0
# for i, l in enumerate(zh_labels):
#     if l != 0  and l != prev and l < 11237:
#         plt.annotate(dictionary[l], xy=(x[i],zh_nonb_y[i]-0.1), fontproperties = ChineseFont2, fontsize=18)
#     prev = l

# plt.plot(x, en_nonb_y, color='blue', linewidth=1.0)
# prev=0
# for i, l in enumerate(en_labels):
#     if l != 0 and l != prev and l < 11237:
#         plt.annotate(dictionary[l], xy=(x[i],en_nonb_y[i]-0.1), fontsize=18)
#     prev = l

# fig.savefig('posterior_line_chart/tmp3.png', dpi=fig.dpi)