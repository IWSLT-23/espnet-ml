import sys
import argparse
import sentencepiece as spm

args = None
char = open('data/lang/10k_units.txt','r')
count = 5001
Man_dict = {}
for line in char:
    line = line.strip()
    c, idx = line.split()
    if int(idx) >= count:
        Man_dict[c] = int(idx) 


Alph = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
unknown = 5000

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True 
    else:
        return False

def main():
    sp = spm.SentencePieceProcessor()
    if not sp.Load('data/lang/gigaspeech.bpe.model'):
        print('loading bpe model error')
        sys.exit() 
    with open(args.input, 'r', encoding='utf-8') as f, open(args.input+'.int', 'w', encoding='utf-8') as f_out:
        lines = f.readlines()
        for line in lines:
            int_lst = []
            try:
                uttid, txt = line.strip().split(' ', 1)
            except:
                import pdb;pdb.set_trace()
            toks = txt.split()
            for ele in toks:
                ele_lst = list(ele)
                if(common_member(ele_lst, Alph)):
                    Eng_list = sp.EncodeAsIds(ele)
                    int_lst.extend(list(Eng_list))
                else:
                    for Man_e in ele_lst:
                       if Man_e in Man_dict.keys():
                           #print(Man_dict)
                           int_lst.append(Man_dict[Man_e])
                       else:
                           int_lst.append(unknown)    
            f_out.write(uttid + ' ' + ' '.join(str(i) for i in int_lst) + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'symbols to integer accoding to trained bpe model')
    parser.add_argument('--input', type=str, help='input transcription file')
    args, unk = parser.parse_known_args()
    main()

