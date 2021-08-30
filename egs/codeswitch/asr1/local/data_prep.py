import argparse
import codecs
import json
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # assumes data dir will have text and wav.scp
    parser.add_argument("--data_dir")
    args = parser.parse_args()

    txt_dict = {}
    with open(args.data_dir+"/text", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            uttid, txt = l.split(' ', 1)
            # uttid, txt = l.split('\t', 1)
            spkid = uttid[:5]   #king asr has first 5 digits as spk
            uttid = 'K190-'+spkid+'_'+uttid[5:] #fix king asr uttid to have dash
            txt_dict[uttid] = txt

    wav_dict = {}
    with open(args.data_dir+"/wav.scp", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            # uttid, wav = l.split(' ', 1)
            uttid, wav = l.split(' ', 1)
            spkid = uttid[:5]   #king asr has first 5 digits as spk
            uttid = 'K190-'+spkid+'_'+uttid[5:] #fix king asr uttid to have dash
            wav_dict[uttid] = wav

    # txt_utts_sorted = sorted(txt_dict.keys())
    # wav_utts_sorted = sorted(wav_dict.keys())
    txt_utts = set(txt_dict.keys())
    wav_utts = set(wav_dict.keys())
    common_utts = txt_utts.intersection(wav_utts)
    print(len(common_utts))
    common_utts_sorted = sorted(common_utts)

    # if len(txt_utts_sorted) != len(wav_utts_sorted):
    #     import pdb; pdb.set_trace()
    # else:
    #     for i in range(len(txt_utts_sorted)):
    #         if txt_utts_sorted[i] != wav_utts_sorted[i]:
    #             import pdb; pdb.set_trace()
    
    #update pattern based on the dataset
    #spkid needs to be a prefix of uttid
    with open(args.data_dir+"/spk2utt", "w", encoding="utf-8") as f1, \
        open(args.data_dir+"/utt2spk", "w", encoding="utf-8") as f2:
        for uttid in common_utts_sorted:
            # toks = uttid.split("_")
            # spkid = "_".join(toks[:5])
            # spkid,_ = uttid.split(".", 1)

            # spkid = uttid[:5]   #king asr has first 5 digits as spk
            spkid,_ = uttid.split("_", 1)
            f1.write(spkid+" "+uttid+"\n")
            f2.write(uttid+" "+spkid+"\n")

    with open(args.data_dir+"/wav.scp", "w", encoding="utf-8") as f1, \
        open(args.data_dir+"/text", "w", encoding="utf-8") as f2:
        for uttid in common_utts_sorted:
            f1.write(uttid+" "+wav_dict[uttid])
            f2.write(uttid+" "+txt_dict[uttid])