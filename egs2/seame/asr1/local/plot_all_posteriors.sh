#!/bin/bash
# mkdir -p posteriors/asr/

# mkdir -p posteriors/st/

post=$1
dir=$2

for i in $(seq 0 100)
do
   python local/plot_posteriors.py \
   --bi ${dir}/posteriors/bi${i}.np \
   --en ${dir}/posteriors/en${i}.np \
   --zh ${dir}/posteriors/zh${i}.np \
   --dst_bi ${dir}/posteriors/${i}bi.$post --dst_en ${dir}/posteriors/${i}en.$post --dst_zh ${dir}/posteriors/${i}zh.$post \
   --tok data/mseame_tedlium_aishell_lid_token_list/bpe_unigram8336/tokens.txt
done