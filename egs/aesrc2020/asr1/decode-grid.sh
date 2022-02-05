#!/bin/bash

first=2
step=1
end=12
train_conf=$1

let i=first
while ((i<=end)); do
    echo "decoding:" $i
    conf="conf/decode_grid/decode${i}.yaml"
    ./run.sh --stage 5 --ngpu 4 --train_config ${train_conf} --dumpdir /scratch/aesrcdump/ --decode_config ${conf}
    i=$((i+step))
done

#let i=first
#while ((i<=end)); do
#    echo "decoding:" $i
#    conf="conf/tuning/md/decode-de/decodedenoctc-${i}.yaml"
#    ./train_model.sh --tgt_lang fr --ngpu 4 --dumpdir /scratch/mustc-fr_fr --train_config conf/tuning/md/mdtrain-sa2g9-lg9.yaml --decode_config ${conf} --stage 6 --nj 12 
#
#    i=$((i+step))
#done
