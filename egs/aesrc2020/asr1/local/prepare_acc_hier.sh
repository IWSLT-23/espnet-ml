#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

dumpdir=$1     # data transformed into kaldi format
nbpe=$2
inter1=$3
inter2=$4

# separate accents in inter layers 
#python local/separate_accents.py --input ${dumpdir}/train_sp/deltafalse/data_unigram${inter1}.json
#python local/separate_accents.py --input ${dumpdir}/dev/deltafalse/data_unigram${inter1}.json
#python local/separate_accents.py --input ${dumpdir}/train_sp/deltafalse/data_unigram${inter2}.json
#python local/separate_accents.py --input ${dumpdir}/dev/deltafalse/data_unigram${inter2}.json

# add aid tags
#python local/add_aid_tag.py --input ${dumpdir}/train_sp/deltafalse/data_unigram${inter1}.json.acc
#python local/add_aid_tag.py --input ${dumpdir}/dev/deltafalse/data_unigram${inter1}.json.acc
#python local/add_aid_tag.py --input ${dumpdir}/train_sp/deltafalse/data_unigram${inter2}.json.acc
#python local/add_aid_tag.py --input ${dumpdir}/dev/deltafalse/data_unigram${inter2}.json.acc
#python local/add_aid_tag.py --input ${dumpdir}/train_sp/deltafalse/data_unigram${nbpe}.json
#python local/add_aid_tag.py --input ${dumpdir}/dev/deltafalse/data_unigram${nbpe}.json

# # combine into one
tag=${inter1}-${inter2}
#python local/make_hier_json.py --input ${dumpdir}/train_sp/deltafalse/data_unigram${nbpe}.json.aid \
#--inter1 ${dumpdir}/train_sp/deltafalse/data_unigram${inter1}.json.acc.aid \
#--inter2 ${dumpdir}/train_sp/deltafalse/data_unigram${inter2}.json.acc.aid --tag ${tag}
#mv ${dumpdir}/train_sp/deltafalse/data_unigram${nbpe}.json.aid.hier${inter1}-${inter2} \
#    ${dumpdir}/train_sp/deltafalse/data_unigram${nbpe}.json.aid.acc.hier${inter1}-${inter2}
#python local/make_hier_json.py --input ${dumpdir}/dev/deltafalse/data_unigram${nbpe}.json.aid \
#--inter1 ${dumpdir}/dev/deltafalse/data_unigram${inter1}.json.acc.aid \
#--inter2 ${dumpdir}/dev/deltafalse/data_unigram${inter2}.json.acc.aid --tag ${tag}
#mv ${dumpdir}/dev/deltafalse/data_unigram${nbpe}.json.aid.hier${inter1}-${inter2} \
#    ${dumpdir}/dev/deltafalse/data_unigram${nbpe}.json.aid.acc.hier${inter1}-${inter2}

python local/add_aid_tag.py --input ${dumpdir}/train_sp/deltafalse/data_unigram${inter1}.json
python local/add_aid_tag.py --input ${dumpdir}/dev/deltafalse/data_unigram${inter1}.json
python local/make_hier_json.py --input ${dumpdir}/train_sp/deltafalse/data_unigram${nbpe}.json.aid \
--inter1 ${dumpdir}/train_sp/deltafalse/data_unigram${inter1}.json.aid \
--tag ${tag}
python local/make_hier_json.py --input ${dumpdir}/dev/deltafalse/data_unigram${nbpe}.json.aid \
--inter1 ${dumpdir}/dev/deltafalse/data_unigram${inter1}.json.aid \
--tag ${tag}
echo ".aid.acc.hier${tag}"
exit 0;
