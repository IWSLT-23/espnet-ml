#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=4        # start from 0 if you need to start from data preparation
stop_stage=8
ngpu=8         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml 
train_config=conf/train.yaml

lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10
lang_model=rnnlm.model.best # set a language model to be used for decoding

use_snapshot_range=false
snapshot_lower=
snapshot_upper=

# bpemode (unigram or bpe)
nbpe=10k
bpemode=bpe

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_cs_zh_en_even
train_dev=dev_trim
recog_set=test

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # subset 190k; cs is ~190k
    utils/subset_data_dir.sh --first data/train_zh 190000 data/train_zh_sub
    utils/subset_data_dir.sh --first data/train_en 190000 data/train_en_sub
    head -n 190000 data/train_zh/textid > data/train_zh_sub/textid
    head -n 190000 data/train_en/textid > data/train_en_sub/textid

    utils/combine_data.sh --extra_files textid data/$train_set data/train_200h_trim data/train_zh_sub data/train_en_sub

    #compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/$train_set/feats.scp data/$train_set/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
fi

dict=data/lang/10k_units.txt
bpemodel=data/lang_char/train_nodup_${bpemode}${nbpe}

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    nj=32
    echo "make json files"
    local/data2json.sh --nj ${nj} --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json

    # add tokens
    python local/add_token.py --input ${feat_tr_dir}/data_${bpemode}${nbpe}.json --dict $dict
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
	expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
	expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Averaging"
    nj=32
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
           recog_model=model.last${n_average}.avg.best${snapshot_upper}
        if ${use_snapshot_range}; then
        average_checkpoints.py \
            --backend ${backend} \
            --snapshots $(seq -f "${expdir}/results/snapshot.ep.%02g" ${snapshot_lower} ${snapshot_upper}) \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}
        else
            average_checkpoints.py \
                --backend ${backend} \
                --snapshots ${expdir}/results/snapshot.ep.* \
                --out ${expdir}/results/${recog_model} \
                --num ${n_average}
        fi
    fi
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Decoding"
    recog_model=model.last${n_average}.avg.best${snapshot_upper}
    nj=64
    pids=() # initialize pids
    for rtask in ${train_dev}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            #--rnnlm ${lmexpdir}/${lang_model}

        local/score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
        sclite -r ${expdir}/${decode_dir}/ref.trn trn -h ${expdir}/${decode_dir}/hyp.trn trn -i rm -o dtl stdout > ${expdir}/${decode_dir}/dtl.txt

        local/score_sclite_onelang.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --lang "en" ${expdir}/${decode_dir} ${dict}
        sclite -r ${expdir}/${decode_dir}/ref.en.trn trn -h ${expdir}/${decode_dir}/hyp.en.trn trn -i rm -o dtl stdout > ${expdir}/${decode_dir}/dtl.en.txt

        local/score_sclite_onelang.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --lang "zh" ${expdir}/${decode_dir} ${dict}
        sclite -r ${expdir}/${decode_dir}/ref.zh.trn trn -h ${expdir}/${decode_dir}/hyp.zh.trn trn -i rm -o dtl stdout > ${expdir}/${decode_dir}/dtl.zh.txt
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding ZH"
    recog_model=model.last${n_average}.avg.best${snapshot_upper}
    nj=64
    pids=() # initialize pids
    train_dev=dev_zh
    for rtask in ${train_dev}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            #--rnnlm ${lmexpdir}/${lang_model}

        local/score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
        sclite -r ${expdir}/${decode_dir}/ref.trn trn -h ${expdir}/${decode_dir}/hyp.trn trn -i rm -o dtl stdout > ${expdir}/${decode_dir}/dtl.txt
        # local/score_sclite_onelang.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --lang "en" ${expdir}/${decode_dir} ${dict}
        # local/score_sclite_onelang.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --lang "zh" ${expdir}/${decode_dir} ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Decoding EN"
    recog_model=model.last${n_average}.avg.best${snapshot_upper}
    nj=64
    pids=() # initialize pids
    train_dev=dev_en
    for rtask in ${train_dev}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            #--rnnlm ${lmexpdir}/${lang_model}

        local/score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
        sclite -r ${expdir}/${decode_dir}/ref.trn trn -h ${expdir}/${decode_dir}/hyp.trn trn -i rm -o dtl stdout > ${expdir}/${decode_dir}/dtl.txt
        # local/score_sclite_onelang.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --lang "en" ${expdir}/${decode_dir} ${dict}
        # local/score_sclite_onelang.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --lang "zh" ${expdir}/${decode_dir} ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Decoding EN test"
    recog_model=model.last${n_average}.avg.best${snapshot_upper}
    nj=64
    pids=() # initialize pids
    train_dev=test_en
    for rtask in ${train_dev}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            #--rnnlm ${lmexpdir}/${lang_model}

        local/score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
        sclite -r ${expdir}/${decode_dir}/ref.trn trn -h ${expdir}/${decode_dir}/hyp.trn trn -i rm -o dtl stdout > ${expdir}/${decode_dir}/dtl.txt
        local/score_sclite_onelang.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --lang "en" ${expdir}/${decode_dir} ${dict}
        local/score_sclite_onelang.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --lang "zh" ${expdir}/${decode_dir} ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Decoding CS for posterior dump"
    recog_model=model.last${n_average}.avg.best${snapshot_upper}
    nj=64
    pids=() # initialize pids
    train_dev=dev_trim
    for rtask in ${train_dev}; do
    # (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}_post_dump
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}_perfect.json

        #### use CPU for decoding
        ngpu=0

        # ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}_perfect.1.json \
        --result-label ${expdir}/${decode_dir}/data.1.json \
        --model ${expdir}/results/${recog_model} \
        #--rnnlm ${lmexpdir}/${lang_model}

    #     local/score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    #     sclite -r ${expdir}/${decode_dir}/ref.trn trn -h ${expdir}/${decode_dir}/hyp.trn trn -i rm -o dtl stdout > ${expdir}/${decode_dir}/dtl.txt
    #     local/score_sclite_onelang.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --lang "en" ${expdir}/${decode_dir} ${dict}
    #     local/score_sclite_onelang.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --lang "zh" ${expdir}/${decode_dir} ${dict}
    # ) &
    # pids+=($!) # store background pids
    done
    # i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    # [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    # echo "Finished"
fi