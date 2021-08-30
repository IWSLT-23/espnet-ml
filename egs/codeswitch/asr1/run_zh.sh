#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=4        # start from 0 if you need to start from data preparation
stop_stage=100
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

train_set=train_zh
train_dev=dev_zh
recog_set=test_zh

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    # for x in train_500h; do
    #     steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
    #         data/${x} exp/make_fbank/${x} ${fbankdir}
    #     utils/fix_data_dir.sh data/${x}
    # done

    # # utils/data/remove_dup_utts.sh 300 data/train_500h data/train_500h_nodup

    # # remove utt having > 2000 frames or < 10 frames or
    # # remove utt having > 400 characters or 0 characters
    # remove_longshortdata.sh --maxchars 400 data/train_500h_nodup data/${train_set}

    # utils/subset_data_dir.sh --first data/train_500h_trim 10000 data/dev_test_zh
    # n=$(($(wc -l < data/train_500h_trim/text) - 10000))
    # utils/subset_data_dir.sh --last data/train_500h_trim ${n} data/${train_set}

    # utils/subset_data_dir.sh --first data/dev_test_zh 5000 data/dev_zh
    # utils/subset_data_dir.sh --last data/dev_test_zh 5000 data/test_zh

    # # compute global CMVN
    # compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    # dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
    #     data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    # for rtask in ${recog_set}; do
    #     feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
    #     dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
    #         data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
    #         ${feat_recog_dir}
    # done
fi

dict=data/lang/10k_units.txt
bpemodel=data/lang_char/train_nodup_${bpemode}${nbpe}

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    nj=32
    echo "make json files"
    python local/sym2int.py --text data/$train_set/text --src data/train_500h_trim/text.int \
        --dst data/$train_set/textid --output_dim 11239 --nbpe 5000
    local/data2json.sh --nj ${nj} --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json

    python local/sym2int.py --text data/$train_dev/text --src data/train_500h_trim/text.int \
        --dst data/$train_dev/textid --output_dim 11239 --nbpe 5000
    local/data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        python local/sym2int.py --text data/$recog_set/text --src data/train_500h_trim/text.int \
            --dst data/$recog_set/textid --output_dim 11239 --nbpe 5000
        local/data2json.sh --feat ${feat_recog_dir}/feats.scp --allow-one-column true \
            --bpecode ${bpemodel}.model data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done

    # add tokens
    python local/add_token.py --input ${feat_tr_dir}/data_${bpemode}${nbpe}.json --dict $dict
    python local/add_token.py --input ${feat_dt_dir}/data_${bpemode}${nbpe}.json --dict $dict
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        python local/add_token.py --input ${feat_recog_dir}/data_${bpemode}${nbpe}.json --dict $dict
    done
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
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend ${backend} \
                    --snapshots ${expdir}/results/snapshot.ep.* \
                    --out ${expdir}/results/${recog_model} \
                    --num ${n_average}
    fi
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Decoding"
    recog_model=model.last${n_average}.avg.best
    nj=32
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
        local/score_sclite_onelang.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --lang "zh" ${expdir}/${decode_dir} ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding acc best"
    recog_model=model.acc.best
    nj=32
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
        local/score_sclite_onelang.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --lang "zh" ${expdir}/${decode_dir} ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi