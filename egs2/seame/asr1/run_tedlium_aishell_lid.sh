#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
# set -o pipefail

stage=1
stop_stage=13

train_set="train_tedlium_aishell_lid"
valid_set="dev_tedlium_aishell_lid"
test_sets="devman devsge"

asr_config=conf/tuning/train_asr_conformer.yaml
lm_config=conf/tuning/train_lm_transformer.yaml
inference_config=conf/decode_asr.yaml


## Starts from tedlium_aishell prep and adds lid
# python local/add_lid.py --src dump/raw/train_tedlium_aishell_lid_sp/text
# python local/add_lid.py --src dump/raw/dev_tedlium_aishell_lid/text

source data/tedlium_aishell_tokens.txt
bpe_nlsyms="▁<en>,▁<zh>,"${bpe_nlsyms}
en_nbpe=1000
nbpe=$((en_nbpe + man_nbpe + 2))

lang="tedlium_aishell_lid"

./asr.sh \
    --lang ${lang} \
    --bpe_train_text dump/raw/train_tedlium_aishell_sp/text \
    --ngpu 1 \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --nbpe ${nbpe} \
    --bpe_nlsyms "${bpe_nlsyms}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --score_opts "-e utf-8 -c NOASCII" \
    --asr_stats_dir "asr_stats_nbpe${nbpe}_t+a_lid" \
    "$@"


    # --lm_train_text "data/${train_set}/text" \
    # --bpe_train_text "data/${train_set}/text.eng.bpe" \