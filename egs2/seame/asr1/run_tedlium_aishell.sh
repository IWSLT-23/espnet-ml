#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
# set -o pipefail

stage=1
stop_stage=13

train_set="train_tedlium_aishell"
valid_set="dev_tedlium_aishell"
test_sets="devman devsge"

asr_config=conf/tuning/train_asr_conformer.yaml
lm_config=conf/tuning/train_lm_transformer.yaml
inference_config=conf/decode_asr.yaml


## Data prep
# ./utils/combine_data.sh dump/raw/train_tedlium_aishell_sp ../../tedlium2/asr1/dump/raw/train_sp/ ../../aishell/asr1/dump/raw/train_sp/
# ./utils/combine_data.sh dump/raw/dev_tedlium_aishell ../../tedlium2/asr1/dump/raw/dev/ ../../aishell/asr1/dump/raw/dev/
## need to add spaces
# python local/tokenize_aishell.py --src dump/raw/train_tedlium_aishell_sp/text --dst dump/raw/train_tedlium_aishell_sp/text.sp
# mv dump/raw/train_tedlium_aishell_sp/text.sp dump/raw/train_tedlium_aishell_sp/text
# python local/tokenize_aishell.py --src dump/raw/dev_tedlium_aishell/text --dst dump/raw/dev_tedlium_aishell/text.sp
# mv dump/raw/dev_tedlium_aishell/text.sp dump/raw/dev_tedlium_aishell/text
# python local/combine_dicts.py --dst data/tedlium_aishell_tokens.txt --en_src ../../tedlium2/asr1/data/en_token_list/bpe_unigram500/ --man_src ../../aishell/asr1/data/zh_token_list/char/
source data/tedlium_aishell_tokens.txt
en_nbpe=1000
nbpe=$((en_nbpe + man_nbpe))

lang="tedlium_aishell"

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
    --asr_stats_dir "asr_stats_nbpe${nbpe}_t+a" \
    "$@"


    # --lm_train_text "data/${train_set}/text" \
    # --bpe_train_text "data/${train_set}/text.eng.bpe" \