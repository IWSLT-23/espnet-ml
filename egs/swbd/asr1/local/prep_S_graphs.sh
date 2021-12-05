#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

langdir="data/lang_char/"
traintxt="${langdir}/input.txt"
units="${langdir}/train_nodup_bpe2000_units.txt"
tokens="${langdir}/tokens_bpe2000.txt"
outdir="${langdir}/S_graphs_bpe2000"

. utils/parse_options.sh || exit 1;

# 0. make tokens
#cut -f 1 -d ' ' $units > $tokens

# 1. build raw graphs in gtn
mkdir -p $outdir
rawdir="${outdir}/gtn_raw/"
mkdir -p $rawdir
#python local/build_S_graphs_gtn_raw.py --text ${traintxt} --tokens ${tokens} --outdir ${rawdir}

# 2. convert raw graphs to ofst format
orawdir="${outdir}/ofst_raw/"
mkdir -p $orawdir
#python local/convert_S_graphs_gtn_to_ofst.py --indir ${rawdir} --outdir ${orawdir}

# 3. determinize and minimize
odetmindir="${outdir}/ofst_det_min/"
mkdir -p $odetmindir
#./local/det_min_S_graphs.sh --indir ${orawdir} --outdir ${odetmindir}

# 4. convert det_min graphs back to gtn format
gdetmindir="${outdir}/gtn_det_min/"
mkdir -p $gdetmindir
python local/convert_S_graphs_ofst_to_gtn.py --indir ${odetmindir} --outdir ${gdetmindir}
