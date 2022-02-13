 #!/usr/bin/env bash
shards=$1
rank=$2
dump=$3
st=$4
ed=$5

for i in $(seq $st $ed)
do
    for s in $(seq 0 8)
    do
        python hubert/simple_kmeans/dump_hubert_feature.py dump/tsv_dir/ $i hubert/hubert_base_ls960.pt 12 $shards $s $dump
    done
done
