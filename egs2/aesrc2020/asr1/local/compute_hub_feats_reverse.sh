 #!/usr/bin/env bash
shards=$1
rank=$2
dump=$3

for i in $(seq 64 -1 30)
do
    python hubert/simple_kmeans/dump_hubert_feature.py dump/tsv_dir/ $i hubert/hubert_base_ls960.pt 12 $shards $rank $dump
done
