 #!/usr/bin/env bash
st=$1
ed=$2

for j in $(seq $st $ed)
do
    for i in $(seq 0 9)
    do
        python hubert/simple_kmeans/dump_km_label.py dump/hub_feats_10shard_9rank/ $j dump/kmeans 10 $i dump/lab_dir
    done
done
