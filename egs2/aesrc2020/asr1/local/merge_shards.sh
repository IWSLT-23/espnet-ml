lab_dir=$1
nshard=$2

echo -n "">$lab_dir/all.km
for split in $(seq 0 63)
do
    for rank in $(seq 0 $((10 - 1))); do
      cat $lab_dir/${i}_${rank}_${nshard}.km
    done > $lab_dir/${split}.km
    cat $lab_dir/${split}.km >> $lab_dir/all.km
done

for x in $(seq 0 $((500 - 1))); do
    echo "$x 1"
done >> $lab_dir/dict.km.txt
