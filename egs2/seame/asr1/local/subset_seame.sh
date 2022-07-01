# python local/add_lid_seame.py --src dump/raw/train_sp/text

# mkdir -p dump/raw/train_seame_mono_sp/
# cp dump/raw/train_sp/feats_type dump/raw/train_seame_mono_sp/

python local/subset_seame_mono.py --src dump/raw/train_sp/ --dst dump/raw/train_seame_mono_sp/