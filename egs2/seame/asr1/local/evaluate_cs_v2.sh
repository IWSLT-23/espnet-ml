#espnet version

dir=$1
hyp=$1/hyp.trn.nolid
ref=$1/ref.trn

# remove trailing tab separated id
cut -d$'\t' -f1 ${hyp} > "${hyp}.noid"
cut -d$'\t' -f1 ${ref} > "${ref}.noid"
hyp="${hyp}.noid"
ref="${ref}.noid"

# if no lid
python local/add_lid_seame_v2.py --src $hyp
python local/add_lid_seame_v2.py --src $ref
hyp="${hyp}_lid"
ref="${ref}_lid"

# masked transcripts: mono en, mono zh, lid only
# also adds an id back to end w/ tab separation, for compatibility w sclite cmd
python local/create_trn_format.py --hyp $hyp --ref $ref

# sclite
echo "all (no lid)"
sclite \
    -r "${hyp}.all" trn \
    -h "${ref}.all" trn \
    -i rm -o dtl stdout > "${hyp}.all.result"
grep "Percent Total Error" "${hyp}.all.result"

echo "en only"
sclite \
    -r "${hyp}.en" trn \
    -h "${ref}.en" trn \
    -i rm -o dtl stdout > "${hyp}.en.result"
grep "Percent Total Error" "${hyp}.en.result"

echo "zh only"
sclite \
    -r "${hyp}.zh" trn \
    -h "${ref}.zh" trn \
    -i rm -o dtl stdout > "${hyp}.zh.result"
grep "Percent Total Error" "${hyp}.zh.result"

echo "lid only"
sclite \
    -r "${hyp}.lid" trn \
    -h "${ref}.lid" trn \
    -i rm -o dtl stdout > "${hyp}.lid.result"
grep "Percent Total Error" "${hyp}.lid.result"