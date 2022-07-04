hyp=$1
ref=$2

# if no lid
python local/add_lid_seame.py --src $hyp
python local/add_lid_seame.py --src $ref
hyp="${hyp}_lid"
ref="${ref}_lid"

# if the utts have leading ids
cut -d' ' -f2- $hyp > "${hyp}_noid"
cut -d' ' -f2- $ref > "${ref}_noid"
hyp="${hyp}_noid"
ref="${ref}_noid"

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