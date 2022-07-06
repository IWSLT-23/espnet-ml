dir=$1
hyp=$1/hyp.trn
ref=$1/ref.trn

# sed -i 's,▁, ,g' ${hyp}
cp ${hyp} "${hyp}.nolid"
sed -i 's,<en> ,,g' "${hyp}.nolid"
sed -i 's,<zh> ,,g' "${hyp}.nolid"

sclite \
    -r "${ref}" trn \
    -h "${hyp}.nolid" trn \
    -i rm -o dtl stdout > ${dir}/result.nolid.txt
grep "Percent Total Error" ${dir}/result.nolid.txt