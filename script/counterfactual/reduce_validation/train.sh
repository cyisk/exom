#!/bin/bash

exp="reduce_validation"

# Comparison target
declare -a reduces=(
    "concat"
    "sum"
    "wsum"
    "attn"
)

# Comparison Range
declare -a nets=(
    "gmm"
    "maf"
    "nice"
    "sospf"
)
declare -a scms=(
    "simpson_nlin"
    "triangle_nlin"
    "largebd_nlin"
    "m"
    "napkin"
)
declare -a js=(
    "5"
    "3"
)

# Seeds
declare -a seeds=(
    0
    7
    42
    3407
    65535
)

for n in "${nets[@]}"; do \
for s in "${scms[@]}"; do \
for j in "${js[@]}"; do \
for r in "${reduces[@]}"; do \
for sd in "${seeds[@]}"; do \
if [[ "$n" == "gmm" ]]; then t=10; else t=5; fi
# context: e+t w_e w_t
# reduce: attn
# transforms: 5(gmm:10)
# hiddens: 64x2
bash script/counterfactual/utils/sample/train_exom.sh\
    -n "$exp"\
    -s "$s"\
    -j "$j"\
    -m "$n"\
    -c "e+t w_e w_t"\
    -k "mb1 mb1 em"\
    -r "$r" -t $t -d 64x2\
    -sv 1024 -bv 32\
    -e 200 -sd $sd\
    $@
done
done
done
done
done