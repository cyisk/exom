#!/bin/bash

exp="density_estimation/rs"

# Iteration Range
declare -a scms=(
    "simpson_nlin"
    "largebd_nlin"
    "triangle_nlin"
)
declare -a js=(
    "5"
    "3"
    "1"
)

# Seeds
declare -a seeds=(
    0
    7
    42
    3407
    65535
)

for s in "${scms[@]}"; do \
for j in "${js[@]}"; do \
for sd in "${seeds[@]}"; do \
# context: e+t w_e w_t
# reduce: attn
# transforms: 5
# hiddens: 64x2
bash script/counterfactual/utils/density/train_rs.sh\
    -n "$exp"\
    -s "$s"\
    -j $j\
    -sv 1024 -bv 1 -ev 1000\
    -sd $sd\
    $@
done
done
done