#!/bin/bash

exp="density_estimation/rs_causal_nf_truth"

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
)
declare -a seeds2=(
    0
    7
    42
    3407
    65535
)

for s in "${scms[@]}"; do \
for j in "${js[@]}"; do \
for sd in "${seeds[@]}"; do \
for sd2 in "${seeds[@]}"; do \
# context: e+t w_e w_t
# reduce: attn
# transforms: 5
# hiddens: 64x2
bash script/counterfactual/utils/density/train_rs_from_causal_nf.sh\
    -n "$exp"\
    -pm nsf -pd 32x3 -ps $sd2\
    -s "$s"\
    -j $j\
    -sv 1024 -bv 1 -ev 10000\
    -sd $sd\
    $@
done
done
done
done