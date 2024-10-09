#!/bin/bash

exp="effect_estimation/rs"

# Iteration Range
declare -a scms=(
    "fairness"
    "fairness_xw"
    "fairness_xy"
    "fairness_yw"
)
declare -a qs=(
    "ate"
    "ett"
    "nde"
    "ctfde"
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
for q in "${qs[@]}"; do \
for sd in "${seeds[@]}"; do \
# context: e+t w_e w_t
# reduce: attn
# transforms: 5
# hiddens: 64x2
bash script/counterfactual/utils/effect/train_rs.sh\
    -n "$exp"\
    -s "$s"\
    -q "$q"\
    -sv 1024 -bv 32 -ev 1000\
    -sd $sd\
    $@
done
done
done