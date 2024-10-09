#!/bin/bash

exp="sample_validation/rs"

# Comparison target
declare -a scms=(
    # Markovian
    "chain_lin_3"
    "chain_nlin_3"
    "chain_lin_4"
    "chain_lin_5"
    "collider_lin"
    "fork_lin"
    "fork_nlin"
    "largebd_nlin"
    "simpson_nlin"
    "simpson_symprod"
    "triangle_lin"
    "triangle_nlin"
    # Recursive
    "back_door"
    "front_door"
    "m"
    "napkin"
    # Canonical
    "fairness"
    "fairness_xw"
    "fairness_xy"
    "fairness_yw"
)

# Comparison Range
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
bash script/counterfactual/utils/sample/train_rs.sh\
    -n "$exp"\
    -s "$s"\
    -j "$j"\
    -sv 1024 -bv 1\
    -sd $sd\
    $@
done
done
done