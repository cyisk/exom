#!/bin/bash

exp="sample_validation/nis"

# Comparison target
declare -a scms=(
    # Markovian
    #"chain_lin_3"
    #"chain_nlin_3"
    #"chain_lin_4"
    #"chain_lin_5"
    #"collider_lin"
    #"fork_lin"
    #"fork_nlin"
    #"largebd_nlin"
    "simpson_nlin"
    #"simpson_symprod"
    #"triangle_lin"
    #"triangle_nlin"
    # Recursive
    #"back_door"
    #"front_door"
    #"m"
    "napkin"
    # Canonical
    #"fairness"
    "fairness_xw"
    #"fairness_xy"
    #"fairness_yw"
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
if [[ "$s" == "fairness_xw" ]] && [[ "$j" == 5 ]]; then st=500; else st=1000; fi
# context: e+t w_e w_t
# reduce: attn
# transforms: 5
# hiddens: 64x2
bash script/counterfactual/utils/sample/train_nis.sh\
    -n "$exp"\
    -s "$s"\
    -j "$j"\
    -c "e+t w_e w_t"\
    -k "mb1 mb1 em"\
    -r attn -t 5 -d 64x2\
    -sv 1024 -bv 32 -st $st\
    -e 50 -sd $sd\
    $@
done
done
done