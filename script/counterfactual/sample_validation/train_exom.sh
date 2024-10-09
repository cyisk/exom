#!/bin/bash

exp="sample_validation/exom"

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
declare -a nets=(
    "gmm"
    "maf"
    "nsf"
    "ncsf"
    "nice"
    "naf"
    "unaf"
    "sospf"
    "bpf"
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

for n in "${nets[@]}"; do \
for s in "${scms[@]}"; do \
for j in "${js[@]}"; do \
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
    -r attn -t $t -d 64x2\
    -sv 1024 -bv 32\
    -e 200 -sd $sd\
    $@
done
done
done
done