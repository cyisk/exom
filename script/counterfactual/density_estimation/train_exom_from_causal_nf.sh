#!/bin/bash

exp="density_estimation/causal_nf"

# Iteration Range
declare -a nets=(
    "maf"
    "nice"
)
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
declare -a seeds2=(
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
for sd2 in "${seeds[@]}"; do \
# context: e+t w_e w_t
# reduce: attn
# transforms: 5
# hiddens: 64x2
bash script/counterfactual/utils/density/train_exom_from_causal_nf.sh\
    -n "$exp"\
    -pm nsf -pd 32x3 -ps $sd2\
    -s "$s"\
    -j $j\
    -m "$n"\
    -c "e+t w_e w_t"\
    -k "mb1 mb1 em"\
    -r attn -t 5 -d 64x2\
    -sv 1024 -bv 32\
    -e 200 -sd $sd\
    $@
done
done
done
done
done