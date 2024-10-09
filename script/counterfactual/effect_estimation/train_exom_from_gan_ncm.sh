#!/bin/bash

exp="effect_estimation/gan_ncm"

# Iteration Range
declare -a nets=(
    "maf"
    "nice"
)
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
declare -a seeds2=(
    0
    7
    42
    3407
    65535
)

for n in "${nets[@]}"; do \
for s in "${scms[@]}"; do \
for q in "${qs[@]}"; do \
for sd in "${seeds[@]}"; do \
for sd2 in "${seeds[@]}"; do \
# context: e+t w_e w_t
# reduce: attn
# transforms: 5
# hiddens: 64x2
bash script/counterfactual/utils/effect/train_exom_from_gan_ncm.sh\
    -n "$exp"\
    -pd 64x2 -ps $sd2\
    -s "$s"\
    -q "$q"\
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