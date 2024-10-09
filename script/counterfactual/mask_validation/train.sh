#!/bin/bash

exp="mask_validation"

# Comparison target
declare -a masks=(
    "em em em"
    "mb mb em"
    "mb1 mb1 em"
    "mb2 mb2 em"
)

# Comparison Range
declare -a nets=(
    #"gmm"
    #"maf"
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
for m in "${masks[@]}"; do \
for sd in "${seeds[@]}"; do \
if [[ "$n" == "gmm" ]]; then t=10; else t=5; fi
if [[ "$n" == "sospf" ]] || [[ "$n" == "nice" ]]; then h=256x2; else h=64x2; fi
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
    -k "$m"\
    -r attn -t $t -d $h\
    -sv 1024 -bv 32\
    -e 200 -sd $sd\
    $@
done
done
done
done
done