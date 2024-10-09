#!/bin/bash

exp="loglikelihood_effective"

# Iteration Range
declare -a nets=(
    "gmm"
    "maf"
    "nice"
    "sospf"
)
declare -a scms=(
    "simpson_nlin"
    "largebd_nlin"
    "napkin"
    "fairness_xw"
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
for sd in "${seeds[@]}"; do \
if [[ "$n" == "gmm" ]]; then t=10; else t=5; fi
if [[ "$n" == "sospf" ]] || [[ "$n" == "nice" ]]; then h=256x2; else h=64x2; fi
# j: 3
# context: e+t w_e w_t
# reduce: attn
# transforms: 5(gmm:10)
# hiddens: 64x2
# seed: 0
bash script/counterfactual/utils/sample/train_exom.sh\
    -n "$exp"\
    -s "$s"\
    -j 3\
    -m "$n"\
    -c "e+t w_e w_t"\
    -k "mb1 mb1 em"\
    -r attn -t $t -d $h\
    -sv 1024 -bv 32 -ev 10\
    -e 200 -sd $sd\
    $@
done
done
done