#!/bin/bash

exp="proxy_scm/gan_ncm"
declare -a scms=(
    "fairness"
    "fairness_xw"
    "fairness_xy"
    "fairness_yw"
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
for sd in "${seeds[@]}"; do \
    # net: nsf
    # hiddens: [32, 32]
    bash script/proxy_scm/gan_ncm/utils/train_gan_ncm.sh\
    -n "$exp"\
    -s "$s"\
    -d 64x2\
    -bv 16384 -ev 50\
    -e 500 -sd $sd\
    $@
done
done