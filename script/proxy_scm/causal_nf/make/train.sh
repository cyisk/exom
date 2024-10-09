#!/bin/bash

exp="proxy_scm/causal_nf"
declare -a scms=(
    "chain_lin_3"
    "chain_lin_4"
    "chain_lin_5"
    "chain_nlin_3"
    "collider_lin"
    "fork_lin"
    "fork_nlin"
    "largebd_nlin"
    "simpson_nlin"
    "simpson_symprod"
    "triangle_lin"
    "triangle_nlin"
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
    # hiddens: [32, 32, 32]
    bash script/proxy_scm/causal_nf/utils/train_causal_nf.sh\
    -n "$exp"\
    -s "$s"\
    -m nsf -d 32x3\
    -bv 4096 -ev 10\
    -e 100 -sd $sd\
    $@
done
done