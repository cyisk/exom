#!/bin/bash

(
    export PYTHONPATH="`pwd`":$PYTHONPATH
    python "`pwd`/script/counterfactual/utils/effect/train_rs_from_gan_ncm.py" $@
)