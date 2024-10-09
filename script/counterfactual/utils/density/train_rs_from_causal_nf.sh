#!/bin/bash

(
    export PYTHONPATH="`pwd`":$PYTHONPATH
    python "`pwd`/script/counterfactual/utils/density/train_rs_from_causal_nf.py" $@
)