#!/bin/bash

(
    export PYTHONPATH="`pwd`":$PYTHONPATH
    python "`pwd`/script/counterfactual/utils/sample/train_exom_from_causal_nf.py" $@
)