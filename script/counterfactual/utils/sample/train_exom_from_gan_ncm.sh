#!/bin/bash

(
    export PYTHONPATH="`pwd`":$PYTHONPATH
    python "`pwd`/script/counterfactual/utils/sample/train_exom_from_gan_ncm.py" $@
)