#!/bin/bash

(
    export PYTHONPATH="`pwd`":$PYTHONPATH
    python "`pwd`/script/proxy_scm/causal_nf/utils/train_causal_nf.py" $@
)