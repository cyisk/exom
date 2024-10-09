#!/bin/bash

(
    export PYTHONPATH="`pwd`":$PYTHONPATH
    python "`pwd`/script/proxy_scm/gan_ncm/utils/train_gan_ncm.py" $@
)