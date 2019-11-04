#!/bin/bash

/usr/local/cuda/bin/nvcc -o ${1} ${1}.cu;
if [ -z "$!" ]; then
./${1}
fi
