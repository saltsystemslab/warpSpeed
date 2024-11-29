#!/bin/bash

echo "First argument: $1"

./tests/adversarial_test -t $1 -c 1000000

./tests/lf_test -t $1 -c 100000000

./tests/lf_probes -t $1 -c 100000000

./tests/phased_test -t $1 -c 100000000

./tests/phased_probes -t $1 -c 100000000

./tests/tile_combination_test -t $1 -c 100000000

./tests/scaling_test -t $1 -c 10000000 -r 3 -s 10 

./tests/aging_combined -t $1 -c 100000000 -n 1000 -i .85 -r .01

./tests/aging_independent -t $1 -c 100000000 -n 1000 -i .85 -r .01

./tests/aging_probes -t $1 -c 100000000 -n 1000 -i .85 -r .01

./tests/cache_test -t $1 -n 1000000000 -h 100000000

./tests/sparse_tensor_test -t $1 -c 80000000 -r ../dataset/nips.tns