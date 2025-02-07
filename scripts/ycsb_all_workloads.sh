#!/bin/bash

echo "First argument: $1"

./tests/ycsb_test -t $1 --cheap_insert -f workloada
./tests/ycsb_test -t $1 --cheap_insert -f workloadb
./tests/ycsb_test -t $1 --cheap_insert -f workloadc
