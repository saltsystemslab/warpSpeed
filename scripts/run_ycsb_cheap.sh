#!/bin/bash

echo "First argument: $1"

./tests/ycsb_test --cheap -f $1 -t p2

./tests/ycsb_test --cheap -f $1 -t p2MD

./tests/ycsb_test --cheap -f $1 -t double

./tests/ycsb_test --cheap -f $1 -t doubleMD

./tests/ycsb_test --cheap -f $1 -t iceberg

./tests/ycsb_test --cheap -f $1 -t icebergMD

./tests/ycsb_test --cheap -f $1 -t cuckoo

./tests/ycsb_test --cheap -f $1 -t chaining