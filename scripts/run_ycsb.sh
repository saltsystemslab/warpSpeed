#!/bin/bash

echo "First argument: $1"

./tests/ycsb_test -f $1 --cheap_insert -t p2

./tests/ycsb_test -f $1 --cheap_insert -t p2MD

./tests/ycsb_test -f $1 --cheap_insert -t double

./tests/ycsb_test -f $1 --cheap_insert -t doubleMD

./tests/ycsb_test -f $1 --cheap_insert -t iceberg

./tests/ycsb_test -f $1 --cheap_insert -t icebergMD

./tests/ycsb_test -f $1 --cheap_insert -t cuckoo

./tests/ycsb_test -f $1 --cheap_insert -t chaining