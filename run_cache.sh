#!/bin/bash

cd dataset

unzip nips.tns.zip

cd ..

./scripts/build.sh

cd build

./tests/cache_test -n 1000000000 -h 100000000 -z -t p2

./tests/cache_test -n 1000000000 -h 100000000 -z -t p2MD

./tests/cache_test -n 1000000000 -h 100000000 -z -t double

./tests/cache_test -n 1000000000 -h 100000000 -z -t doubleMD

./tests/cache_test -n 1000000000 -h 100000000 -z -t iceberg

./tests/cache_test -n 1000000000 -h 100000000 -z -t icebergMD

./tests/cache_test -n 1000000000 -h 100000000 -z -t chaining

cd ..