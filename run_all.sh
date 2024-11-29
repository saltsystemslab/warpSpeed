#!/bin/bash

cd dataset

unzip nips.tns.zip

cd ..

./scripts/build.sh

cd build

../scripts/run_stable.sh p2

../scripts/run_stable.sh p2MD

../scripts/run_stable.sh double

../scripts/run_stable.sh doubleMD

../scripts/run_stable.sh iceberg

../scripts/run_stable.sh icebergMD

../scripts/run_unstable.sh cuckoo

../scripts/run_stable.sh chaining

../scripts/run_bght.sh

cd ..