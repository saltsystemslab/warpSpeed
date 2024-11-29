#!/bin/bash

mkdir build
cd build
cmake ..


make adversarial_test &

make lf_test &

make lf_probes &

make phased_test &

make phased_probes &

make phased_probes & 

make tile_combination_test &


make scaling_test &

make aging_combined &

make aging_independent &

make aging_probes &

make cache_test &

make sparse_tensor_test &

wait

cd ..