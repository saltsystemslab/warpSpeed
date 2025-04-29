#!/bin/bash

git submodule update --init --recursive
git submodule update --recursive


echo "patching cmake instructions"
rm SlabHash/CMakeLists.txt
cp patches/slabhash_cmake_patch.txt SlabHash/CMakeLists.txt

rm BGHT/CMakeLists.txt
cp patches/bght_cmake_patch.txt BGHT/CMakeLists.txt


echo "copying SlabAlloc into namespace"

cp SlabHash/SlabAlloc/src/slab_alloc.cuh SlabHash/src/slab_alloc.cuh
cp SlabHash/SlabAlloc/src/slab_alloc_global.cuh SlabHash/src/slab_alloc_global.cuh
cp patches/patched_cmap.cuh SlabHash/src/concurrent_map/cmap_class.cuh


mkdir build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5

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

make fill_test &

wait

cd ..