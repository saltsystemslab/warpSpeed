#!/bin/bash

echo "First argument: $1"

./tests/sparse_tensor_test -c 80000000 -r ../dataset/nips.tns -t p2

./tests/sparse_tensor_test -c 80000000 -r ../dataset/nips.tns -t p2MD

./tests/sparse_tensor_test -c 80000000 -r ../dataset/nips.tns -t double

./tests/sparse_tensor_test -c 80000000 -r ../dataset/nips.tns -t doubleMD

./tests/sparse_tensor_test -c 80000000 -r ../dataset/nips.tns -t iceberg

./tests/sparse_tensor_test -c 80000000 -r ../dataset/nips.tns -t icebergMD

./tests/sparse_tensor_test -c 80000000 -r ../dataset/nips.tns -t chaining