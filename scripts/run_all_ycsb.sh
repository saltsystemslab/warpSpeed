#!/bin/bash


../scripts/run_ycsb.sh workloada
../scripts/run_ycsb_cheap.sh workloada
../scripts/run_ycsb.sh workloadb
../scripts/run_ycsb_cheap.sh workloadb
../scripts/run_ycsb.sh workloadc
../scripts/run_ycsb_cheap.sh workloadc