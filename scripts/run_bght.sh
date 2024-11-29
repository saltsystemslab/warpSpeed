#!/bin/bash

./tests/phased_test -t bght_p2 -c 100000000

./tests/phased_probes -t bght_p2 -c 100000000

./tests/phased_test -t bght_cuckoo -c 100000000

./tests/phased_probes -t bght_cuckoo -c 100000000
