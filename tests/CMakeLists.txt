cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

message("Parsing tests")
#mainline tests
ConfigureExecutableHT(cache_test "${CMAKE_CURRENT_SOURCE_DIR}/src/cache_test.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(adversarial_test "${CMAKE_CURRENT_SOURCE_DIR}/src/adversarial_test.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(lf_test "${CMAKE_CURRENT_SOURCE_DIR}/src/lf_test.cu" "${HT_TESTS_BINARY_DIR}")
ConfigureExecutableHT(lf_probes "${CMAKE_CURRENT_SOURCE_DIR}/src/lf_probes.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(phased_test "${CMAKE_CURRENT_SOURCE_DIR}/src/phased_test.cu" "${HT_TESTS_BINARY_DIR}")
ConfigureExecutableHT(phased_probes "${CMAKE_CURRENT_SOURCE_DIR}/src/phased_probes.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(tile_combination_test "${CMAKE_CURRENT_SOURCE_DIR}/src/tile_combination_test.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(scaling_test "${CMAKE_CURRENT_SOURCE_DIR}/src/scaling_test.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(aging_independent "${CMAKE_CURRENT_SOURCE_DIR}/src/aging_independent.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(aging_probes "${CMAKE_CURRENT_SOURCE_DIR}/src/aging_probes.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(aging_combined "${CMAKE_CURRENT_SOURCE_DIR}/src/aging_combined.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(aging_micro "${CMAKE_CURRENT_SOURCE_DIR}/src/aging_micro.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(sparse_tensor_test "${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_tensor_test.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(ycsb_test "${CMAKE_CURRENT_SOURCE_DIR}/src/ycsb_test.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(sanity_test "${CMAKE_CURRENT_SOURCE_DIR}/src/sanity_test.cu" "${HT_TESTS_BINARY_DIR}")

