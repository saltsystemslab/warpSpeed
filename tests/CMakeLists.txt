cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

message("Parsing tests")
#mainline tests
ConfigureExecutableHT(cache_test "${CMAKE_CURRENT_SOURCE_DIR}/src/cache_test.cu" "${HT_TESTS_BINARY_DIR}")


ConfigureExecutableHT(lock_test "${CMAKE_CURRENT_SOURCE_DIR}/src/lock_test.cu" "${HT_TESTS_BINARY_DIR}")

ConfigureExecutableHT(sawtooth_test "${CMAKE_CURRENT_SOURCE_DIR}/src/sawtooth_test.cu" "${HT_TESTS_BINARY_DIR}")
ConfigureExecutableHT(lf_test "${CMAKE_CURRENT_SOURCE_DIR}/src/lf_test.cu" "${HT_TESTS_BINARY_DIR}")

# target_link_libraries(cache_test PRIVATE bght)
# target_link_libraries(cache_test PRIVATE warpcore)
# target_link_libraries(cache_test PRIVATE hashing_project)
# target_link_libraries(cache_test PRIVATE ssl crypto)

