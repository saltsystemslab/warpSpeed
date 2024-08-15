## Concurrent GPU Hash Tables

This repository contains a set of concurrent, thread-safe hash tables for GPU operations. Each table exposes a similar API, and a set of tests are included for benchmarking the tables.


# API
--------------------

- `__device__ bool upsert_replace(const tile_type & my_tile, const Key & key, const Val & val)`: atomically insert the key-value pair `key, val` into the table. If `key` has already been inserted, replace the existing value with `val`.
- `__device__ bool upsert_function(const tile_type & my_tile, const Key & key, const Val & val, void (*replace_func)(packed_pair_type *, Key, Val))`: atomically insert the key value pair if it does not exist. If `key` has already been inserted, invokes the callback function `replace_func` on the key value pair stored in memory. A lock is held for the duration of the invocation, but `replace func` is responsible for ensuring memory coherency at the written address.
- `__device__ bool find_with_reference(tile_type my_tile, Key key, Val & val)`: Returns true if `key` is stored in the table. If `key` is found, `val` will be filled with the current value associated with the key. If the key is not found, the function returns false and does not modify `val`.
- `__device__ bool remove(tile_type my_tile, Key key)`: Deletes any key-val pair associated with `key` from the table, returns true if the key was present.

All functions come with a lockless variant for constructing compound operations. These operations guarantee coherency when run inside of a critical region (lock has been acquired), but do not enforce coherency without locking. The API for each lockless variant is identical to the main function but with an added `_no_lock`, so  `upsert_replace()` becomes `upsert_replace_no_lock()`. To acquire a lock, `__device__ uint64_t get_lock_bucket(tile_type my_tile, Key key)` can be used to determine the bucket associated with the key, and `__device__ void stall_lock(tile_type my_tile, uint64_t bucket)` and `__device__ void unlock(tile_type my_tile, uint64_t bucket)` are used to acquire and release the associated lock.



# Tables
------------------
Tables can be found in `include/hashing_project/tables`. The following tables are implemented

-  `Chaining`
-  `Cuckoo`
-  `Double Hashing`
-  `Double Hashing (Metadata)`
-  `Iceberg Hashing`
-  `Iceberg Hashing (Metadata)`
-  `Power-of-two-choice Hashing`
-  `Power-of-two-choice Hashing (Metadata)`


#Benchmarks
------------------

- `lf_test`: Load benchmark in the paper, tests the perfomance of the table from 5%-90% load.
- `sawtooth_test`: Aging benchmark in the paper, tests performance as the table has 1000 slices of data iteratively inserted/removed. Table is held at 85% load for the duration of the benchmark.
- `looped_cache_test`: Cache benchmark in the paper - tests the performance of the table when used for CPU-GPU data caching.
- `sparse_tensor_test`: Tests 1 and 3 mode contraction of a tensor with itself. The NIPS tensor is provided in the `dataset` folder. (must be unzipped first)
- `adversarial_test`: Runs the adversarial benchmark from the paper. This replays an attack on every bucket in a table. The attack attempts to trigger a race condition between insertion and deletion and exploit this race to emplace two copies of a key into a bucket. 