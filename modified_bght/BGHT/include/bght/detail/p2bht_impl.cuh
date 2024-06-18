/*
 *   Copyright 2021 The Regents of the University of California, Davis
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#pragma once
#include <cooperative_groups.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <bght/detail/benchmark_metrics.cuh>
#include <bght/detail/bucket.cuh>
#include <bght/detail/kernels.cuh>
#include <bght/detail/rng.hpp>
#include <iterator>
#include <bght/p2bht.hpp>
#include <random>

#define SHORTCUT_CUTOFF .75

#ifndef P2_PRINT
#define P2_PRINT 0
#endif

namespace bght {


#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits) ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SET_BIT_MASK(index) ((1ULL << index))


template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::p2bht(std::size_t capacity,
                                                          Key empty_key_sentinel,
                                                          Key empty_key_tombstone,
                                                          T empty_value_sentinel,
                                                          Allocator const& allocator)
    : capacity_{std::max(capacity, std::size_t{1})}
    , sentinel_key_{empty_key_sentinel}
    , tombstone_key_{empty_key_tombstone}
    , sentinel_value_{empty_value_sentinel}
    , allocator_{allocator}
    , atomic_pairs_allocator_{allocator}
    , pool_allocator_{allocator}
    , size_type_allocator_{allocator} {
  // capacity_ must be multiple of bucket size
  auto remainder = capacity_ % bucket_size;
  if (remainder) {
    capacity_ += (bucket_size - remainder);
  }
  num_buckets_ = capacity_ / bucket_size;
  d_table_ = std::allocator_traits<atomic_pair_allocator_type>::allocate(
      atomic_pairs_allocator_, capacity_);
  table_ =
      std::shared_ptr<atomic_pair_type>(d_table_, bght::cuda_deleter<atomic_pair_type>());

  d_build_success_ =
      std::allocator_traits<pool_allocator_type>::allocate(pool_allocator_, 1);
  build_success_ = std::shared_ptr<bool>(d_build_success_, bght::cuda_deleter<bool>());

  value_type empty_pair{sentinel_key_, sentinel_value_};

  thrust::fill(thrust::device, d_table_, d_table_ + capacity_, empty_pair);

  std::mt19937 rng(2);
  hf0_ = initialize_hf<hasher>(rng);
  hf1_ = initialize_hf<hasher>(rng);

  bool success = true;
  cuda_try(cudaMemcpy(d_build_success_, &success, sizeof(bool), cudaMemcpyHostToDevice));
}


template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
__host__ p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B> * p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::generate_on_device(std::size_t capacity,
                                                          Key empty_key_sentinel,
                                                          Key empty_key_tombstone,
                                                          T empty_value_sentinel,
                                                          Allocator const& allocator){



  using table_type = p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>;

  table_type * host_version;

  cudaMallocHost((void **)&host_version, sizeof(table_type));

  host_version->capacity_ = std::max(capacity, std::size_t{1});
  host_version->sentinel_key_ = empty_key_sentinel;
  host_version->tombstone_key_ = empty_key_tombstone;
  host_version->sentinel_value_ = empty_value_sentinel;
  host_version->allocator_ = allocator;
  host_version->atomic_pairs_allocator_= allocator;
  host_version->pool_allocator_ = allocator;
  host_version->size_type_allocator_ = allocator;

  // capacity_ must be multiple of bucket size
  auto remainder = host_version->capacity_ % B;
  if (remainder) {
    host_version->capacity_ += (B - remainder);
  }
  host_version->num_buckets_ = host_version->capacity_ / B;
  host_version->d_table_ = std::allocator_traits<atomic_pair_allocator_type>::allocate(
      host_version->atomic_pairs_allocator_, host_version->capacity_);
  host_version->table_ =
      std::shared_ptr<atomic_pair_type>(host_version->d_table_, bght::cuda_deleter<atomic_pair_type>());

  uint64_t n_lock_uints = (host_version->num_buckets_-1)/64+1;

  uint64_t * dev_locks;


  //not sure what the spec is for their allocator, just using cudaMalloc to save time.
  cudaMalloc((void **)&dev_locks, (n_lock_uints*sizeof(uint64_t)));

  cudaMemset(dev_locks, 0, n_lock_uints*sizeof(uint64_t));

  host_version->locks = dev_locks;

  host_version->d_build_success_ =
      std::allocator_traits<pool_allocator_type>::allocate(host_version->pool_allocator_, 1);
  host_version->build_success_ = std::shared_ptr<bool>(host_version->d_build_success_, bght::cuda_deleter<bool>());

  value_type empty_pair{host_version->sentinel_key_, host_version->sentinel_value_};

  thrust::fill(thrust::device, host_version->d_table_, host_version->d_table_ + host_version->capacity_, empty_pair);

  std::mt19937 rng(2);
  host_version->hf0_ = initialize_hf<hasher>(rng);
  host_version->hf1_ = initialize_hf<hasher>(rng);

  bool success = true;
  cuda_try(cudaMemcpy(host_version->d_build_success_, &success, sizeof(bool), cudaMemcpyHostToDevice));


  table_type * dev_version;
  cudaMalloc((void **)&dev_version, sizeof(table_type));

  cudaMemcpy(dev_version, host_version, sizeof(table_type), cudaMemcpyHostToDevice);
  cudaFreeHost(host_version);

  return dev_version;

}


template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::p2bht(const p2bht& other)
    : capacity_(other.capacity_)
    , sentinel_key_(other.sentinel_key_)
    , tombstone_key_(other.tombstone_key_)
    , sentinel_value_(other.sentinel_value_)
    , allocator_(other.allocator_)
    , atomic_pairs_allocator_(other.atomic_pairs_allocator_)
    , pool_allocator_(other.pool_allocator_)
    , d_table_(other.d_table_)
    , table_(other.table_)
    , d_build_success_(other.d_build_success_)
    , build_success_(other.build_success_)
    , hf0_(other.hf0_)
    , hf1_(other.hf1_)
    , num_buckets_(other.num_buckets_) {}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::~p2bht() {}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
void p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::clear() {
  value_type empty_pair{sentinel_key_, sentinel_value_};
  thrust::fill(thrust::device, d_table_, d_table_ + capacity_, empty_pair);
  bool success = true;
  cuda_try(cudaMemcpy(d_build_success_, &success, sizeof(bool), cudaMemcpyHostToDevice));
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
template <typename InputIt>
bool p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::insert(InputIt first,
                                                                InputIt last,
                                                                cudaStream_t stream) {
  const auto num_keys = std::distance(first, last);

  const uint32_t block_size = 128;
  const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
  detail::kernels::tiled_insert_kernel<<<num_blocks, block_size, 0, stream>>>(
      first, last, *this);
  bool success;
  cuda_try(cudaMemcpyAsync(
      &success, d_build_success_, sizeof(bool), cudaMemcpyDeviceToHost, stream));
  return success;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
template <typename InputIt, typename OutputIt>
void p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::find(InputIt first,
                                                              InputIt last,
                                                              OutputIt output_begin,
                                                              cudaStream_t stream) {
  const auto num_keys = last - first;
  const uint32_t block_size = 128;
  const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;

  detail::kernels::tiled_find_kernel<<<num_blocks, block_size, 0, stream>>>(
      first, last, output_begin, *this);
}


template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ bool bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::insert(
    value_type const& pair,
    tile_type const& tile) {

  using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;

  const int elected_lane = 0;
  auto lane_id = tile.thread_rank();
  auto bucket0_id = hf0_(pair.first) % num_buckets_;
  auto bucket1_id = hf1_(pair.first) % num_buckets_;


  if (key_equal{}(pair.first, sentinel_key_)) {
    return false;
  }

  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  value_type tombstone_pair{tombstone_key_, sentinel_value_};

  //broadcast from 0 just in case
  //insertion_pair = tile.shfl(insertion_pair, 0);

  bucket_type bucket(&d_table_[bucket0_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  bucket.load(cuda::memory_order_relaxed, tile);

  //make a stab, why not
  if (bucket.upsert(pair, key_equal{}, tile)) return true;

  //main load determines shortcutting
  //tombstone load determines insert positions/fill.
  int shortcut_load = bucket.compute_load(sentinel_pair, tile);
  int main_load = bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);

  while(1.0*shortcut_load/B < SHORTCUT_CUTOFF){

    //shortcutting
    bool success = bucket.insert_tombstone(pair, key_equal{}, sentinel_pair, tombstone_pair, tile);

    if (success) return true;

   shortcut_load = bucket.compute_load(sentinel_pair, tile);


  }

  //load alt bucket

  bucket_type alt_bucket(&d_table_[bucket1_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  alt_bucket.load(cuda::memory_order_relaxed, tile);


  int alt_load = alt_bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);

  if (alt_bucket.upsert(pair, key_equal{}, tile)) return true;
 
  do {
    //bucket_type bucket(&d_table_[bucket0_id * bucket_size], tile);
    // bucket.load(cuda::memory_order_relaxed, tile);
    // int load = bucket.compute_load(sentinel_pair);
    // INCREMENT_PROBES_IN_TILE
    // bucket_type bucket1(&d_table_[bucket1_id * bucket_size], tile);
    // bucket1.load(cuda::memory_order_relaxed);
    // int load1 = bucket1.compute_load(sentinel_pair);
    // INCREMENT_PROBES_IN_TILE
    if (alt_load < main_load) {
      main_load = alt_load;
      bucket = alt_bucket;
    } else if (main_load == bucket_size && alt_load == bucket_size) {
      return false;
    }


    bool cas_success = bucket.insert_tombstone(pair, key_equal{}, sentinel_pair, tombstone_pair, tile);

    // bucket is not full
    // bool cas_success = false;
    // if (lane_id == elected_lane) {
    //   cas_success = bucket.strong_cas_at_location(pair,
    //                                               main_load,
    //                                               sentinel_pair,
    //                                               cuda::memory_order_relaxed,
    //                                               cuda::memory_order_relaxed);
    // }
    // cas_success = tile.shfl(cas_success, elected_lane);


    if (cas_success) {
      return true;
    }

    //RELOAD
    bucket_type bucket_copy(&d_table_[bucket0_id * bucket_size]);
    INCREMENT_PROBES_IN_TILE
    bucket = bucket_copy;
    bucket.load(cuda::memory_order_relaxed, tile);
    main_load = bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);



    bucket_type alt_bucket_copy(&d_table_[bucket1_id * bucket_size]);
    INCREMENT_PROBES_IN_TILE
    alt_bucket = alt_bucket_copy;
    alt_bucket.load(cuda::memory_order_relaxed, tile);
    alt_load = alt_bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);



  } while (true);
  return false;
}






//generic, collision-free upsert
//uses locks :( but in exchange any chain of upserts is resolved in a linearizable order.
// template <class Key,
//           class T,
//           class Hash,
//           class KeyEqual,
//           cuda::thread_scope Scope,
//           class Allocator,
//           int B>
// __device__ bool bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::upsert_generic(
//     value_type const& pair,
//     tile_type const& tile) {

//   using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;

//   const int elected_lane = 0;
//   auto lane_id = tile.thread_rank();
//   auto bucket0_id = hf0_(pair.first) % num_buckets_;
//   auto bucket1_id = hf1_(pair.first) % num_buckets_;

//   //for fairness all implementations lock all buckets.

//   if (key_equal{}(pair.first, sentinel_key_) || key_equal{}(pair.first, tombstone_key_)) {
//     return false;
//   }


//   //upsert procedure
//   //load first bucket

//   //below cutoff
//   // - no lock needed - try upsert, try regular insert.
//   // - else load alt bucket
//   // - try upsert
//   // - now need lock.

//   //for dummy impl late at night, always lock.
//   //lock_buckets(tile, bucket0_id, bucket1_id);
//   //unlock_buckets(tile, bucket0_id, bucket1_id);

//   value_type sentinel_pair{sentinel_key_, sentinel_value_};
//   value_type tombstone_pair{tombstone_key_, sentinel_value_};

//   //broadcast from 0 just in case
//   //insertion_pair = tile.shfl(insertion_pair, 0);

//   bucket_type primary_bucket(&d_table_[bucket0_id * bucket_size]);
//   INCREMENT_PROBES_IN_TILE
//   primary_bucket.load(cuda::memory_order_relaxed, tile);

//   //make a stab, why not
//   if (primary_bucket.upsert(pair, key_equal{}, tile)){
    
//     // unlock(tile, bucket0_id);
//     // unlock(tile, bucket1_id);
//     //unlock_buckets(tile, bucket0_id, bucket1_id);
//     return true;

//   }

//   //main load determines shortcutting
//   //tombstone load determines insert positions/fill.
//   int shortcut_load = primary_bucket.compute_load(sentinel_pair, tile);
  
//   //should always complete in one loop.
//   while(1.0*shortcut_load/B < SHORTCUT_CUTOFF){


//     bool cas_success = false;
//     if (lane_id == elected_lane) {
//       cas_success = primary_bucket.strong_cas_at_location(pair,
//                                                   shortcut_load,
//                                                   sentinel_pair,
//                                                   cuda::memory_order_relaxed,
//                                                   cuda::memory_order_relaxed);
//     }

//     cas_success = tile.shfl(cas_success, elected_lane);
//     //bool success = bucket.insert_sentinel(pair, key_equal{}, sentinel_pair, tile);

//     if (cas_success){

//       //unlock_buckets(tile, bucket0_id, bucket1_id);
//       return true;
//     }

//     primary_bucket.load(cuda::memory_order_relaxed, tile);
//     shortcut_load = primary_bucket.compute_load(sentinel_pair, tile);


//   }

//   //load alt bucket

//   bucket_type alt_bucket(&d_table_[bucket1_id * bucket_size]);
//   INCREMENT_PROBES_IN_TILE
//   alt_bucket.load(cuda::memory_order_relaxed, tile);

//   //int alt_load = alt_bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);

//   if (alt_bucket.upsert(pair, key_equal{}, tile)){

//     unlock_buckets(tile, bucket0_id, bucket1_id);
//     return true;

//   }

//   lock_buckets(tile, bucket0_id, bucket1_id);
  
//   // if (bucket0_id < bucket1_id){

//   //   stall_lock(tile, bucket0_id);
//   //   stall_lock(tile, bucket1_id);

//   // } else {
//   //   stall_lock(tile, bucket1_id);
//   //   stall_lock(tile, bucket0_id);
//   // }

//  // main_load = bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);


//   do {

//     bucket_type bucket(&d_table_[bucket0_id * bucket_size]);
//     bucket.load(cuda::memory_order_relaxed, tile);
//     int main_load = bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);
//     INCREMENT_PROBES_IN_TILE

//     alt_bucket.load(cuda::memory_order_relaxed, tile);
//     int alt_load = bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);
//     // bucket_type bucket1(&d_table_[bucket1_id * bucket_size], tile);
//     // bucket1.load(cuda::memory_order_relaxed);
//     // int load1 = bucket1.compute_load(sentinel_pair);
//     INCREMENT_PROBES_IN_TILE

//     if (alt_load < main_load) {
//       main_load = alt_load;
//       bucket = alt_bucket;
//     } else if (main_load == bucket_size && alt_load == bucket_size) {

//       //printf("Both buckets full\n");
//       unlock_buckets(tile, bucket0_id, bucket1_id);
//       return false;
//     }


//     bool cas_success = bucket.insert_tombstone(pair, key_equal{}, sentinel_pair, tombstone_pair, tile);

//     // bucket is not full
//     // bool cas_success = false;
//     // if (lane_id == elected_lane) {
//     //   cas_success = bucket.strong_cas_at_location(pair,
//     //                                               main_load,
//     //                                               sentinel_pair,
//     //                                               cuda::memory_order_relaxed,
//     //                                               cuda::memory_order_relaxed);
//     // }
//     // cas_success = tile.shfl(cas_success, elected_lane);


//     if (cas_success) {

//       unlock_buckets(tile, bucket0_id, bucket1_id);
//       return true;
//     }

//     //printf("Looping in main loop\n");

//     //RELOAD
//     // bucket_type bucket_copy(&d_table_[bucket0_id * bucket_size]);
//     // INCREMENT_PROBES_IN_TILE
//     // bucket = bucket_copy;
//     // bucket.load(cuda::memory_order_relaxed, tile);
//     // main_load = bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);



//     // bucket_type alt_bucket_copy(&d_table_[bucket1_id * bucket_size]);
//     // INCREMENT_PROBES_IN_TILE
//     // alt_bucket = alt_bucket_copy;
//     // alt_bucket.load(cuda::memory_order_relaxed, tile);
//     // alt_load = alt_bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);

//   } while (true);


//   unlock_buckets(tile, bucket0_id, bucket1_id);
//   return false;
// }


//direct copy of insert that does work.
// template <class Key,
//           class T,
//           class Hash,
//           class KeyEqual,
//           cuda::thread_scope Scope,
//           class Allocator,
//           int B>
// __device__ bool bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::upsert_generic(
//     value_type const& pair,
//     tile_type const& tile) {

//   using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;

//   const int elected_lane = 0;
//   auto lane_id = tile.thread_rank();
//   auto bucket0_id = hf0_(pair.first) % num_buckets_;
//   auto bucket1_id = hf1_(pair.first) % num_buckets_;

//   //for fairness all implementations lock all buckets.

//   if (key_equal{}(pair.first, sentinel_key_) || key_equal{}(pair.first, tombstone_key_)) {
//     return false;
//   }


//   //upsert procedure
//   //load first bucket


//   value_type sentinel_pair{sentinel_key_, sentinel_value_};
//   value_type tombstone_pair{tombstone_key_, sentinel_value_};

//   //broadcast from 0 just in case
//   //insertion_pair = tile.shfl(insertion_pair, 0);

//   bucket_type bucket(&d_table_[bucket0_id * bucket_size]);
//   INCREMENT_PROBES_IN_TILE
//   bucket.load(cuda::memory_order_relaxed, tile);


//   //make a stab, why not
//   //if bucket contains then drop - cannot insert.
//   if (bucket.compute_load(pair, tile) != B) return false;

//   //main load determines shortcutting
//   //tombstone load determines insert positions/fill.
//   int main_load = bucket.compute_load(sentinel_pair, tile);
//   //int main_load = bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);

//   while(1.0*main_load/B < SHORTCUT_CUTOFF){

//     //shortcutting


//     bool cas_success = false;
//     if (lane_id == elected_lane) {
//       cas_success = bucket.strong_cas_at_location(pair,
//                                                   main_load,
//                                                   sentinel_pair,
//                                                   cuda::memory_order_relaxed,
//                                                   cuda::memory_order_relaxed);
//     }

//     cas_success = tile.shfl(cas_success, elected_lane);
//     //bool success = bucket.insert_sentinel(pair, key_equal{}, sentinel_pair, tile);

//     if (cas_success) return true;

//     bucket.load(cuda::memory_order_relaxed, tile);

//     if (bucket.compute_load(pair, tile) != B) return false;

//     main_load = bucket.compute_load(sentinel_pair, tile);


//     #if P2_PRINT
//     if (tile.thread_rank() == 0) printf("Looping in shortcut %d\n", main_load);
//     #endif


//   }


//   //load alt bucket

//   bucket_type alt_bucket(&d_table_[bucket1_id * bucket_size]);
//   INCREMENT_PROBES_IN_TILE
//   alt_bucket.load(cuda::memory_order_relaxed, tile);


//   int alt_load = alt_bucket.compute_load(sentinel_pair, tile);


//   if (alt_bucket.compute_load(pair, tile) != B) return false;
//   //if (alt_bucket.upsert(pair, key_equal{})) return true;


 
//   do {

//     #if P2_PRINT
//     if (tile.thread_rank() == 0) printf("Looping in insert loop\n");
//     #endif
//     //bucket_type bucket(&d_table_[bucket0_id * bucket_size], tile);
//     // bucket.load(cuda::memory_order_relaxed, tile);
//     // int load = bucket.compute_load(sentinel_pair);
//     // INCREMENT_PROBES_IN_TILE
//     // bucket_type bucket1(&d_table_[bucket1_id * bucket_size], tile);
//     // bucket1.load(cuda::memory_order_relaxed);
//     // int load1 = bucket1.compute_load(sentinel_pair);
//     // INCREMENT_PROBES_IN_TILE
//     if (alt_load < main_load) {
//       main_load = alt_load;
//       bucket = alt_bucket;
//     } else if (main_load == bucket_size && alt_load == bucket_size) {
        
//        continue;
//     }

//     //bool cas_success = bucket.insert_tombstone(pair, key_equal{}, sentinel_pair, tombstone_pair, tile);

//     //bucket is not full
//     bool cas_success = false;
//     if (lane_id == elected_lane) {
//       cas_success = bucket.strong_cas_at_location(pair,
//                                                   main_load,
//                                                   sentinel_pair,
//                                                   cuda::memory_order_relaxed,
//                                                   cuda::memory_order_relaxed);
//     }
//     cas_success = tile.shfl(cas_success, elected_lane);


//     if (cas_success) {
//       return true;
//     }

//     //RELOAD
//     bucket_type bucket_copy(&d_table_[bucket0_id * bucket_size]);
//     INCREMENT_PROBES_IN_TILE
//     bucket = bucket_copy;
//     bucket.load(cuda::memory_order_relaxed, tile);
//     main_load = bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);



//     bucket_type alt_bucket_copy(&d_table_[bucket1_id * bucket_size]);
//     INCREMENT_PROBES_IN_TILE
//     alt_bucket = alt_bucket_copy;
//     alt_bucket.load(cuda::memory_order_relaxed, tile);
//     alt_load = alt_bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);


//     if (bucket.compute_load(pair, tile) != B) return false;
//     if (alt_bucket.compute_load(pair, tile) != B) return false;

//   } while (main_load != bucket_size && alt_load != bucket_size);

//   #if P2_PRINT
//   if (tile.thread_rank() == 0) printf("swap-out code triggers here.\n");
//   #endif
//   //upsert functionality.
//   return false;
// }


template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ bool bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::upsert_generic(
    value_type const& pair,
    tile_type const& tile) {

  using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;

  const int elected_lane = 0;
  auto lane_id = tile.thread_rank();
  auto bucket0_id = hf0_(pair.first) % num_buckets_;
  auto bucket1_id = hf1_(pair.first) % num_buckets_;

  //for fairness all implementations lock all buckets.

  if (key_equal{}(pair.first, sentinel_key_) || key_equal{}(pair.first, tombstone_key_)) {
    return false;
  }


  //upsert procedure
  //load first bucket


  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  value_type tombstone_pair{tombstone_key_, sentinel_value_};

  //broadcast from 0 just in case
  //insertion_pair = tile.shfl(insertion_pair, 0);

  bucket_type bucket(&d_table_[bucket0_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  bucket.load(cuda::memory_order_relaxed, tile);


  //make a stab, why not
  //if bucket contains then drop - cannot insert.

  {

    auto prev_key_location = bucket.find_key_location(pair.first, key_equal{}, tile);

    if (prev_key_location != -1){


          bool cas_success = bucket.upsert_at_location(pair, prev_key_location, tile);

          if (cas_success){
            return true;
          }
    }

  }

  //main load determines shortcutting
  //tombstone load determines insert positions/fill.
  int main_load = bucket.compute_load(sentinel_pair, tile);
  //int main_load = bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);

  while(1.0*main_load/B < SHORTCUT_CUTOFF){

    //shortcutting


    bool cas_success = false;
    if (lane_id == elected_lane) {
      cas_success = bucket.strong_cas_at_location(pair,
                                                  main_load,
                                                  sentinel_pair,
                                                  cuda::memory_order_relaxed,
                                                  cuda::memory_order_relaxed);
    }

    cas_success = tile.shfl(cas_success, elected_lane);
    //bool success = bucket.insert_sentinel(pair, key_equal{}, sentinel_pair, tile);

    if (cas_success) return true;

    bucket.load(cuda::memory_order_relaxed, tile);


    {

      auto prev_key_location = bucket.find_key_location(pair.first, key_equal{}, tile);

      if (prev_key_location != -1){


            bool cas_success = bucket.upsert_at_location(pair, prev_key_location, tile);

            if (cas_success){
              return true;
            }
      }

    } 

    main_load = bucket.compute_load(sentinel_pair, tile);


    //if (tile.thread_rank() == 0) printf("Looping in shortcut %d\n", main_load);



  }

  //load alt bucket

  bucket_type alt_bucket(&d_table_[bucket1_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  alt_bucket.load(cuda::memory_order_relaxed, tile);


  int alt_load = alt_bucket.compute_load(sentinel_pair, tile);


  //if (alt_bucket.compute_load(pair, tile) != B) return false;
  //if (alt_bucket.upsert(pair, key_equal{})) return true;


  lock_buckets(tile, bucket0_id, bucket1_id);

  bool should_loop;
 
  do {

    should_loop = false;


    //if (tile.thread_rank() == 0) printf("Looping in insert loop %d vs %d\n", main_load, alt_load);


    //check for upsert
    {

      auto prev_key_location = bucket.find_key_location(pair.first, key_equal{}, tile);

      if (prev_key_location != -1){

        should_loop = true;

        bool cas_success = bucket.upsert_at_location(pair, prev_key_location, tile);

        if (cas_success){
          unlock_buckets(tile, bucket0_id, bucket1_id);
          return true;
        }


      }

      auto alt_prev_key_location = alt_bucket.find_key_location(tombstone_pair.first, key_equal{}, tile);

      if (alt_prev_key_location != -1){

        should_loop = true;

        bool cas_success = alt_bucket.upsert_at_location(pair, alt_prev_key_location, tile);

        if (cas_success){
          unlock_buckets(tile, bucket0_id, bucket1_id);
          return true;
        }

      }

    }


    //attempt tombstone update
    {

      auto tomb_location = bucket.find_key_location(tombstone_pair.first, key_equal{}, tile);

      if (tomb_location != -1){

        should_loop = true;

        bool cas_success = bucket.replace_tombstone_at_location(pair, tombstone_pair, key_equal{}, tile);
            // bool cas_success = false;
            // if (lane_id == elected_lane) {
            //   cas_success = bucket.strong_cas_at_location(pair,
            //                                               tomb_location,
            //                                               tombstone_pair,
            //                                               cuda::memory_order_relaxed,
            //                                               cuda::memory_order_relaxed);
            // }

            // cas_success = tile.shfl(cas_success, elected_lane);

        if (cas_success){
          unlock_buckets(tile, bucket0_id, bucket1_id);
          return true;
        }
      }

      auto alt_tomb_location = alt_bucket.find_key_location(tombstone_pair.first, key_equal{}, tile);

      if (alt_tomb_location != -1){

        should_loop = true;


        bool cas_success = alt_bucket.replace_tombstone_at_location(pair, tombstone_pair, key_equal{}, tile);

        // bool cas_success = false;
        // if (lane_id == elected_lane) {
        //   cas_success = alt_bucket.strong_cas_at_location(pair,
        //                                               alt_tomb_location,
        //                                               tombstone_pair,
        //                                               cuda::memory_order_relaxed,
        //                                               cuda::memory_order_relaxed);
        // }

        // cas_success = tile.shfl(cas_success, elected_lane);

        if (cas_success){
          unlock_buckets(tile, bucket0_id, bucket1_id);
          return true;
        }

      }




    }

    //bucket_type bucket(&d_table_[bucket0_id * bucket_size], tile);
    // bucket.load(cuda::memory_order_relaxed, tile);
    // int load = bucket.compute_load(sentinel_pair);
    // INCREMENT_PROBES_IN_TILE
    // bucket_type bucket1(&d_table_[bucket1_id * bucket_size], tile);
    // bucket1.load(cuda::memory_order_relaxed);
    // int load1 = bucket1.compute_load(sentinel_pair);
    // INCREMENT_PROBES_IN_TILE
    if (alt_load < main_load) {
      main_load = alt_load;
      bucket = alt_bucket;

    }
    // } else if (main_load == bucket_size && alt_load == bucket_size) {
        
    //    continue;
    // }

    if (main_load < bucket_size || alt_load < bucket_size){
       should_loop = true;
    }

    //bool cas_success = bucket.insert_tombstone(pair, key_equal{}, sentinel_pair, tombstone_pair, tile);

    //bucket is not full
    bool cas_success = false;
    if (lane_id == elected_lane) {
      cas_success = bucket.strong_cas_at_location(pair,
                                                  main_load,
                                                  sentinel_pair,
                                                  cuda::memory_order_relaxed,
                                                  cuda::memory_order_relaxed);
    }
    cas_success = tile.shfl(cas_success, elected_lane);


    if (cas_success) {
      unlock_buckets(tile, bucket0_id, bucket1_id);
      return true;
    }

    //RELOAD
    bucket_type bucket_copy(&d_table_[bucket0_id * bucket_size]);
    INCREMENT_PROBES_IN_TILE
    bucket = bucket_copy;
    bucket.load(cuda::memory_order_relaxed, tile);
    main_load = bucket.compute_load(sentinel_pair, tile);



    bucket_type alt_bucket_copy(&d_table_[bucket1_id * bucket_size]);
    INCREMENT_PROBES_IN_TILE
    alt_bucket = alt_bucket_copy;
    alt_bucket.load(cuda::memory_order_relaxed, tile);
    alt_load = alt_bucket.compute_load(sentinel_pair, tile);

  } while (should_loop);

  #if P2_PRINT
  if (tile.thread_rank() == 0) printf("swap-out code triggers here.\n");
  #endif
  //upsert functionality.
  unlock_buckets(tile, bucket0_id, bucket1_id);
  return false;
}

//insert if and only if a previous version does not exit.
template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ bool bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::insert_exact(
    tile_type const& tile,
    key_type insert_key,
    mapped_type insert_val
    ) {

  using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;

  value_type pair{insert_key, insert_val};

  const int elected_lane = 0;
  auto lane_id = tile.thread_rank();
  auto bucket0_id = hf0_(pair.first) % num_buckets_;
  auto bucket1_id = hf1_(pair.first) % num_buckets_;


  if (key_equal{}(pair.first, sentinel_key_) || key_equal{}(pair.first, tombstone_key_)) {
    return false;
  }


  

  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  value_type tombstone_pair{tombstone_key_, sentinel_value_};

  //broadcast from 0 just in case
  //insertion_pair = tile.shfl(insertion_pair, 0);

  bucket_type bucket(&d_table_[bucket0_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  bucket.load(cuda::memory_order_relaxed, tile);


  //make a stab, why not
  //if bucket contains then drop - cannot insert.
  if (bucket.compute_load(pair, tile) != B) return false;

  //main load determines shortcutting
  //tombstone load determines insert positions/fill.
  int main_load = bucket.compute_load(sentinel_pair, tile);
  //int main_load = bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);

  while(1.0*main_load/B < SHORTCUT_CUTOFF){

    //shortcutting


    bool cas_success = false;
    if (lane_id == elected_lane) {
      cas_success = bucket.strong_cas_at_location(pair,
                                                  main_load,
                                                  sentinel_pair,
                                                  cuda::memory_order_relaxed,
                                                  cuda::memory_order_relaxed);
    }

    cas_success = tile.shfl(cas_success, elected_lane);
    //bool success = bucket.insert_sentinel(pair, key_equal{}, sentinel_pair, tile);

    if (cas_success) return true;

    bucket.load(cuda::memory_order_relaxed, tile);

    if (bucket.compute_load(pair, tile) != B) return false;

    main_load = bucket.compute_load(sentinel_pair, tile);


    #if P2_PRINT
    if (tile.thread_rank() == 0) printf("Looping in shortcut %d\n", main_load);
    #endif


  }


  //load alt bucket

  bucket_type alt_bucket(&d_table_[bucket1_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  alt_bucket.load(cuda::memory_order_relaxed, tile);


  int alt_load = alt_bucket.compute_load(sentinel_pair, tile);


  if (alt_bucket.compute_load(pair, tile) != B) return false;
  //if (alt_bucket.upsert(pair, key_equal{})) return true;


 
  do {

    #if P2_PRINT
    if (tile.thread_rank() == 0) printf("Looping in insert loop\n");
    #endif
    //bucket_type bucket(&d_table_[bucket0_id * bucket_size], tile);
    // bucket.load(cuda::memory_order_relaxed, tile);
    // int load = bucket.compute_load(sentinel_pair);
    // INCREMENT_PROBES_IN_TILE
    // bucket_type bucket1(&d_table_[bucket1_id * bucket_size], tile);
    // bucket1.load(cuda::memory_order_relaxed);
    // int load1 = bucket1.compute_load(sentinel_pair);
    // INCREMENT_PROBES_IN_TILE
    if (alt_load < main_load) {
      main_load = alt_load;
      bucket = alt_bucket;
    } else if (main_load == bucket_size && alt_load == bucket_size) {
        
       continue;
    }

    //bool cas_success = bucket.insert_tombstone(pair, key_equal{}, sentinel_pair, tombstone_pair, tile);

    //bucket is not full
    bool cas_success = false;
    if (lane_id == elected_lane) {
      cas_success = bucket.strong_cas_at_location(pair,
                                                  main_load,
                                                  sentinel_pair,
                                                  cuda::memory_order_relaxed,
                                                  cuda::memory_order_relaxed);
    }
    cas_success = tile.shfl(cas_success, elected_lane);


    if (cas_success) {
      return true;
    }

    //RELOAD
    bucket_type bucket_copy(&d_table_[bucket0_id * bucket_size]);
    INCREMENT_PROBES_IN_TILE
    bucket = bucket_copy;
    bucket.load(cuda::memory_order_relaxed, tile);
    main_load = bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);



    bucket_type alt_bucket_copy(&d_table_[bucket1_id * bucket_size]);
    INCREMENT_PROBES_IN_TILE
    alt_bucket = alt_bucket_copy;
    alt_bucket.load(cuda::memory_order_relaxed, tile);
    alt_load = alt_bucket.compute_load_tombstone(sentinel_pair, tombstone_pair, tile);


    if (bucket.compute_load(pair, tile) != B) return false;
    if (alt_bucket.compute_load(pair, tile) != B) return false;

  } while (main_load != bucket_size && alt_load != bucket_size);

  #if P2_PRINT
  if (tile.thread_rank() == 0) printf("swap-out code triggers here.\n");
  #endif
  //upsert functionality.
  return false;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ bool bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::replace_exact(
    tile_type const& tile,
    key_type const& insert_key,
    mapped_type const& insert_val,
    value_type const& pair_to_replace
    ) {

  using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;

  value_type pair{insert_key, insert_val};

  const int elected_lane = 0;
  auto lane_id = tile.thread_rank();
  auto bucket0_id = hf0_(pair.first) % num_buckets_;
  auto bucket1_id = hf1_(pair.first) % num_buckets_;


  if (key_equal{}(pair.first, sentinel_key_) || key_equal{}(pair.first, tombstone_key_)) {
    return false;
  }


  

  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  value_type tombstone_pair{tombstone_key_, sentinel_value_};

  //broadcast from 0 just in case
  //insertion_pair = tile.shfl(insertion_pair, 0);

  bucket_type bucket(&d_table_[bucket0_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  bucket.load(cuda::memory_order_relaxed, tile);

  if (bucket.replace_exact(pair, pair_to_replace, key_equal{}, tile)) return true;


  bucket_type alt_bucket(&d_table_[bucket1_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  alt_bucket.load(cuda::memory_order_relaxed, tile);


  if (alt_bucket.replace_exact(pair, pair_to_replace, key_equal{}, tile)) return true;

  return false;

}


template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ bool bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::upsert_exact(
    tile_type const& tile,
    key_type insert_key,
    mapped_type insert_val,
    key_type old_key,
    mapped_type old_val
    ) {

  using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;

  value_type pair{insert_key, insert_val};
  value_type old_pair{old_key, old_val};
  value_type sentinel_pair{sentinel_key_, sentinel_value_};

  const int elected_lane = 0;
  auto lane_id = tile.thread_rank();
  auto bucket0_id = hf0_(pair.first) % num_buckets_;
  auto bucket1_id = hf1_(pair.first) % num_buckets_;


  if (key_equal{}(pair.first, sentinel_key_) || key_equal{}(pair.first, tombstone_key_)) {
    return false;
  }


  bucket_type bucket(&d_table_[bucket0_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  bucket.load(cuda::memory_order_relaxed, tile);

  //make a stab, why not
  if (bucket.upsert_exact(pair, old_pair, key_equal{}, tile)) return true;
  //main load determines shortcutting
  //tombstone load determines insert positions/fill.
  int shortcut_load = bucket.compute_load(sentinel_pair, tile);


  if(1.0*shortcut_load/B < SHORTCUT_CUTOFF){

    //shortcutting
    return false;


  }


  //load alt bucket

  bucket_type alt_bucket(&d_table_[bucket1_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  alt_bucket.load(cuda::memory_order_relaxed, tile);

  if (alt_bucket.upsert_exact(pair, old_pair, key_equal{}, tile)) return true;

  return false;
  
}


//find a random key_val_pair in the same bucket space as me
//this is needed to identify if there is any bucket that can be identified.
template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::value_type bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::find_smaller_hash(
    tile_type const& tile,
    Key const& key) {

  using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;

  const int elected_lane = 0;
  auto lane_id = tile.thread_rank();
  auto bucket0_id = hf0_(key) % num_buckets_;
  auto bucket1_id = hf1_(key) % num_buckets_;


  bucket_type bucket(&d_table_[bucket0_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  bucket.load(cuda::memory_order_relaxed, tile);


  auto location = bucket.find_replace_location(key, bucket0_id, tile);


  //success! attempt swap.
  if (location != -1){

    return bucket.get_lane_pair(location, tile);
  }

  //test - only allow first bucket to be accessed
  bucket_type alt_bucket(&d_table_[bucket1_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  alt_bucket.load(cuda::memory_order_relaxed, tile);

  location = alt_bucket.find_replace_location(key, bucket1_id, tile);

  //success! attempt swap.
  if (location != -1){
    return alt_bucket.get_lane_pair(location, tile);
  }

  //in worst case always return last slot...
  #if P2_PRINT
  if (tile.thread_rank() == 0) printf("alt_bucket failed for %lu, returning last item\n", key);
  #endif
  
  return alt_bucket.get_lane_pair(B-1, tile);

}

//find a random key_val_pair in the same bucket space as me
//this is needed to identify if there is any bucket that can be identified.
template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::value_type bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::find_random(
    tile_type const& tile,
    Key const& key) {

  detail::mars_rng_32 rng;

  using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;

  const int elected_lane = 0;
  auto lane_id = tile.thread_rank();
  auto bucket0_id = hf0_(key) % num_buckets_;
  auto bucket1_id = hf1_(key) % num_buckets_;


  auto random = rng() % 2*bucket_size;

  if (random < bucket_size){

    bucket_type bucket(&d_table_[bucket0_id * bucket_size]);
    INCREMENT_PROBES_IN_TILE
    bucket.load(cuda::memory_order_relaxed, tile);

    return bucket.get_lane_pair(random, tile);


  } else {
    random = random-bucket_size;

    bucket_type bucket(&d_table_[bucket1_id * bucket_size]);
    INCREMENT_PROBES_IN_TILE
    bucket.load(cuda::memory_order_relaxed, tile);

    return bucket.get_lane_pair(random, tile);

  }

}



template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ bool bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::remove_exact(
    tile_type const& tile,
    value_type const& pair_to_remove) {

  using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;

  const int elected_lane = 0;
  auto lane_id = tile.thread_rank();
  auto bucket0_id = hf0_(pair_to_remove.first) % num_buckets_;
  auto bucket1_id = hf1_(pair_to_remove.first) % num_buckets_;


  if (key_equal{}(pair_to_remove.first, sentinel_key_) || key_equal{}(pair_to_remove.first, tombstone_key_)){
    return false;
  }

  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  value_type tombstone_pair{tombstone_key_, sentinel_value_};

  //broadcast from 0 just in case
  //insertion_pair = tile.shfl(insertion_pair, 0);

  bucket_type bucket(&d_table_[bucket0_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  bucket.load(cuda::memory_order_relaxed, tile);

  //make a stab, why not
  if (bucket.remove_exact(pair_to_remove, key_equal{}, tombstone_pair, tile)) return true;

  int main_load = bucket.compute_load(sentinel_pair, tile);

  if (1.0*main_load/B < SHORTCUT_CUTOFF){

    return false;

  }

  //load alt bucket

  bucket_type alt_bucket(&d_table_[bucket1_id * bucket_size]);
  INCREMENT_PROBES_IN_TILE
  alt_bucket.load(cuda::memory_order_relaxed, tile);


  if (alt_bucket.remove_exact(pair_to_remove, key_equal{}, tombstone_pair, tile)) return true;
 

  return false;
}



template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ bool bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::remove(
    tile_type const& tile,
    key_type const& key) {

  using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;

  const int elected_lane = 0;
  auto lane_id = tile.thread_rank();
  auto bucket0_id = hf0_(key) % num_buckets_;
  auto bucket1_id = hf1_(key) % num_buckets_;


  if (key_equal{}(key, sentinel_key_) || key_equal{}(key, tombstone_key_)){
    return false;
  }





  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  value_type tombstone_pair{tombstone_key_, sentinel_value_};


  //lock_buckets(tile, bucket0_id, bucket1_id);

  //new approach


    bucket_type bucket(&d_table_[bucket0_id * bucket_size]);
    INCREMENT_PROBES_IN_TILE
    bucket.load(cuda::memory_order_relaxed, tile);

    auto key_location = bucket.find_key_location(key, key_equal{}, tile);

    if (key_location != -1){

      auto val = bucket.get_value_from_lane(key_location, tile);

      value_type pair_to_remove{key, val};

      bool cas_success = false;
      if (lane_id == elected_lane) {
        cas_success = bucket.strong_cas_at_location(tombstone_pair,
                                                    key_location,
                                                    pair_to_remove,
                                                    cuda::memory_order_relaxed,
                                                    cuda::memory_order_relaxed);
      }

      cas_success = tile.shfl(cas_success, elected_lane);

      if (cas_success){
        
       // unlock_buckets(tile, bucket0_id, bucket1_id);
        return true;
      } 
    }


    bucket_type alt_bucket(&d_table_[bucket1_id * bucket_size]);
    INCREMENT_PROBES_IN_TILE
    alt_bucket.load(cuda::memory_order_relaxed, tile);

    auto alt_key_location = alt_bucket.find_key_location(key, key_equal{}, tile);

    if (alt_key_location != -1){

      auto val = alt_bucket.get_value_from_lane(alt_key_location, tile);


      value_type pair_to_remove{key, val};

      //printf("Val stored = %lu\n", val);

      bool cas_success = false;
      if (lane_id == elected_lane) {
        cas_success = alt_bucket.strong_cas_at_location(tombstone_pair,
                                                    alt_key_location,
                                                    pair_to_remove,
                                                    cuda::memory_order_relaxed,
                                                    cuda::memory_order_relaxed);
      }

      cas_success = tile.shfl(cas_success, elected_lane);

      if (cas_success){
        //unlock_buckets(tile, bucket0_id, bucket1_id);
        return true;
      }

      //printf("Looping in delete %d vs %d\n", key_location, alt_key_location);

    }

    if (key_location == -1 && alt_key_location == -1){
     
      //unlock_buckets(tile, bucket0_id, bucket1_id);
      return false;

    }


  //unlock_buckets(tile, bucket0_id, bucket1_id);
  return false;

}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::value_type bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::pack_together(
    tile_type const& tile,
    key_type insert_key,
    mapped_type insert_val
    ) {


      value_type pair{insert_key, insert_val};
      return pair;

    }


// template <class Key,
//           class T,
//           class Hash,
//           class KeyEqual,
//           cuda::thread_scope Scope,
//           class Allocator,
//           int B>
// template <typename tile_type>
// __device__ bool bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::insert_old(
//     value_type const& pair,
//     tile_type const& tile) {

//   using bucket_type = detail::bucket<atomic_pair_type, value_type, tile_type>;


//   auto bucket0_id = hf0_(pair.first) % num_buckets_;
//   auto bucket1_id = hf1_(pair.first) % num_buckets_;


//   if (key_equal{}(pair.first, sentinel_key_)) {
//     return false;
//   }


//   if (find(pair.first, tile) != sentinel_value_){

//     //upsert
//     //printf("Upsert triggering\n");
//     bucket_type bucket(&d_table_[bucket0_id * bucket_size], tile);
//     INCREMENT_PROBES_IN_TILE
//     bucket.load(cuda::memory_order_relaxed, tile);

//     if (bucket.upsert(pair, key_equal{})) return true;

//     bucket_type bucket1(&d_table_[bucket1_id * bucket_size], tile);
//     INCREMENT_PROBES_IN_TILE
//     bucket1.load(cuda::memory_order_relaxed);

//     if (bucket1.upsert(pair, key_equal{})) return true;

//   }



//   auto lane_id = tile.thread_rank();
//   const int elected_lane = 0;
//   value_type sentinel_pair{sentinel_key_, sentinel_value_};
//   value_type insertion_pair = pair;
//   do {
//     bucket_type bucket(&d_table_[bucket0_id * bucket_size], tile);
//     bucket.load(cuda::memory_order_relaxed, tile);
//     int load = bucket.compute_load(sentinel_pair);
//     INCREMENT_PROBES_IN_TILE
//     bucket_type bucket1(&d_table_[bucket1_id * bucket_size], tile);
//     bucket1.load(cuda::memory_order_relaxed);
//     int load1 = bucket1.compute_load(sentinel_pair);
//     INCREMENT_PROBES_IN_TILE
//     if (load1 < load) {
//       load = load1;
//       bucket = bucket1;
//     } else if (load1 == bucket_size && load == bucket_size) {
//       return false;
//     }
//     // bucket is not full
//     bool cas_success = false;
//     if (lane_id == elected_lane) {
//       cas_success = bucket.strong_cas_at_location(insertion_pair,
//                                                   load,
//                                                   sentinel_pair,
//                                                   cuda::memory_order_relaxed,
//                                                   cuda::memory_order_relaxed);
//     }
//     cas_success = tile.shfl(cas_success, elected_lane);
//     if (cas_success) {
//       return true;
//     }
//   } while (true);
//   return false;
// }


template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ void bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::lock_buckets(tile_type const& tile, uint64_t bucket0, uint64_t bucket1){


  if (bucket0 == bucket1){
    stall_lock(tile, bucket0);
  } else if (bucket0 < bucket1){

    stall_lock(tile, bucket0);
    //stall_lock(tile, bucket1);

  } else {
    //stall_lock(tile, bucket1);
    stall_lock(tile, bucket0);
  }

}


template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ void bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::unlock_buckets(tile_type const& tile, uint64_t bucket0, uint64_t bucket1){


  if (bucket0 == bucket1){
    unlock(tile, bucket0);
  } else {
    //unlock(tile, bucket1);
    unlock(tile, bucket0);
  }

}



template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ void bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::stall_lock(tile_type const& tile, uint64_t bucket){

  if (tile.thread_rank() == 0){

    uint64_t high = bucket/64;
    uint64_t low = bucket % 64;

    //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
    while (atomicOr((unsigned long long int *)&locks[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low)){
      //printf("TID %llu Stalling for Bucket %llu/%llu\n", threadIdx.x+blockIdx.x*blockDim.x, bucket, num_buckets_);
    }

  }

  tile.sync();
}


template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ void bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::unlock(tile_type const& tile, uint64_t bucket){

  if (tile.thread_rank() == 0){

    uint64_t high = bucket/64;
    uint64_t low = bucket % 64;

    //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
    atomicAnd((unsigned long long int *)&locks[high], (unsigned long long int) ~SET_BIT_MASK(low));

  }

  tile.sync();
}



template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::mapped_type
bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::find(key_type const& key,
                                                               tile_type const& tile) {
  const int num_hfs = 2;
  auto bucket_id = hf0_(key) % num_buckets_;
  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;
  for (int hf = 0; hf < num_hfs; hf++) {
    bucket_type cur_bucket(&d_table_[bucket_id * bucket_size]);
    cur_bucket.load(cuda::memory_order_relaxed, tile);
    INCREMENT_PROBES_IN_TILE
    int key_location = cur_bucket.find_key_location(key, key_equal{}, tile);
    if (key_location != -1) {
      auto found_value = cur_bucket.get_value_from_lane(key_location, tile);
      return found_value;
    } else {


      //shortcut exit.
      if (1.0*cur_bucket.compute_load(sentinel_pair, tile)/B < SHORTCUT_CUTOFF) return sentinel_value_;


      bucket_id = hf1_(key) % num_buckets_;
    }
  }

  return sentinel_value_;
}


// template <class Key,
//           class T,
//           class Hash,
//           class KeyEqual,
//           cuda::thread_scope Scope,
//           class Allocator,
//           int B>
// __device__ bool
// bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::find_by_reference(tile_type const& tile, key_type const& key,
//                                                                mapped_type &value) {
//   const int num_hfs = 2;
//   auto bucket_id = hf0_(key) % num_buckets_;
//   value_type sentinel_pair{sentinel_key_, sentinel_value_};
//   using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;
//   for (int hf = 0; hf < num_hfs; hf++) {
//     bucket_type cur_bucket(&d_table_[bucket_id * bucket_size]);
//     cur_bucket.load(cuda::memory_order_relaxed, tile);
//     INCREMENT_PROBES_IN_TILE
//     int key_location = cur_bucket.find_key_location(key, key_equal{}, tile);
//     if (key_location != -1) {
//       auto found_value = cur_bucket.get_value_from_lane(key_location, tile);

//       value = found_value;
//       return true;
//     } else {


//       //shortcut exit.
//       //if (1.0*cur_bucket.compute_load(sentinel_pair, tile)/B < SHORTCUT_CUTOFF) return sentinel_value_;


//       bucket_id = hf1_(key) % num_buckets_;
//     }
//   }

//   return false;
// }


//no loop - exact code variant
//saves a check to compute load.
template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
__device__ bool
bght::p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::find_by_reference(tile_type const& tile, key_type const& key,
                                                               mapped_type &value) {
  auto bucket_id = hf0_(key) % num_buckets_;
  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  using bucket_type = detail::bucket<atomic_pair_type, value_type, B>;
 
  bucket_type cur_bucket(&d_table_[bucket_id * bucket_size]);
  cur_bucket.load(cuda::memory_order_relaxed, tile);
  INCREMENT_PROBES_IN_TILE
  int key_location = cur_bucket.find_key_location(key, key_equal{}, tile);
  if (key_location != -1) {
    auto found_value = cur_bucket.get_value_from_lane(key_location, tile);

    value = found_value;
    return true;
  }


  //shortcut exit.
  if (1.0*cur_bucket.compute_load(sentinel_pair, tile)/B < SHORTCUT_CUTOFF) return sentinel_value_;


  bucket_id = hf1_(key) % num_buckets_;

 
  bucket_type alt_bucket(&d_table_[bucket_id * bucket_size]);
  alt_bucket.load(cuda::memory_order_relaxed, tile);
  INCREMENT_PROBES_IN_TILE
  

  key_location = alt_bucket.find_key_location(key, key_equal{}, tile);
  if (key_location != -1) {
    auto found_value = alt_bucket.get_value_from_lane(key_location, tile);

    value = found_value;
    return true;
  }

  return false;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
template <typename RNG>
void p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::randomize_hash_functions(
    RNG& rng) {
  hf0_ = initialize_hf<hasher>(rng);
  hf1_ = initialize_hf<hasher>(rng);
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
typename p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::size_type
p2bht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::size(cudaStream_t stream) {
  const auto sentinel_key{sentinel_key_};

  auto d_count = std::allocator_traits<size_type_allocator_type>::allocate(
      size_type_allocator_, static_cast<size_type>(1));
  cuda_try(cudaMemsetAsync(d_count, 0, sizeof(std::size_t), stream));
  const uint32_t block_size = 128;
  const uint32_t num_blocks = (capacity_ + block_size - 1) / block_size;

  detail::kernels::count_kernel<block_size>
      <<<num_blocks, block_size, 0, stream>>>(sentinel_key, d_count, *this);
  std::size_t num_invalid_keys;
  cuda_try(cudaMemcpyAsync(
      &num_invalid_keys, d_count, sizeof(std::size_t), cudaMemcpyDeviceToHost));

  cudaFree(d_count);
  return capacity_ - num_invalid_keys;
}

}  // namespace bght
