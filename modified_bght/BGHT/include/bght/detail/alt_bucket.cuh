/*
 *   Copyright 2021-2024 The Regents of the University of California, Davis
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
#include <cuda/atomic>
#include <cuda/std/atomic>
#include <bght/detail/MurmurHash3_32.hpp>

#ifndef BUCKET_PRINT
#define BUCKET_PRINT 0
#endif

namespace bght {
namespace detail {
template <typename atomic_pair_type, typename pair_type, int B>
struct bucket {
  bucket() = delete;
  DEVICE_QUALIFIER
  bucket(atomic_pair_type* ptr) : ptr_(ptr) {}

  DEVICE_QUALIFIER
  bucket(const bucket& other) : lane_pair_(other.lane_pair_), ptr_(other.ptr_) {}

  DEVICE_QUALIFIER
  void load(cuda::memory_order order, cg::thread_block_tile<B> tile_) {
    lane_pair_ = ptr_[tile_.thread_rank()].load(order);
  }


  //modification - checks 
  DEVICE_QUALIFIER
  int compute_load(const pair_type& sentinel, cg::thread_block_tile<B> tile_) {
    auto load_bitmap = tile_.ballot(lane_pair_.first != sentinel.first);
    int load = __popc(load_bitmap);
    return load;
  }


  DEVICE_QUALIFIER
  int compute_load_tombstone(const pair_type& sentinel, const pair_type& tombstone, cg::thread_block_tile<B> tile_) {
    auto load_bitmap = tile_.ballot(lane_pair_.first != sentinel.first && lane_pair_.first != tombstone.first);
    int load = __popc(load_bitmap);
    return load;
  }

  // returns -1 if not found
  template <class KeyEqual>
  DEVICE_QUALIFIER int find_key_location(const typename pair_type::first_type& key,
                                         const KeyEqual key_equal, cg::thread_block_tile<B> tile_) {
    bool key_exist = key_equal(key, lane_pair_.first);
    auto key_exist_bmap = tile_.ballot(key_exist);
    int key_lane = __ffs(key_exist_bmap);
    return key_lane - 1;
  }

  //generate comparative hashes
  //return the first index that features a comparative hash less than the current key
  //all keys must use the same hash

  DEVICE_QUALIFIER int find_replace_location(const typename pair_type::first_type& key,
                                         uint32_t bucket_id, cg::thread_block_tile<B> tile_) {

    MurmurHash3_32<typename pair_type::first_type> hasher(bucket_id);

    auto my_hash = hasher(key);
    auto bucket_hash = hasher(lane_pair_.first);
    //my_hash_function  = 
    bool key_exist = (bucket_hash < my_hash) && (tile_.thread_rank() != B-1);


    auto key_exist_bmap = tile_.ballot(key_exist);
    int key_lane = __ffs(key_exist_bmap);


    if (tile_.thread_rank() == key_lane-1){
      #if BUCKET_PRINT
      printf("Key %lu has smaller hash than %lu\n", lane_pair_.first, key);
      #endif
    }

    return key_lane - 1;
  }


  DEVICE_QUALIFIER
  typename pair_type::second_type get_value_from_lane(int location, cg::thread_block_tile<B> tile_) {
    return tile_.shfl(lane_pair_.second, location);
  }

  DEVICE_QUALIFIER
  pair_type get_lane_pair(int location, cg::thread_block_tile<B> tile_){
    return tile_.shfl(lane_pair_, location);
  }

  DEVICE_QUALIFIER
  bool weak_cas_at_location(const pair_type& pair,
                            const int location,
                            const pair_type& sentinel,
                            cuda::memory_order success = cuda::memory_order_seq_cst,
                            cuda::memory_order failure = cuda::memory_order_seq_cst) {
    pair_type expected = sentinel;
    pair_type desired = pair;
    bool cas_success =
        ptr_[location].compare_exchange_weak(expected, desired, success, failure);
    return cas_success;
  }

  DEVICE_QUALIFIER
  bool strong_cas_at_location(const pair_type& pair,
                              const int location,
                              const pair_type& sentinel,
                              cuda::memory_order success = cuda::memory_order_seq_cst,
                              cuda::memory_order failure = cuda::memory_order_seq_cst) {
    pair_type expected = sentinel;
    pair_type desired = pair;
    bool cas_success =
        ptr_[location].compare_exchange_strong(expected, desired, success, failure);
    return cas_success;
  }

  DEVICE_QUALIFIER
  pair_type strong_cas_at_location_ret_old(
      const pair_type& pair,
      const int location,
      const pair_type& sentinel,
      cuda::memory_order success = cuda::memory_order_seq_cst,
      cuda::memory_order failure = cuda::memory_order_seq_cst) {
    pair_type expected = sentinel;
    pair_type desired = pair;
    ptr_[location].compare_exchange_strong(expected, desired, success, failure);
    return expected;
  }

  DEVICE_QUALIFIER
  pair_type exch_at_location(const pair_type& pair,
                             const int location,
                             cuda::memory_order order = cuda::memory_order_seq_cst) {
    auto old = ptr_[location].exchange(pair, order);
    return old;
  }


  DEVICE_QUALIFIER bool upsert_at_location(const pair_type& pair, int location, cg::thread_block_tile<B> tile){



    auto val = get_value_from_lane(location, tile);

    pair_type replace_pair{pair.first, val};

    bool cas_success = false;
    if (tile.thread_rank() == 0) {
      cas_success = strong_cas_at_location(pair,
                                                  location,
                                                  replace_pair,
                                                  cuda::memory_order_relaxed,
                                                  cuda::memory_order_relaxed);
    }

    cas_success = tile.shfl(cas_success, 0);

    return cas_success;


  }


  template <class KeyEqual>
  DEVICE_QUALIFIER bool replace_tombstone_at_location(const pair_type& pair, const pair_type& tombstone_pair, const KeyEqual key_equal, cg::thread_block_tile<B> tile){


    auto location = find_key_location(tombstone_pair.first, key_equal, tile);

    bool cas_success = false;
    if (tile.thread_rank() == 0) {
      cas_success = strong_cas_at_location(pair,
                                                  location,
                                                  tombstone_pair,
                                                  cuda::memory_order_relaxed,
                                                  cuda::memory_order_relaxed);
    }

    cas_success = tile.shfl(cas_success, 0);

    return cas_success;


  }


  template <class KeyEqual>
  DEVICE_QUALIFIER bool upsert(const pair_type& pair, const KeyEqual key_equal, cg::thread_block_tile<B> tile_){


    int key_location = find_key_location(pair.first, key_equal, tile_);
    while (key_location != -1){

      pair_type new_pair(pair.first, get_value_from_lane(key_location, tile_));

      bool success = false;
      if (tile_.thread_rank() == key_location){

        //printf("Tile thread %lu updating key %lu in location %lu\n", tile_.thread_rank(), pair.first, key_location);
        success = strong_cas_at_location(new_pair, key_location, pair);
      }

      

      success = tile_.ballot(success);

      if (success){

        //printf("Insert succeeded\n");
        return true;
      }

      key_location = find_key_location(pair.first, key_equal, tile_);

    }


    return false;

  }

  template <class KeyEqual>
  DEVICE_QUALIFIER bool upsert_exact(const pair_type& pair, const pair_type&old_pair, const KeyEqual key_equal, cg::thread_block_tile<B> tile_){


    int key_location = find_key_location(pair.first, key_equal, tile_);
    if (key_location != -1){

      //pair_type new_pair(pair.first, get_value_from_lane(key_location, tile_));

      bool success = false;
      if (tile_.thread_rank() == key_location){

        //printf("Tile thread %lu updating key %lu in location %lu\n", tile_.thread_rank(), pair.first, key_location);
        success = strong_cas_at_location(pair, key_location, old_pair);
      }

      

      success = tile_.ballot(success);

      if (success){

        //printf("Insert succeeded\n");
        return true;
      }

    }


    return false;

  }


  template <class KeyEqual>
  DEVICE_QUALIFIER bool replace_exact(const pair_type& pair, const pair_type&old_pair, const KeyEqual key_equal, cg::thread_block_tile<B> tile_){


    int key_location = find_key_location(old_pair.first, key_equal, tile_);
    if (key_location != -1){

      //pair_type new_pair(pair.first, get_value_from_lane(key_location, tile_));

      bool success = false;
      if (tile_.thread_rank() == key_location){

        //printf("Tile thread %lu updating key %lu in location %lu\n", tile_.thread_rank(), pair.first, key_location);
        success = strong_cas_at_location(pair, key_location, old_pair);
      }

      

      success = tile_.ballot(success);

      if (success){

        //printf("Insert succeeded\n");
        return true;
      }

    }


    return false;

  }



  //insert a key, taking care to insert into tombstone spots first.
  template <class KeyEqual>
  DEVICE_QUALIFIER bool insert_tombstone_old(const pair_type& insert_pair, const KeyEqual key_equal, const pair_type& sentinel, const pair_type& tombstone_pair, cg::thread_block_tile<B> tile_){

    bool sentinel_key_exist = key_equal(sentinel.first, lane_pair_.first);
    bool tombstone_key_exist = key_equal(tombstone_pair.first, lane_pair_.first);

    bool empty = sentinel_key_exist || tombstone_key_exist;



    auto key_exist_bmap = tile_.ballot(empty);
    int key_lane = __ffs(key_exist_bmap) -1;


    while(key_lane != -1){


      bool success = false;


      if (tile_.thread_rank() == key_lane){

        if (sentinel_key_exist){
          success = strong_cas_at_location(insert_pair, key_lane, sentinel);
        } else {
          //must be tombstone in position.
          success = strong_cas_at_location(insert_pair, key_lane, tombstone_pair);

        }


      }


      success = tile_.ballot(success);

      if (success) return true;

      //else fail
      key_exist_bmap ^= 1UL << key_lane;
      key_lane = __ffs(key_exist_bmap)-1;

      //printf("Looping in tombstone %lx - lane %d\n", key_exist_bmap, key_lane);

    }

    return false;


  }

  //variant of insert tombstone 
  //run logic in two phases.
  template <class KeyEqual>
  DEVICE_QUALIFIER bool insert_tombstone(const pair_type& insert_pair, const KeyEqual key_equal, const pair_type& sentinel, const pair_type& tombstone_pair, cg::thread_block_tile<B> tile_){

    bool sentinel_key_exist = key_equal(sentinel.first, lane_pair_.first);
    bool tombstone_key_exist = key_equal(tombstone_pair.first, lane_pair_.first);

    bool empty = sentinel_key_exist || tombstone_key_exist;



    auto key_exist_bmap = tile_.ballot(sentinel_key_exist);
    int key_lane = __ffs(key_exist_bmap) -1;


    while(key_lane != -1){


      bool success = false;


      if (tile_.thread_rank() == 0){
    
        success = strong_cas_at_location(insert_pair, key_lane, sentinel);

      }


      success = tile_.ballot(success);

      if (success) return true;

      //else fail
      key_exist_bmap ^= 1UL << key_lane;
      key_lane = __ffs(key_exist_bmap)-1;

      //printf("Looping in tombstone %lx - lane %d\n", key_exist_bmap, key_lane);

    }

    key_exist_bmap = tile_.ballot(tombstone_key_exist);
    key_lane = __ffs(key_exist_bmap) -1;

    while(key_lane != -1){


      bool success = false;


      if (tile_.thread_rank() == 0){
    
        success = strong_cas_at_location(insert_pair, key_lane, tombstone_pair);

      }


      success = tile_.ballot(success);

      if (success) return true;

      //else fail
      key_exist_bmap ^= 1UL << key_lane;
      key_lane = __ffs(key_exist_bmap)-1;

      //printf("Looping in tombstone lower %lx - lane %d\n", key_exist_bmap, key_lane);

    }



    return false;


  }

  template <class KeyEqual>
  DEVICE_QUALIFIER bool insert_sentinel(const pair_type& insert_pair, const KeyEqual key_equal, const pair_type& sentinel, cg::thread_block_tile<B> tile_){

    bool sentinel_key_exist = key_equal(sentinel.first, lane_pair_.first);
    //bool tombstone_key_exist = key_equal(tombstone_pair.first, lane_pair_.first);

    bool empty = sentinel_key_exist;



    auto key_exist_bmap = tile_.ballot(empty);
    int key_lane = __ffs(key_exist_bmap) -1;


    while(key_lane != -1){


      bool success = false;


      if (tile_.thread_rank() == key_lane){

        if (sentinel_key_exist){
          success = strong_cas_at_location(insert_pair, key_lane, sentinel);
        } 

      }


      success = tile_.ballot(success);

      if (success) return true;

      //else fail
      key_exist_bmap &= (~(1U << key_lane));
      key_lane = __ffs(key_exist_bmap)-1;

    }

    return false;


  }

  template <class KeyEqual>
  DEVICE_QUALIFIER bool remove_exact(const pair_type& key_pair, const KeyEqual key_equal, const pair_type& tombstone_pair, cg::thread_block_tile<B> tile_){


    int key_location = find_key_location(key_pair.first, key_equal, tile_);

    //pair_type new_pair(key, get_value_from_lane(key_location));

    if (key_location == -1) return false;

    bool success = false;
    if (tile_.thread_rank() == key_location){

      //printf("Tile thread %lu updating key %lu in location %lu\n", tile_.thread_rank(), pair.first, key_location);
      success = strong_cas_at_location(tombstone_pair, key_location, key_pair);
    }

    

    success = tile_.ballot(success);


    return success;




  }




  template <class KeyEqual>
  DEVICE_QUALIFIER bool remove(const typename pair_type::first_type& key, const KeyEqual key_equal, const pair_type& tombstone_pair, cg::thread_block_tile<B> tile_){

    bool match = (lane_pair_.first == key);

    auto key_exist_bmap = tile_.ballot(match);
    int key_lane = __ffs(key_exist_bmap) -1;

    while(key_lane != -1){


      bool success = false;

      pair_type new_pair(key, get_value_from_lane(key_lane, tile_));

      if (tile_.thread_rank() == key_lane){

        success = strong_cas_at_location(tombstone_pair, key_lane, new_pair);

      }


      success = tile_.ballot(success);

      if (success) return true;

      //else fail

      key_exist_bmap ^= 1UL << key_lane;
      key_lane = __ffs(key_exist_bmap)-1;

      //printf("Stalling in remove: %x\n", key_exist_bmap);

    }

    return false;

  }

  DEVICE_QUALIFIER
  atomic_pair_type* begin() { return ptr_; }


  DEVICE_QUALIFIER
  void copy_other_bucket(const bucket& other){
      lane_pair_ = other.lane_pair_;
      ptr_ = other.ptr_;
  }

  pair_type lane_pair_;
  atomic_pair_type* ptr_;
};
}  // namespace detail
}  // namespace bght
