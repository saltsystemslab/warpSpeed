#ifndef P2_BGHT_WRAPPER
#define P2_BGHT_WRAPPER

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/alloc_utils.cuh>

#include <bght/p2bht.hpp>

#include <cuda/atomic>
#include <cuda/std/atomic>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

// helper_macro
// define macros
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits) ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SET_BIT_MASK(index) ((1ULL << index))

// a pointer list managing a set section of device memory



//cache protocol
//query cache
//on success add to pin?
//need delete from potential buckets implementation - need to download warpcore...
//buidld with primary p2bht first.



namespace hashing_project {

namespace wrappers {

   // template <typename Key, typename Val>
   // struct bght_packed_pair {
   //    Key first;
   //    Val second;
   // };


   template <typename Key, typename Val, uint bucket_size>
   struct p2_wrapper {


      using tile_type = cg::thread_block_tile<bucket_size>;

      using internal_table_type = bght::p2bht_generic<Key, Val, bucket_size>;

      using my_type = p2_wrapper<Key, Val, bucket_size>;

      using packed_pair_type = typename internal_table_type::value_type;

      internal_table_type * internal_table;

      //dummy handle
      static __host__ my_type * generate_on_device(uint64_t cache_capacity, Key sentinel_key, Key tombstone_key, Val sentinel_val){

         my_type * host_version = gallatin::utils::get_host_version<my_type>();

         host_version->internal_table = internal_table_type::generate_on_device(cache_capacity, sentinel_key, tombstone_key, sentinel_val);
         
         return gallatin::utils::move_to_device<my_type>(host_version);

      }

      static __host__ void free_on_device(my_type * device_version){
         
         return;

      }

      //nope! no storage
      __device__ bool find_with_reference(tile_type my_tile, Key key, Val & val){
         return internal_table->find_by_reference(my_tile, key, val);
      }

      __device__ bool upsert(tile_type my_tile, Key old_key, Val old_val, Key new_key, Val new_val){
         return internal_table->upsert_exact(my_tile, old_key, old_val, new_key, new_val);
      }

      __device__ bool upsert(tile_type my_tile, packed_pair_type old_pair, packed_pair_type new_pair){
         return upsert(my_tile, old_pair.first, old_pair.second, new_pair.first, new_pair.second);
      }

      __device__ bool insert_if_not_exists(tile_type my_tile, Key key, Val val){
         return internal_table->insert_exact(my_tile, key, val);
      }
      
      __device__ packed_pair_type find_replaceable_pair(tile_type my_tile, Key key){
         return internal_table->find_smaller_hash(my_tile, key);
      }

      static __device__ packed_pair_type pack_together(Key key, Val val){
         return packed_pair_type{key, val};
      }


      __device__ void upsert_generic(tile_type my_tile, Key key, Val val){
         return internal_table->upsert_generic(my_tile, key, val);
      }

      static char * get_name(){
         return "p2_hashing";
      }


   };



} //namespace wrappers

}  // namespace ht_project

#endif  // GPU_BLOCK_