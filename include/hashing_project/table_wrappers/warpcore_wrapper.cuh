#ifndef WARPCORE_WRAPPER
#define WARPCORE_WRAPPER

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/alloc_utils.cuh>

//#include <warpcore/svht_upsert.cuh>
#include <warpcore/svht_2.cuh>

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

   // warpcore_type host_version;

   // warpcore_type * device_version;

   template <typename Key, typename Val, uint tile_size, uint bucket_size>
   struct warpcore_wrapper {


      using tile_type = cg::thread_block_tile<bucket_size>;

      using internal_table_type = warpcore::svht2_generic<Key, Val, bucket_size>;

      using my_type = warpcore_wrapper<Key, Val, tile_size, bucket_size>;

      internal_table_type * internal_table;

      internal_table_type * host_ref;

      //dummy handle
      static __host__ my_type * generate_on_device(uint64_t cache_capacity, uint64_t seed){

         my_type * host_version = gallatin::utils::get_host_version<my_type>();


         internal_table_type * host_table = new internal_table_type(cache_capacity, 42);

         internal_table_type * dev_table = gallatin::utils::get_device_version<internal_table_type>();

         cudaMemcpy(dev_table, host_table, sizeof(internal_table_type), cudaMemcpyHostToDevice);

         host_version->host_ref = host_table;
         //delete old version so resources cannot be cleared.
         host_table = nullptr;

         host_version->internal_table = dev_table;
         //host_version->internal_table = internal_table_type::generate_on_device(cache_capacity, sentinel_key, tombstone_key, sentinel_val);
         
         return gallatin::utils::move_to_device<my_type>(host_version);

      }

      static __host__ void free_on_device(my_type * device_version){


         my_type * host_version = gallatin::utils::move_to_host<my_type>(device_version);

         internal_table_type * host_internal = gallatin::utils::move_to_host(host_version->internal_table);

         cudaFree(host_internal->locks);

         //generate version to fall out of scope.
         internal_table_type host_to_free = *host_internal;

         cudaFreeHost(host_internal);
         cudaFreeHost(host_version); 
         
         return;

      }


      __device__ bool upsert_replace(tile_type my_tile, Key key, Val val){

         return internal_table->upsert(key, val, my_tile);

         //return internal_table->upsert_replace(pack_together(key, val), my_tile);
      }

      // //nope! no storage
      __device__ bool find_with_reference(tile_type my_tile, Key key, Val & val){
         
         return internal_table->query(key, val, my_tile);
         //return internal_table->find_by_reference(my_tile, key, val);
      }

      __device__ bool remove(tile_type my_tile, Key key){
         return internal_table->erase_bool(key, my_tile);
         //return internal_table->remove(my_tile, key);
      }

      // __device__ bool upsert(tile_type my_tile, Key old_key, Val old_val, Key new_key, Val new_val){
      //    return internal_table->upsert_exact(my_tile, old_key, old_val, new_key, new_val);
      // }

      // __device__ bool upsert(tile_type my_tile, packed_pair_type old_pair, packed_pair_type new_pair){
      //    return upsert(my_tile, old_pair.first, old_pair.second, new_pair.first, new_pair.second);
      // }

      // __device__ bool insert_if_not_exists(tile_type my_tile, Key key, Val val){
      //    return internal_table->insert_exact(my_tile, key, val);
      // }
      
      // __device__ packed_pair_type find_replaceable_pair(tile_type my_tile, Key key){
      //    return internal_table->find_smaller_hash(my_tile, key);
      // }

      // static __device__ packed_pair_type pack_together(Key key, Val val){
      //    return packed_pair_type{key, val};
      // }


      __host__ float load(){

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         auto load_factor = host_version->host_ref->load_factor();

         cudaFreeHost(host_version);

         return load_factor;

      }
      static char * get_name(){
         return "warpcore_hashing";
      }


      __host__ void print_space_usage(){


         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);
         internal_table_type * host_internal = gallatin::utils::copy_to_host<internal_table_type>(host_version->internal_table);


         uint64_t capacity = host_internal->capacity();



         printf("Warpcore using %lu bytes\n", capacity*(sizeof(Key)+sizeof(Val))+(capacity-1)/8+1);

         cudaFreeHost(host_internal);
         cudaFreeHost(host_version);


      }


   };



} //namespace wrappers

}  // namespace ht_project

#endif  // GPU_BLOCK_