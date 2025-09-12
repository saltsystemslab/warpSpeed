#ifndef WARPSPEED_WARPCORE_WRAPPER
#define WARPSPEED_WARPCORE_WRAPPER

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <warpcore/single_value_hash_table.cuh>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;



//helper that wraps warpcore for use in other tests
//exposes the same base API as us.

using namespace warpcore;

namespace warpSpeed {

namespace tables {

   template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
   struct warpcore_wrapper {

      //must be the same for warpcore
      static_assert(partition_size==bucket_size);

      using my_type = warpcore_wrapper<Key, defaultKey, tombstoneKey, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;

      using tile_type = cg::thread_block_tile<partition_size>;

      using md_bucket_type = hash_table_metadata<Key, defaultKey, tombstoneKey, Val, tombstoneVal, partition_size, bucket_size>;

      using bucket_type = hash_table_metadata_bucket<Key, defaultKey, tombstoneKey, tombstoneKey-1, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;

      using packed_pair_type = ht_pair<Key, Val>;

      using probing_scheme_t = defaults::probing_scheme_t<Key, partition_size>;

      using warpcore_type =
           SingleValueHashTable<
               Key, Val,
               defaultKey,
               tombstoneKey,
               probing_scheme_t,
               defaults::table_storage_t<Key, Val>,
               defaults::temp_memory_bytes()>;


      warpcore_type * table;
      warpcore_type * host_table;

      //dummy handle
      static __host__ my_type * generate_on_device(uint64_t cache_capacity, uint64_t ext_seed){


         warpcore_type * host_table = new warpcore_type(cache_capacity);

         warpcore_type * table = gallatin::utils::get_device_version<warpcore_type>();

         cudaMemcpy(table, host_table, sizeof(warpcore_type), cudaMemcpyHostToDevice);

         my_type * host_version = gallatin::utils::get_host_version<my_type>();

         host_version->table = table;

         host_version->host_table = host_table;

         my_type * device_version = gallatin::utils::move_to_device<my_type>(host_version);

         cudaDeviceSynchronize();

         return device_version;

      }

      static __host__ void free_on_device(my_type * device_version){

         my_type * host_version = gallatin::utils::move_to_host<my_type>(device_version);

         cudaFree(host_version->table);
         delete host_version->host_table;

         cudaFreeHost(host_version);
         
         return;

      }

      __device__ bool upsert_replace(const tile_type & my_tile, const Key & key, const Val & val){

         auto insert_status = table->insert(key, val, my_tile);
      
         if (insert_status != Status::none()){
            return false;
         }

         return true;

       }
   
      [[nodiscard]] __device__ bool find_with_reference(tile_type my_tile, Key key, Val & val){

         auto query_status = table->retrieve(key, val, my_tile);

         if (query_status != Status::none()){

            return false;

         }

         return true;

      }



      __device__ bool remove(tile_type my_tile, Key key){

         auto delete_status = table->erase(key, my_tile);


         if (delete_status != Status::none()){

            return false;
         }

         return true;

      }

      static std::string get_name(){
         return "warpcore";
      }


   };

template <typename T>
constexpr T generate_warpcore_md_tombstone(uint64_t offset) {
  return (~((T) 0)) - offset;
};

template <typename T>
constexpr T generate_warpcore_md_sentinel() {
  return ((T) 0);
};


// template <typename Key, Key sentinel, Key tombstone, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
  
template <typename Key, typename Val, uint tile_size, uint bucket_size>
using warpcore_wrapper_generic = typename warpSpeed::tables::warpcore_wrapper<Key,
                                    generate_warpcore_md_sentinel<Key>(),
                                    generate_warpcore_md_tombstone<Key>(0),
                                    Val,
                                    generate_warpcore_md_sentinel<Val>(),
                                    generate_warpcore_md_tombstone<Val>(0),
                                    tile_size,
                                    bucket_size>;




} //namespace wrappers

}  // namespace ht_project

#endif  // GPU_BLOCK_