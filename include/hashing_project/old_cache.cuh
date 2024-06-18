#ifndef CACHE
#define CACHE

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

#define CACHE_PRINT 0

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



template <typename cache, uint tile_size>
__global__ void write_back_host_kernel(cache * dev_cache, uint64_t n_items){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_items) return;


   uint64_t current_val;
   if (dev_cache->map->find_with_reference(my_tile, tid, current_val)){


      dev_cache->write_back_host(my_tile, tid, current_val);

      // auto packed = dev_cache->map->pack_together(tid, current_val);

      // if (!dev_cache->map->remove_exact(my_tile, packed)){
      //    #if CACHE_PRINT
      //     if (my_tile.thread_rank() == 0) printf("Failed to delete key %lu\n", tid);
      //    #endif
      // }

      // //and check for double insert.
      // if (dev_cache->map->find_with_reference(my_tile, tid, current_val)){

      //    #if CACHE_PRINT
      //    if (my_tile.thread_rank() == 0) printf("Multiple copies of %lu stored\n", tid);
      //    #endif
      // }

   }
}

template <typename cache>
__global__ void compare_cache_arrays(cache * dev_cache, uint64_t * dev_array, uint64_t n_host_items){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= n_host_items) return;


   if (dev_cache->host_items[tid] != dev_array[tid]){
      printf("Mismatch at %lu: cache %lu != host %lu \n",tid, dev_cache->host_items[tid], dev_array[tid] );
   }


}


namespace hashing_project {

   template <template<typename, typename, uint> typename Hashtable, uint tile_size>
   struct cache {


      using ht_type = Hashtable<uint64_t, uint64_t, tile_size>;

      using my_type = cache<Hashtable, tile_size>;

      uint64_t * host_items;

      ht_type * map;


      uint64_t host_capacity;
      uint64_t cache_capacity;
      





      __host__ static my_type * generate_on_device(uint64_t ext_host_capacity, uint64_t ext_cache_capacity){


         my_type * host_version = gallatin::utils::get_host_version<my_type>();

         // size, sentinel, tombstone, sentinel_value
         //ht_type host_map(ext_cache_capacity, ext_host_capacity, ext_host_capacity+1, 0ULL);

         //ht_type * dev_table = gallatin::utils::get_device_version<ht_type>();

         //cudaMemcpy(dev_table, &host_map, sizeof(ht_type), cudaMemcpyHostToDevice);'

         printf("Starting test with hash table %s\n", ht_type::get_name());

         host_version->map = ht_type::generate_on_device(ext_cache_capacity, ext_host_capacity, ext_host_capacity+1, 0ULL);

         host_version->host_items = gallatin::utils::get_host_version<uint64_t>(ext_host_capacity);

         cudaMemset(host_version->host_items, 0, sizeof(uint64_t)*ext_host_capacity);

         host_version->host_capacity = ext_host_capacity;

         host_version->cache_capacity = ext_cache_capacity;

         my_type * device_version = gallatin::utils::move_to_device<my_type>(host_version);

         return device_version;



      }

      __device__ void increment_index(cg::thread_block_tile<tile_size> my_tile, uint64_t index){


         //first, query ht and see if key is loaded
         uint64_t current_val;

         while (true){


            #if CACHE_PRINT
            if (my_tile.thread_rank() == 0) printf("Looping on %lu\n", index);
            #endif

            if (map->find_with_reference(my_tile, index, current_val)){

               #if CACHE_PRINT
               if (my_tile.thread_rank() == 0) printf("Reference found %lu, current val %lu\n", index, current_val);
               #endif

               uint64_t next_val = current_val+1; 

               //exact replacement
               if (map->upsert(my_tile, index, next_val, index, current_val)){

                  #if CACHE_PRINT
                  if (my_tile.thread_rank() == 0) printf("%lu updated value %lu->%lu\n", index, current_val, next_val);
                  #endif

                  return;
               }

               //Two cases - 1 not found - key was deleted before upsert
               // 2 - key was modified
               // in both cases loop is sufficient.
               __threadfence();
               continue;

            } else {

               #if CACHE_PRINT
               if (my_tile.thread_rank() == 0) printf("%lu not found\n", index);
               #endif

               //load new key.
               //go ahead and increment?
               current_val = load_index_host(my_tile, index);

               if (current_val == ~0ULL){

                  //loop
                  __threadfence();
                  continue;
               } else {
                  current_val = current_val+1;
               }

               if (map->insert_if_not_exists(my_tile, index, current_val)){

                  // if (map->insert_exact(my_tile, index, current_val)){
                  //    printf("Double insert in same group\n");
                  // }

                  #if CACHE_PRINT
                  if (my_tile.thread_rank() == 0) printf("%lu moved to host with value %lu->%lu\n", index, current_val-1, current_val);
                  #endif

                  return;

               } else {

                  //if (map->find_by_reference(my_tile, index, current_val)) continue;

                  #if CACHE_PRINT
                  if (my_tile.thread_rank() == 0) printf("%lu needs replacement.\n", index);
                  #endif

                  //locate one slot and try replace exact.

                  auto replacement_pair = map->find_replaceable_pair(my_tile, index);

                  #if CACHE_PRINT
                  if (my_tile.thread_rank() == 0) printf("%lu replacing %lu\n", index, replacement_pair.first);
                  #endif
                  // //don't write back sentinels

                  auto packed_insert = map->pack_together(index, current_val);

                  if (map->upsert(my_tile, packed_insert, replacement_pair)){

                     #if CACHE_PRINT
                     if (my_tile.thread_rank() == 0) printf("%lu replaced %lu\n", index, replacement_pair.first);
                     #endif

                     if (replacement_pair.first < host_capacity){
                        write_back_host(my_tile, replacement_pair.first, replacement_pair.second);
                     } else {

                        #if CACHE_PRINT
                         if (my_tile.thread_rank() == 0) printf("Fake key %lu pulled from table\n", replacement_pair.first);
                        #endif
                     }

                     __threadfence();

                     return;
                  }


                  write_back_host(my_tile, index, current_val);

                  #if CACHE_PRINT
                  if (my_tile.thread_rank() == 0) printf("%lu failed to replace %lu\n", index, replacement_pair.first);
                  #endif
                  //at this point the key is not found - therefore both buckets are full.
                  //let's try some replacement boys?

                  // if (map->upsert_)

                  // auto found_pair = map->find_random(my_tile, index);

                  // //force a write_back.


                  //write_back_host(my_tile, replacement_pair.first, replacement_pair.second);



                  // //then delete
                  // if (map->remove_exact(my_tile, found_pair)){
                  //    map->insert_exact(my_tile, index, current_val);
                  // }

                  return;

               }

            }

         }




      }

      __device__ void write_back_host(cg::thread_block_tile<tile_size> my_tile, uint64_t index, uint64_t val){

         if (my_tile.thread_rank() == 0){


            if (!gallatin::utils::typed_atomic_write<uint64_t>(&host_items[index], ~0ULL, val)){
               #if CACHE_PRINT
               printf("Failed to write back %lu with value %lu\n", index, val);
               #endif

            }
           
         }

         my_tile.sync();


      }


      __device__ uint64_t load_index_host(cg::thread_block_tile<tile_size> my_tile, uint64_t index){
         
         uint64_t return_val;

         //global read
         if (my_tile.thread_rank() == 0){
            return_val = gallatin::utils::typed_atomic_exchange<uint64_t>(&host_items[index], ~0ULL);

            if (return_val == ~0ULL){
               #if CACHE_PRINT
               printf("index %lu already loaded\n", index);
               #endif
            }
         }

         return_val = my_tile.shfl(return_val, 0);

         return return_val;

      }

      __host__ void force_host_write_back(uint64_t n_host_items){


         write_back_host_kernel<my_type, tile_size><<<(n_host_items*tile_size -1)/256+1,256>>>(this, n_host_items);
         cudaDeviceSynchronize();


      }


       __host__ void check_compared_to_host(uint64_t * host_op_list, uint64_t n_ops, uint64_t n_host_items){


         printf("Starting host check...\n");

         force_host_write_back(n_host_items);

         //copy dev_op_list
         //uint64_t * host_op_list = gallatin::utils::copy_to_host<uint64_t>(dev_op_list, n_ops);

         uint64_t * host_array = gallatin::utils::get_host_version<uint64_t>(n_host_items);

         memset(host_array, 0, sizeof(uint64_t)*n_host_items);

         //set host_array
         for (uint64_t i = 0; i < n_ops; i++){

            uint64_t index = host_op_list[i] % n_host_items;

            //printf("Host modifying %lu\n", index);

            host_array[index]+=1;

         }


         uint64_t * dev_array = gallatin::utils::move_to_device<uint64_t>(host_array, n_host_items);

         compare_cache_arrays<my_type><<<(n_host_items-1)/256+1,256>>>(this, dev_array, n_host_items);

         cudaDeviceSynchronize();
         //cudaFreeHost(host_op_list);


         printf("Correctness check over.\n");

         //cudaFree(dev_array);


      }
      

   };

}  // namespace gallatin

#endif  // GPU_BLOCK_