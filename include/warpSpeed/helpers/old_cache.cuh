#ifndef CACHE
#define CACHE

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>

//#include <warpSpeed/helpers/fifo_queue.cuh>
#include <warpSpeed/helpers/fifo_queue.cuh>

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


namespace warpSpeed {

   template <template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
   struct ht_fifo_cache {


      using queue_type = warpSpeed::helpers::fifo_queue<uint64_t, ~0ULL>;

      using ht_type = hash_table_type<uint64_t, uint64_t, tile_size, bucket_size>;

      using my_type = ht_fifo_cache<hash_table_type, tile_size, bucket_size>;

      uint64_t * host_items;

      queue_type * fifo_queue;
      ht_type * map;


      uint64_t host_capacity;
      uint64_t cache_capacity;
      int live_items;



      __host__ static my_type * generate_on_device(uint64_t ext_host_capacity, uint64_t ext_cache_capacity, float cache_fill_ratio){


         my_type * host_version = gallatin::utils::get_host_version<my_type>();

         // size, sentinel, tombstone, sentinel_value
         //ht_type host_map(ext_cache_capacity, ext_host_capacity, ext_host_capacity+1, 0ULL);

         //ht_type * dev_table = gallatin::utils::get_device_version<ht_type>();

         //cudaMemcpy(dev_table, &host_map, sizeof(ht_type), cudaMemcpyHostToDevice);'

         printf("Starting test with hash table %s\n", ht_type::get_name());

         host_version->map = ht_type::generate_on_device(ext_cache_capacity, 424242ULL);

         host_version->fifo_queue = queue_type::generate_on_device(ext_cache_capacity);

         //this uses cudaMallocHost in the backend so safe.
         host_version->host_items = gallatin::utils::get_host_version<uint64_t>(ext_host_capacity);

         cudaMemset(host_version->host_items, 0, sizeof(uint64_t)*ext_host_capacity);

         host_version->host_capacity = ext_host_capacity;

         host_version->cache_capacity = ext_cache_capacity*cache_fill_ratio;

         host_version->live_items = 0ULL;

         my_type * device_version = gallatin::utils::move_to_device<my_type>(host_version);

         return device_version;



      }

      __host__ static void free_on_device(my_type * device_version){


         my_type * host_version = gallatin::utils::move_to_host<my_type>(device_version);

         queue_type::free_on_device(host_version->fifo_queue);

         ht_type::free_on_device(host_version->map);

         cudaFreeHost(host_version->host_items);

         cudaFreeHost(host_version);

      }


      __device__ uint64_t read_item(cg::thread_block_tile<tile_size> my_tile, uint64_t index){
         

        int n_iters = 0;

        while (n_iters < 1000){

          //if (my_tile.thread_rank() == 0) printf("Looping\n");


          uint64_t return_val;

          map->lock_key(my_tile, index);

          if (map->find_with_reference_no_lock(my_tile, index, return_val)){


            map->unlock_key(my_tile, index);
            return return_val;

          }



          bool registered = register_cache(my_tile);



          if (registered){

            //insert pipeline
            if (my_tile.thread_rank() == 0){

              //update cache.
              fifo_queue->enqueue(index);

              return_val = gallatin::utils::ld_acq(&host_items[index]);

            }

            return_val = my_tile.shfl(return_val, 0);
            
            bool success = map->upsert_no_lock(my_tile, index, return_val);

            if (!success){

              //map->upsert_no_lock(my_tile, index, return_val);

              if (my_tile.thread_rank() == 0){
                printf("Failed to insert %lu\n", index);
              }
            }

            map->unlock_key(my_tile, index);

            return return_val;

          } else {


            //need to update

            map->unlock_key(my_tile, index);

            remove_fifo_item_debug(my_tile);

            __threadfence();

            //if (my_tile.thread_rank() == 0) printf("Removed item, entering new round\n");

            


          }

          n_iters++;


        }



        printf("Reached end of execution\n");

      }

      // __device__ uint64_t read_item(cg::thread_block_tile<tile_size> my_tile, uint64_t index){

         

      //   while (true){


      //     uint64_t return_val;

      //     map->lock_key(my_tile, index);

      //     // if (map->find_with_reference_no_lock(my_tile, index, return_val)){


      //     //   map->unlock_key(my_tile, return_val);
      //     //   return return_val;

      //     // }

      //     // bool registered = false;

      //     // if (my_tile.thread_rank() == 0){

      //     //   registered = register_cache(index);



      //     // }


      //       //need to update

      //       map->unlock_key(my_tile, index);

      

      //       return 0;

      //     }


      // }


      __device__ void remove_fifo_item_debug(cg::thread_block_tile<tile_size> my_tile){

        uint64_t index;

        if (!deregister_cache(my_tile)) return;

        if (my_tile.thread_rank() == 0){

          fifo_queue->dequeue(index);
          //fifo_queue->enqueue(index);

        }

        index = my_tile.shfl(index, 0);

        map->lock_key(my_tile, index);


        //fifo_queue->enqueue(index);
        map->remove_no_lock(my_tile, index);

        map->unlock_key(my_tile, index);

      }


      __device__ bool deregister_cache(cg::thread_block_tile<tile_size> my_tile){

        bool ballot = false;
        if (my_tile.thread_rank() == 0){
          int live_item = atomicAdd(&live_items, -1);

          ballot = (live_item <= 0);

          if (!ballot) atomicAdd(&live_items, 1);
        }

        return my_tile.ballot(ballot);

      }

      __device__ bool register_cache(cg::thread_block_tile<tile_size> my_tile){


        bool ballot = false;

        if (my_tile.thread_rank() == 0){

          int live_item = atomicAdd(&live_items, 1);

          ballot = (live_item < cache_capacity);

          if (!ballot) atomicAdd(&live_items, -1);
        }

        return my_tile.ballot(ballot);


      }



      __device__ void remove_fifo_item(cg::thread_block_tile<tile_size> my_tile){



        int live_item;
        if (my_tile.thread_rank() == 0){
          live_item = atomicAdd(&live_items, -1);
        }
        
        live_item = my_tile.shfl(live_item, 0);

        if (live_item <= 0){
          //undo - no valid item

          if (my_tile.thread_rank() == 0){
            atomicAdd(&live_items, 1);
          }

          printf("No valid items\n");
          
          return;
        }

        //pull;

        uint64_t index;

        if (my_tile.thread_rank() == 0){

          if (!fifo_queue->dequeue(index)){
            printf("Failed to dequeue?\n");
          }

        }

        index = my_tile.shfl(index,0);

        write_back_host(my_tile, index);

        //map->remove_no_lock(my_tile, index);



      }



      __device__ void write_back_host(cg::thread_block_tile<tile_size> my_tile, uint64_t index){

        map->lock_key(my_tile, index);

        uint64_t value;

        if (!map->find_with_reference_no_lock(my_tile, index, value)){
          //failure - should never fail to find or delete a key that made it into the system.

          if (my_tile.thread_rank() == 0){
            printf("Failure to query %llu\n", index);
          }

          map->unlock_key(my_tile, index);
          return;
        }



        if (!map->remove_no_lock(my_tile, index)){
          
          if (my_tile.thread_rank() == 0){
            printf("Failure to remove %llu\n", index);
          }

          map->unlock_key(my_tile, index);
          return;
        }

        if (my_tile.thread_rank() == 0){
          gallatin::utils::st_rel(&host_items[index], value);
        }

        my_tile.sync();

        __threadfence();

        map->unlock_key(my_tile, index);


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