#ifndef HOST_CACHE
#define HOST_CACHE

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>

//#include <hashing_project/helpers/fifo_queue.cuh>
#include <hashing_project/helpers/fifo_queue_wrap.cuh>

#include<hashing_project/helpers/cache_counters.cuh>
#include <hashing_project/helpers/cache_cycle_counters.cuh>

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

//disabled unless specifically enabled.
#ifndef COUNT_CACHE_OPS
#define COUNT_CACHE_OPS 0
#endif

#ifndef COUNT_CACHE_CYCLES
#define COUNT_CACHE_CYCLES 0
#endif


//cache protocol
//query cache
//on success add to pin?
//need delete from potential buckets implementation - need to download warpcore...
//buidld with primary p2bht first.

namespace hashing_project {

   template <uint tile_size>
   struct host_cache {

      using my_type = host_cache<tile_size>;

      uint64_t * host_items;


      uint64_t host_capacity;
      uint64_t cache_capacity;
      int live_items;

      static char * get_name(){
        return "host_cache";
      }


      __host__ static my_type * generate_on_device(uint64_t ext_host_capacity, uint64_t ext_cache_capacity, float cache_fill_ratio){


         my_type * host_version = gallatin::utils::get_host_version<my_type>();

         // size, sentinel, tombstone, sentinel_value
         //ht_type host_map(ext_cache_capacity, ext_host_capacity, ext_host_capacity+1, 0ULL);

         //ht_type * dev_table = gallatin::utils::get_device_version<ht_type>();

         //cudaMemcpy(dev_table, &host_map, sizeof(ht_type), cudaMemcpyHostToDevice);'

         printf("Starting test with hash table host cache\n");

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

         cudaFreeHost(host_version->host_items);

         cudaFreeHost(host_version);

      }




      __host__ void print_space_usage(){

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         printf("Cache using %llu\n", host_version->host_capacity*(16));
            
         //uint64_t capacity = host_version->n_buckets*(sizeof(bucket_type)+sizeof(md_bucket_type)) + (host_version->n_buckets-1)/8+1; 

         cudaFreeHost(host_version);

      }
      
      //read item from cache - loads from host if not found.
      __device__ uint64_t read_item(cg::thread_block_tile<tile_size> my_tile, uint64_t index){

        START_TOTAL_THROUGHPUT


        uint64_t val;

        if (my_tile.thread_rank() == 0){

          __threadfence();
          val = gallatin::utils::ld_acq(&host_items[index]);

        }

        my_tile.shfl(val, 0);

        END_TOTAL_THROUGHPUT


        return val;

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


      // __device__ void remove_fifo_item_debug(cg::thread_block_tile<tile_size> my_tile){

      //   uint64_t index;

      //   if (!deregister_cache(my_tile)) return;

      //   if (my_tile.thread_rank() == 0){

      //     fifo_queue->dequeue(index);
      //     //fifo_queue->enqueue(index);

      //   }

      //   index = my_tile.shfl(index, 0);

      //   map->lock_key(my_tile, index);


      //   //fifo_queue->enqueue(index);
      //   map->remove_no_lock(my_tile, index);

      //   map->unlock_key(my_tile, index);

      // }


      // __device__ bool deregister_cache(cg::thread_block_tile<tile_size> my_tile){

      //   bool ballot = false;
      //   if (my_tile.thread_rank() == 0){
      //     int live_item = atomicAdd(&live_items, -1);

      //     ballot = (live_item <= 0);

      //     if (!ballot) atomicAdd(&live_items, 1);
      //   }

      //   return my_tile.ballot(ballot);

      // }

      // __device__ bool register_cache(cg::thread_block_tile<tile_size> my_tile){


      //   bool ballot = false;

      //   if (my_tile.thread_rank() == 0){

      //     int live_item = atomicAdd(&live_items, 1);

      //     ballot = (live_item < cache_capacity);

      //     if (!ballot) atomicAdd(&live_items, -1);
      //   }

      //   return my_tile.ballot(ballot);


      // }



      // __device__ void remove_fifo_item(cg::thread_block_tile<tile_size> my_tile){



      //   int live_item;
      //   if (my_tile.thread_rank() == 0){
      //     live_item = atomicAdd(&live_items, -1);
      //   }
        
      //   live_item = my_tile.shfl(live_item, 0);

      //   if (live_item <= 0){
      //     //undo - no valid item

      //     if (my_tile.thread_rank() == 0){
      //       atomicAdd(&live_items, 1);
      //     }

      //     printf("No valid items\n");
          
      //     return;
      //   }

      //   //pull;

      //   uint64_t index;

      //   if (my_tile.thread_rank() == 0){

      //     if (!fifo_queue->dequeue(index)){
      //       printf("Failed to dequeue?\n");
      //     }

      //   }

      //   index = my_tile.shfl(index,0);

      //   write_back_host(my_tile, index);

      //   //map->remove_no_lock(my_tile, index);



      // }



      // __device__ void write_back_host(cg::thread_block_tile<tile_size> my_tile, uint64_t index){

      //   map->lock_key(my_tile, index);

      //   uint64_t value;

      //   if (!map->find_with_reference_no_lock(my_tile, index, value)){
      //     //failure - should never fail to find or delete a key that made it into the system.

      //     if (my_tile.thread_rank() == 0){
      //       printf("Failure to query %llu\n", index);
      //     }

      //     map->unlock_key(my_tile, index);
      //     return;
      //   }



      //   if (!map->remove_no_lock(my_tile, index)){
          
      //     if (my_tile.thread_rank() == 0){
      //       printf("Failure to remove %llu\n", index);
      //     }

      //     map->unlock_key(my_tile, index);
      //     return;
      //   }

      //   if (my_tile.thread_rank() == 0){
      //     gallatin::utils::st_rel(&host_items[index], value);
      //   }

      //   my_tile.sync();

      //   __threadfence();

      //   map->unlock_key(my_tile, index);


      // }


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