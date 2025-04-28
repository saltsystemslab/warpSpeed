/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

// A set of sanity benchmarks to assert that tables are performing operations correctly.
// Tests are:
// 1. Insert-Query-Delete: Insert, then query, then delete the same batch of keys over and over and over.
//    - Any reasonable hash table design should be able to handle this. 


#define COUNT_PROBES 0

#define LOAD_CHEAP 0

#include <argparse/argparse.hpp>

#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <warpcore/single_value_hash_table.cuh>
#include <gallatin/allocators/timer.cuh>

#include <bght/p2bht.hpp>

#

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <chrono>
#include <openssl/rand.h>

#include <filesystem>

namespace fs = std::filesystem;


//#include <warpSpeed/table_wrappers/p2_wrapper.cuh>
//#include <warpSpeed/table_wrappers/dummy_ht.cuh>
//#include <warpSpeed/table_wrappers/iht_wrapper.cuh>
#include <warpSpeed/table_wrappers/warpcore_wrapper.cuh>
#include <warpSpeed/tables/p2_hashing.cuh>
#include <warpSpeed/tables/p2_hashing_internal.cuh>
#include <warpSpeed/tables/double_hashing.cuh>
#include <warpSpeed/tables/iht_p2.cuh>
#include <warpSpeed/tables/chaining.cuh>
#include <warpSpeed/tables/p2_hashing_metadata.cuh>
#include <warpSpeed/tables/cuckoo.cuh>
#include <warpSpeed/tables/double_hashing_metadata.cuh>
#include <warpSpeed/tables/iht_p2_metadata_full.cuh>

#include <slabhash/gpu_hash_table.cuh>

#include <iostream>
#include <locale>


//thrust stuff.
#include <thrust/shuffle.h>
#include <thrust/random.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

using namespace gallatin::allocators;



#define MEASURE_FAILS 1

#define MEASURE_INDEPENDENT 1

#define PRINT_THROUGHPUT_ONLY 1


#define DATA_TYPE uint64_t


template <typename T>
__host__ T * generate_data(uint64_t nitems){


   //malloc space

   T * vals;

   cudaMallocHost((void **)&vals, sizeof(T)*nitems);


   //          100,000,000
   uint64_t cap = 100000000ULL;

   for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

      uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


      RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(T));



      to_fill += togen;

      //printf("Generated %llu/%llu\n", to_fill, nitems);

   }

   //printf("Generation done\n");
   return vals;
}


//generate data within the range from 0, cutoff
//modulus any keys which exceed the range.
//this generates a random list of keys to operate on.
template <typename T>
__host__ T * generate_clipped_data(uint64_t nitems, uint64_t cutoff){


   T * host_data = generate_data<T>(nitems);

   for (uint64_t i =0; i < nitems; i++){
      host_data[i] = host_data[i] % cutoff;
   }


   return host_data;


}


template <typename HT>
__global__ void generate_adversarial_workload(HT table, uint32_t num_keys, uint32_t * keys, uint32_t * first_round, uint32_t * second_round){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= num_keys) return;

   uint32_t my_key = keys[tid];

   uint32_t hash = table.computeBucket(my_key);

   //attempt to set 1/2 buckets.

   uint32_t result = atomicCAS(&first_round[hash], 0, my_key);

   if (result == 0 || result == my_key) return;
  
   atomicCAS(&second_round[hash], 0, my_key);



   //with sufficiently large setup this does work. - interestingly seems to be an off by 1 error?
   //bucket 0 is never selectable by table.computeBucket(my_key);
   //not sure why.

}


__global__ void verify_setup(uint32_t num_keys, uint32_t * first_round, uint32_t * second_round){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= num_keys) return;

   if (first_round[tid] == 0) printf("Fail on first for bucket %llu\n", tid);

   if (second_round[tid] == 0) printf("Fail on second for bucket %llu\n", tid);

   if (first_round[tid] == second_round[tid]) printf("Duplicates in %llu\n", tid);

}

__global__ void verify_setup_misses(uint32_t num_keys, uint32_t * first_round, uint32_t * second_round, uint64_t * misses){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= num_keys) return;

   if (first_round[tid] == 0) atomicAdd((unsigned long long int *)&misses[0], 1ULL); 

   if (second_round[tid] == 0) atomicAdd((unsigned long long int *)&misses[0], 1ULL);

   if (first_round[tid] == second_round[tid]) atomicAdd((unsigned long long int *)&misses[0], 1ULL);

}


template <typename HT>
__global__ void round_1_adversarial_workload(HT table, uint32_t n_buckets, uint32_t * first_round){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<32> my_tile = cg::tiled_partition<32>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_buckets) return;

   bool to_be_inserted = my_tile.thread_rank() == 0;

   //slabAlloc specific setup.
   uint32_t stid = threadIdx.x + blockIdx.x * blockDim.x;
   uint32_t slaneId = threadIdx.x & 0x1F;

   AllocatorContextT local_allocator_ctx(table.getAllocatorContext());
   local_allocator_ctx.initAllocator(stid, slaneId);

   uint32_t my_key = first_round[tid];

   uint32_t my_val = 0;

   uint32_t myBucket = table.computeBucket(my_key);

   table.insertPairUnique(to_be_inserted, slaneId, my_key, my_val, myBucket, local_allocator_ctx);




}


template <typename HT>
__global__ void round_2_adversarial_workload(HT table, uint32_t n_buckets, uint32_t * first_round, uint32_t * second_round){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<32> my_tile = cg::tiled_partition<32>(thread_block);


   uint64_t mega_tid = gallatin::utils::get_tile_tid(my_tile);


   uint64_t tid = mega_tid/3;

   uint64_t key_select = mega_tid % 3;

   if (tid >= n_buckets) return;

   bool to_be_inserted = (my_tile.thread_rank() == 0);

   //slabAlloc specific setup.
   uint32_t stid = threadIdx.x + blockIdx.x * blockDim.x;
   uint32_t slaneId = threadIdx.x & 0x1F;

   AllocatorContextT local_allocator_ctx(table.getAllocatorContext());
   local_allocator_ctx.initAllocator(stid, slaneId);


   if (key_select == 0){
      uint32_t my_key = first_round[tid];
      uint32_t my_val = 0;

      uint32_t myBucket = table.computeBucket(my_key);

      //table.insertPairUnique(to_be_inserted, slaneId, my_key, my_val, myBucket, local_allocator_ctx);
      table.deleteKey(to_be_inserted, slaneId, my_key, myBucket);

   } else {

      uint32_t my_key = second_round[tid];
      uint32_t my_val = key_select;

      uint32_t myBucket = table.computeBucket(my_key);

      table.insertPairUnique(to_be_inserted, slaneId, my_key, my_val, myBucket, local_allocator_ctx);

   }

}


template <typename HT>
__global__ void round_3_adversarial_workload(HT table, uint32_t n_buckets, uint32_t * second_round, uint64_t * misses){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<32> my_tile = cg::tiled_partition<32>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_buckets) return;

   bool to_be_inserted = (my_tile.thread_rank() == 0);

   //slabAlloc specific setup.
   uint32_t stid = threadIdx.x + blockIdx.x * blockDim.x;
   uint32_t slaneId = threadIdx.x & 0x1F;

   AllocatorContextT local_allocator_ctx(table.getAllocatorContext());
   local_allocator_ctx.initAllocator(stid, slaneId);

   uint32_t my_key = second_round[tid];

   uint32_t my_val = 0;

   uint32_t myBucket = table.computeBucket(my_key);


   uint32_t prev_val = 0;

   table.searchKey(to_be_inserted, slaneId, my_key, prev_val, myBucket);

   if (my_tile.thread_rank() == 0 && prev_val == SEARCH_NOT_FOUND){
      atomicAdd((unsigned long long int *)&misses[3], 1ULL);
   }


   to_be_inserted = (my_tile.thread_rank() == 0);




   bool deleted_first = table.deleteKey(to_be_inserted, slaneId, my_key, myBucket);

   if (my_tile.thread_rank() == 0 && !deleted_first){
      atomicAdd((unsigned long long int *)&misses[4], 1ULL);
   }

   to_be_inserted = (my_tile.thread_rank() == 0);


   table.searchKey(to_be_inserted, slaneId, my_key, my_val, myBucket);

   if (my_tile.thread_rank() == 0 && my_val != SEARCH_NOT_FOUND){

      atomicAdd((unsigned long long int *)&misses[1], 1ULL);
      //printf("Key still exists, bucket %llu prev val %u query value: %u\n", tid, prev_val, my_val);
   }


   to_be_inserted = (my_tile.thread_rank() == 0);


   bool deleted = table.deleteKey(to_be_inserted, slaneId, my_key, myBucket);


   if (my_tile.ballot(deleted)){

      if (my_tile.thread_rank() == 0){

         atomicAdd((unsigned long long int *)&misses[2], 1ULL);
         //printf("Double deletion in bucket %lu\n", tid);
      }
   }



}

template <typename HT, uint tile_size>
__global__ void generate_adversarial_workload_generic(HT * table, uint32_t num_keys, uint32_t * keys, uint32_t * first_round, uint32_t * second_round){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);;

   if (tid >= num_keys) return;

   uint32_t my_key = keys[tid];

   uint32_t hash = table->get_lock_bucket(my_tile, my_key);

   //attempt to set 1/2 buckets.

   if (my_tile.thread_rank() == 0){

      uint32_t result = atomicCAS(&first_round[hash], 0, my_key);

      if (result == 0 || result == my_key) return;
     
      atomicCAS(&second_round[hash], 0, my_key);

   }



}

template <typename HT, uint tile_size>
__global__ void generate_adversarial_workload_warpcore(HT * table, uint32_t n_buckets, uint32_t num_keys, uint32_t * keys, uint32_t * first_round, uint32_t * second_round){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);;

   if (tid >= num_keys) return;

   uint32_t my_key = keys[tid];

   uint32_t hash = table->get_lock_bucket(my_tile, my_key);

   if (hash >= n_buckets) return;

   //attempt to set 1/2 buckets.

   if (my_tile.thread_rank() == 0){

      uint32_t result = atomicCAS(&first_round[hash], 0, my_key);

      if (result == 0 || result == my_key) return;
     
      atomicCAS(&second_round[hash], 0, my_key);

   }



}

template <typename HT>
__global__ void iterative_IQD_workload_warpcore(HT * table, uint32_t n_buckets, uint32_t * first_round, uint64_t * misses){

   using namespace warpcore;
   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<8> my_tile = cg::tiled_partition<8>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_buckets) return;

   uint32_t my_key = first_round[tid];

   uint32_t my_val = 0;



   for (uint i = 0; i < 10000; i++){

      auto insert_status = table->insert(my_key, i, my_tile);
      
      if (insert_status != Status::none()){

         if (my_tile.thread_rank() == 0){
            atomicAdd((unsigned long long int *)&misses[0], 1ULL);
         }

         return;
         
      }

      if (insert_status == Status::probing_length_exceeded()){
         printf("Probe exceeded passed external!\n");
      }

      auto query_status = table->retrieve(my_key, my_val, my_tile);

      if (query_status != Status::none()){

         if (my_tile.thread_rank() == 0){
            atomicAdd((unsigned long long int *)&misses[1], 1ULL);
         }

         return;

      }

      if (my_val != i){
         printf("Bad read %u != %u\n", my_val, i);
      }

      auto delete_status = table->erase(my_key, my_tile);


      if (delete_status != Status::none()){

         if (my_tile.thread_rank() == 0){
            atomicAdd((unsigned long long int *)&misses[2], 1ULL);
         }
         return;
      }

   }





}


template <typename HT>
__global__ void round_2_adversarial_workload_warpcore(HT * table, uint32_t n_buckets, uint32_t * first_round, uint32_t * second_round, uint32_t * erase_set){

   using namespace warpcore;
   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<8> my_tile = cg::tiled_partition<8>(thread_block);


   uint64_t mega_tid = gallatin::utils::get_tile_tid(my_tile);


   uint64_t tid = mega_tid/4;

   uint64_t key_select = mega_tid % 4;

   if (tid >= n_buckets) return;

   if (key_select == 0){
      uint32_t my_key = first_round[tid];
      uint32_t my_val = 0;

      auto status = table->erase(my_key, my_tile);

      if (status != Status::none()){
         printf("Key failed to delete\n");
      }
      // __threadfence();

      if (my_tile.thread_rank() == 0){
         ht_store(&erase_set[tid], (uint32_t) 1);
         __threadfence();
      }

      my_tile.sync();

   } else if (key_select == 1) {

      uint32_t my_key = first_round[tid];

      uint32_t my_val = 0;

      bool set = false;

      while (!set){

         if (my_tile.thread_rank() == 0){

            __threadfence();
            set = hash_table_load(&erase_set[tid]);
         }

         set = my_tile.shfl(set, 0);
      }

      auto status = table->retrieve(my_key, my_val, my_tile);



      if (my_tile.thread_rank() == 0 && status == Status::none()){
         printf("Key found after deletion!!!\n");
      }

      my_tile.sync();





   } else {

      uint32_t my_key = second_round[tid];
      uint32_t my_val = key_select;

      __threadfence();
      table->insert(my_key, my_val, my_tile);

   }

}


template <typename HT>
__global__ void round_3_adversarial_workload_warpcore(HT * table, uint32_t n_buckets, uint32_t * second_round, uint64_t * misses){

   using namespace warpcore;
   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<8> my_tile = cg::tiled_partition<8>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_buckets) return;

   //slabAlloc specific setup.
   uint32_t my_key = second_round[tid];

   uint32_t my_val = 0;


   auto status = table->retrieve(my_key, my_val, my_tile);



   if (my_tile.thread_rank() == 0 && status == Status::key_not_found()){
      atomicAdd((unsigned long long int *)&misses[3], 1ULL);
   }



   auto delete_status = table->erase(my_key, my_tile);


   if (my_tile.thread_rank() == 0 && delete_status != Status::none()){
      atomicAdd((unsigned long long int *)&misses[4], 1ULL);
   }


   auto second_query = table->retrieve(my_key, my_val, my_tile);


   if (my_tile.thread_rank() == 0 && second_query != Status::key_not_found()){

      atomicAdd((unsigned long long int *)&misses[1], 1ULL);
      //printf("Key still exists, bucket %llu prev val %u query value: %u\n", tid, prev_val, my_val);
   }



   auto second_delete_status = table->erase(my_key, my_tile);



   if (second_delete_status != Status::key_not_found()){

      if (my_tile.thread_rank() == 0){

         atomicAdd((unsigned long long int *)&misses[2], 1ULL);
         //printf("Double deletion in bucket %lu\n", tid);
      }
   }



}

__host__ void test_warpcore(uint64_t n_buckets){

   printf("Starting test for warpcore\n");

   uint32_t num_keys = n_buckets*8;

   uint64_t seed = 461; 

   using namespace warpcore;

   using Key = uint32_t;
   using Value = uint32_t;

   //table setup
   using probing_scheme_t = defaults::probing_scheme_t<Key, 8>;

    using warpcore_type =
        SingleValueHashTable<
            Key, Value,
            defaults::empty_key<Key>(),
            defaults::tombstone_key<Key>(),
            probing_scheme_t,
            defaults::table_storage_t<Key, Value>,
            defaults::temp_memory_bytes()>;


   warpcore_type * host_table = new warpcore_type(num_keys);

   warpcore_type * table = gallatin::utils::get_device_version<warpcore_type>();

   cudaMemcpy(table, host_table, sizeof(warpcore_type), cudaMemcpyHostToDevice);


   // warpcore_type table(num_keys);
   

   //arrays for insert data
   uint32_t * first_round;

   uint32_t * second_round;

   cudaMalloc((void **)&first_round, sizeof(uint32_t)*n_buckets);
   cudaMalloc((void **)&second_round, sizeof(uint32_t)*n_buckets);

   cudaMemset(first_round, 0, sizeof(uint32_t)*n_buckets);
   cudaMemset(second_round, 0, sizeof(uint32_t)*n_buckets);

   uint32_t * erase_set;
   cudaMalloc((void **)&erase_set, sizeof(uint32_t)*n_buckets);
   cudaMemset(erase_set, 0, sizeof(uint32_t)*n_buckets);

   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*5);

   misses[0] = 1;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;


   int gen_round = 0;

   while (misses[0] != 0){

      misses[0] = 0;


      //printf("Starting generation round %d\n", gen_round);

      gen_round++;

      uint32_t * data = generate_data<uint32_t>(num_keys);

      uint32_t * device_data;

      cudaMalloc((void **)&device_data, sizeof(uint32_t)*num_keys);

      cudaMemcpy(device_data, data, sizeof(uint32_t)*num_keys, cudaMemcpyHostToDevice);

      generate_adversarial_workload_warpcore<warpcore_type, 8><<<(num_keys-1)/512+1,512>>>(table, n_buckets, num_keys, device_data, first_round, second_round);

      verify_setup_misses<<<(n_buckets-1)/512+1,512>>>(n_buckets, first_round, second_round, misses);

      cudaDeviceSynchronize();


      cudaMemcpy(device_data, data, sizeof(uint32_t)*num_keys, cudaMemcpyHostToDevice);

      cudaFree(device_data);
      cudaFreeHost(data);

      //printf("Round %d missed %lu times\n",gen_round, misses[0]);

   }

   cudaDeviceSynchronize();


   iterative_IQD_workload_warpcore<warpcore_type><<<(n_buckets*8-1)/256+1,256>>>(table, n_buckets, first_round, misses);

   cudaDeviceSynchronize();


   bool failed = false;
   for (int i =0; i < 3; i++){
      if (misses[i] != 0){
         failed = true;
      }
   }


   if (failed){
      printf("Table Warpcore FAILED:\n- Inserts: %lu\n- Queries: %lu\n- Deletions: %lu\n", misses[0], misses[1], misses[2]);
   } else {
      printf("Table Warpcore PASSED\n");
   }



   //free

   cudaFree(first_round);
   cudaFree(second_round);
   cudaFree(erase_set);
   cudaFree(table);
   delete host_table;



}


__host__ void test_slabhash(uint64_t n_buckets){

   printf("Starting test for slabhash\n");

   uint32_t num_keys = n_buckets*32;

   uint64_t seed = 461; 

   //table setup

   using slabhash_type = gpu_hash_table<uint32_t, uint32_t, SlabHashTypeT::ConcurrentMap>;


   slabhash_type slabhash_table(num_keys, n_buckets, 0, seed);


   using gpu_slabhash_type = GpuSlabHash<uint32_t, uint32_t, SlabHashTypeT::ConcurrentMap>;
   gpu_slabhash_type * dev_table = slabhash_table.slab_hash_;

   using contextType = GpuSlabHashContext<uint32_t, uint32_t, SlabHashTypeT::ConcurrentMap>;

   contextType slabhash_context = dev_table->gpu_context_;


   //arrays for insert data
   uint32_t * first_round;

   uint32_t * second_round;

   cudaMalloc((void **)&first_round, sizeof(uint32_t)*n_buckets);
   cudaMalloc((void **)&second_round, sizeof(uint32_t)*n_buckets);

   cudaMemset(first_round, 0, sizeof(uint32_t)*n_buckets);
   cudaMemset(second_round, 0, sizeof(uint32_t)*n_buckets);

   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*5);

   misses[0] = 1;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;


   int gen_round = 0;

   while (misses[0] != 0){

      misses[0] = 0;


      //printf("Starting generation round %d\n", gen_round);

      gen_round++;

      uint32_t * data = generate_data<uint32_t>(num_keys);

      uint32_t * device_data;

      cudaMalloc((void **)&device_data, sizeof(uint32_t)*num_keys);

      cudaMemcpy(device_data, data, sizeof(uint32_t)*num_keys, cudaMemcpyHostToDevice);

      generate_adversarial_workload<contextType><<<(num_keys-1)/512+1,512>>>(slabhash_context, num_keys, device_data, first_round, second_round);


      verify_setup_misses<<<(n_buckets-1)/512+1,512>>>(n_buckets, first_round, second_round, misses);

      cudaDeviceSynchronize();


      cudaMemcpy(device_data, data, sizeof(uint32_t)*num_keys, cudaMemcpyHostToDevice);

      cudaFree(device_data);
      cudaFreeHost(data);

      //printf("Round %d missed %lu times\n",gen_round, misses[0]);

   }

   cudaDeviceSynchronize();


   round_1_adversarial_workload<contextType><<<(n_buckets*32-1)/256+1,256>>>(slabhash_context, n_buckets, first_round);

   round_2_adversarial_workload<contextType><<<(n_buckets*32*3-1)/256+1,256>>>(slabhash_context, n_buckets, first_round, second_round);

   round_3_adversarial_workload<contextType><<<(n_buckets*32-1)/256+1,256>>>(slabhash_context, n_buckets, second_round, misses);

   cudaDeviceSynchronize();


   bool failed = false;
   for (int i =1; i < 5; i++){
      if (misses[i] != 0){
         failed = true;
      }
   }


   if (failed){
      printf("Table slabhash FAILED:\n- Query after delete: %lu\n- Delete after delete: %lu\n- Failed first query: %lu\n- Failed original delete: %lu\n", misses[1], misses[2], misses[3], misses[4]);
   } else {
      printf("Table slabhash PASSED\n");
   }



   //free

   cudaFree(first_round);
   cudaFree(second_round);


}






template <typename HT, uint tile_size>
__global__ void round_1_adversarial_workload_generic(HT * table, uint32_t n_buckets, uint32_t * first_round){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_buckets) return;

   uint32_t my_key = first_round[tid];

   uint32_t my_val = 0;


   table->upsert_replace(my_tile, my_key, my_val);

}


template <typename HT, uint tile_size>
__global__ void round_2_adversarial_workload_generic(HT * table, uint32_t n_buckets, uint32_t * first_round, uint32_t * second_round){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);

   uint64_t mega_tid = gallatin::utils::get_tile_tid(my_tile);

   uint64_t tid = mega_tid/3;

   uint64_t key_select = mega_tid % 3;

   if (tid >= n_buckets) return;


   if (key_select == 0){
      uint32_t my_key = first_round[tid];
      uint32_t my_val = 0;

      table->remove(my_tile, my_key);

   } else {

      uint32_t my_key = second_round[tid];
      uint32_t my_val = key_select;

      table->upsert_replace(my_tile, my_key, my_val);
   }

}


template <typename HT, uint tile_size>
__global__ void round_3_adversarial_workload_generic(HT * table, uint32_t n_buckets, uint32_t * second_round, uint64_t * misses){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_buckets) return;

   bool to_be_inserted = (my_tile.thread_rank() == 0);

   uint32_t my_key = second_round[tid];

   uint32_t my_val = 0;

   uint32_t prev_val = 0;


   if (!table->find_with_reference(my_tile, my_key, prev_val)){

      if (my_tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *)&misses[3], 1ULL);
      }
   }

   if (!table->remove(my_tile, my_key)){

      if (my_tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *)&misses[4], 1ULL);
      }

   }

   if (table->find_with_reference(my_tile, my_key, my_val)){


      //uint64_t key_bucket = table->get_lock_bucket(my_tile, my_key);
      
      if (my_tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *)&misses[1], 1ULL);
      }
     
   }


   if (table->remove(my_tile, my_key)){

      //uint64_t key_bucket = table->get_lock_bucket(my_tile, my_key);
      
      if (my_tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *)&misses[2], 1ULL);
      }

   }


}


template <template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ void test_table(uint64_t n_buckets){


   using ht_type = hash_table_type<uint32_t, uint32_t, tile_size, bucket_size>;


   printf("Starting test for table %s\n", ht_type::get_name().c_str());

   uint32_t num_keys = n_buckets*bucket_size;

   //uint64_t generated_keys = num_keys*8;

   uint64_t seed = 461; 

   ht_type * table = ht_type::generate_on_device(num_keys, 42);

   n_buckets = table->get_num_locks();

   uint32_t * first_round;

   uint32_t * second_round;

   cudaMalloc((void **)&first_round, sizeof(uint32_t)*n_buckets);
   cudaMalloc((void **)&second_round, sizeof(uint32_t)*n_buckets);

   cudaMemset(first_round, 0, sizeof(uint32_t)*n_buckets);
   cudaMemset(second_round, 0, sizeof(uint32_t)*n_buckets);

   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*5);

   misses[0] = 1;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;


   int gen_round = 0;

   //generate new batches of data until all buckets have 2 keys.
   while (misses[0] != 0){

      misses[0] = 0;

      //printf("Starting generation round %d\n", gen_round);

      gen_round++;

      uint32_t * data = generate_data<uint32_t>(num_keys);

      uint32_t * device_data;

      cudaMalloc((void **)&device_data, sizeof(uint32_t)*num_keys);


      cudaMemcpy(device_data, data, sizeof(uint32_t)*num_keys, cudaMemcpyHostToDevice);

      generate_adversarial_workload_generic<ht_type, tile_size><<<(num_keys*tile_size-1)/256+1,256>>>(table, num_keys, device_data, first_round, second_round);

      verify_setup_misses<<<(n_buckets-1)/512+1,512>>>(n_buckets, first_round, second_round, misses);

      cudaDeviceSynchronize();

      cudaFree(device_data);
      cudaFreeHost(data);

   }

   //printf("Setup done\n");

   

   cudaDeviceSynchronize();


   round_1_adversarial_workload_generic<ht_type, tile_size><<<(n_buckets*tile_size-1)/256+1,256>>>(table, n_buckets, first_round);

   round_2_adversarial_workload_generic<ht_type, tile_size><<<(n_buckets*tile_size*3-1)/256+1,256>>>(table, n_buckets, first_round, second_round);

   round_3_adversarial_workload_generic<ht_type, tile_size><<<(n_buckets*tile_size-1)/256+1,256>>>(table, n_buckets, second_round, misses);

   cudaDeviceSynchronize();

   bool failed = false;
   for (int i =1; i < 5; i++){
      if (misses[i] != 0){
         failed = true;
      }
   }


   if (failed){
      printf("Table %s FAILED:\n- Query after delete: %lu\n- Delete after delete: %lu\n- Failed first query: %lu\n- Failed original delete: %lu\n", ht_type::get_name().c_str(), misses[1], misses[2], misses[3], misses[4]);
   } else {
      printf("Table %s PASSED\n", ht_type::get_name().c_str());
   }



   //free resources.

   cudaFree(first_round);
   cudaFree(second_round);

   cudaFree(misses);

   ht_type::free_on_device(table);


}


__host__ void execute_test(std::string table, uint64_t table_capacity){


   if (table == "p2"){

      test_table<warpSpeed::tables::p2_ext_generic, 8, 32>(table_capacity);

      //p2 p2MD double doubleMD iceberg icebergMD cuckoo chaining bght_p2 bght_cuckoo");


   } else if (table == "p2MD"){

      test_table<warpSpeed::tables::md_p2_generic, 4, 32>(table_capacity);

   } else if (table == "double"){
      test_table<warpSpeed::tables::double_generic, 8, 8>(table_capacity);

   } else if (table == "doubleMD"){

      test_table<warpSpeed::tables::md_double_generic, 4, 32>(table_capacity);


   } else if (table == "iceberg"){

      test_table<warpSpeed::tables::iht_p2_generic, 8, 32>(table_capacity);
     
   } else if (table == "icebergMD"){

      test_table<warpSpeed::tables::iht_p2_metadata_full_generic, 4, 32>(table_capacity);

   } else if (table == "cuckoo") {
       test_table<warpSpeed::tables::cuckoo_generic, 4, 8>(table_capacity);
   
   } else if (table == "chaining"){

      init_global_allocator(20ULL*1024*1024*1024, 111);

      test_table<warpSpeed::tables::chaining_generic, 4, 8>(table_capacity);

      free_global_allocator();
   } else if (table == "slabhash") {

      test_slabhash(table_capacity);

   } else if (table == "warpcore") {

      test_warpcore(table_capacity);

   } else {
      throw std::runtime_error("Unknown table");
   }


}


int main(int argc, char** argv) {

   // uint64_t n_buckets;



   // if (argc < 2){
   //    n_buckets = 1000000;
   // } else {
   //    n_buckets = std::stoull(argv[1]);
   // }


   argparse::ArgumentParser program("adverarial_test");

   // program.add_argument("square")
   // .help("display the square of a given integer")
   // .scan<'i', int>();

   program.add_argument("--table", "-t")
   .required()
   .help("Specify table type. Options [p2 p2MD double doubleMD iceberg icebergMD cuckoo chaining slabhash warpcore");

   program.add_argument("--capacity", "-c").required().scan<'u', uint64_t>().help("Number of slots in the table. Default is 100,000,000");

   try {
    program.parse_args(argc, argv);
   }
   catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
   }

   auto table = program.get<std::string>("--table");
   auto table_capacity = program.get<uint64_t>("--capacity");


   //start of tests.
   execute_test(table, table_capacity);

   // test_table<warpSpeed::tables::md_p2_generic, 4, 32>(n_buckets);

   // test_table<warpSpeed::tables::p2_ext_generic, 8, 32>(n_buckets);

   // test_table<warpSpeed::tables::double_generic, 4, 8>(n_buckets);

   // test_table<warpSpeed::tables::iht_p2_generic, 8, 32>(n_buckets);


   // init_global_allocator(16ULL*1024*1024*1024, 111);

   // test_table<warpSpeed::tables::chaining_generic, 4, 8>(n_buckets);

   // free_global_allocator();

   // test_table<warpSpeed::tables::cuckoo_generic, 4, 8>(n_buckets);

   // //test_table<warpSpeed::wrappers::warpcore_wrapper, 8, 8>(n_buckets);

   // test_table<warpSpeed::tables::iht_p2_metadata_full_generic, 4, 32>(n_buckets);
   
   // test_table<warpSpeed::tables::md_double_generic, 4, 32>(n_buckets);


   cudaDeviceReset();
   return 0;

}
