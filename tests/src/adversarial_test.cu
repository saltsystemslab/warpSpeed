/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#define COUNT_PROBES 0

#define LOAD_CHEAP 0

#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <warpcore/single_value_hash_table.cuh>
#include <gallatin/allocators/timer.cuh>

#include <bght/p2bht.hpp>

#include <hashing_project/cache.cuh>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <chrono>
#include <openssl/rand.h>

#include <filesystem>

namespace fs = std::filesystem;


#include <hashing_project/table_wrappers/p2_wrapper.cuh>
#include <hashing_project/table_wrappers/dummy_ht.cuh>
#include <hashing_project/table_wrappers/iht_wrapper.cuh>
#include <hashing_project/table_wrappers/warpcore_wrapper.cuh>
#include <hashing_project/tables/p2_hashing_external.cuh>
#include <hashing_project/tables/p2_hashing_internal.cuh>
#include <hashing_project/tables/double_hashing.cuh>
#include <hashing_project/tables/iht_p2.cuh>
#include <hashing_project/tables/chaining.cuh>
#include <hashing_project/tables/p2_hashing_metadata.cuh>
#include <hashing_project/tables/cuckoo.cuh>

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
__global__ void generate_adverserial_workload(HT table, uint32_t num_keys, uint32_t * keys, uint32_t * first_round, uint32_t * second_round){

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
__global__ void round_1_adverserial_workload(HT table, uint32_t n_buckets, uint32_t * first_round){

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
__global__ void round_2_adverserial_workload(HT table, uint32_t n_buckets, uint32_t * first_round, uint32_t * second_round){


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
__global__ void round_3_adverserial_workload(HT table, uint32_t n_buckets, uint32_t * second_round, uint64_t * misses){

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

      generate_adverserial_workload<contextType><<<(num_keys-1)/512+1,512>>>(slabhash_context, num_keys, device_data, first_round, second_round);


      verify_setup_misses<<<(n_buckets-1)/512+1,512>>>(n_buckets, first_round, second_round, misses);

      cudaDeviceSynchronize();


      cudaMemcpy(device_data, data, sizeof(uint32_t)*num_keys, cudaMemcpyHostToDevice);

      cudaFree(device_data);
      cudaFreeHost(data);

      //printf("Round %d missed %lu times\n",gen_round, misses[0]);

   }

   cudaDeviceSynchronize();


   round_1_adverserial_workload<contextType><<<(n_buckets*32-1)/256+1,256>>>(slabhash_context, n_buckets, first_round);

   round_2_adverserial_workload<contextType><<<(n_buckets*32*3-1)/256+1,256>>>(slabhash_context, n_buckets, first_round, second_round);

   round_3_adverserial_workload<contextType><<<(n_buckets*32-1)/256+1,256>>>(slabhash_context, n_buckets, second_round, misses);

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
__global__ void generate_adverserial_workload_generic(HT * table, uint32_t num_keys, uint32_t * keys, uint32_t * first_round, uint32_t * second_round){


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
__global__ void round_1_adverserial_workload_generic(HT * table, uint32_t n_buckets, uint32_t * first_round){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_buckets) return;

   uint32_t my_key = first_round[tid];

   uint32_t my_val = 0;


   table->upsert_generic(my_tile, my_key, my_val);

}


template <typename HT, uint tile_size>
__global__ void round_2_adverserial_workload_generic(HT * table, uint32_t n_buckets, uint32_t * first_round, uint32_t * second_round){


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

      table->upsert_generic(my_tile, my_key, my_val);
   }

}


template <typename HT, uint tile_size>
__global__ void round_3_adverserial_workload_generic(HT * table, uint32_t n_buckets, uint32_t * second_round, uint64_t * misses){

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


   printf("Starting test for table %s\n", ht_type::get_name());

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

      generate_adverserial_workload_generic<ht_type, tile_size><<<(num_keys*tile_size-1)/256+1,256>>>(table, num_keys, device_data, first_round, second_round);

      verify_setup_misses<<<(n_buckets-1)/512+1,512>>>(n_buckets, first_round, second_round, misses);

      cudaDeviceSynchronize();

      cudaFree(device_data);
      cudaFreeHost(data);

   }

   //printf("Setup done\n");

   

   cudaDeviceSynchronize();


   round_1_adverserial_workload_generic<ht_type, tile_size><<<(n_buckets*tile_size-1)/256+1,256>>>(table, n_buckets, first_round);

   round_2_adverserial_workload_generic<ht_type, tile_size><<<(n_buckets*tile_size*3-1)/256+1,256>>>(table, n_buckets, first_round, second_round);

   round_3_adverserial_workload_generic<ht_type, tile_size><<<(n_buckets*tile_size-1)/256+1,256>>>(table, n_buckets, second_round, misses);

   cudaDeviceSynchronize();

   bool failed = false;
   for (int i =1; i < 5; i++){
      if (misses[i] != 0){
         failed = true;
      }
   }


   if (failed){
      printf("Table %s FAILED:\n- Query after delete: %lu\n- Delete after delete: %lu\n- Failed first query: %lu\n- Failed original delete: %lu\n", ht_type::get_name(), misses[1], misses[2], misses[3], misses[4]);
   } else {
      printf("Table %s PASSED\n", ht_type::get_name());
   }



   //free resources.

   cudaFree(first_round);
   cudaFree(second_round);

   cudaFree(misses);

   ht_type::free_on_device(table);


}




int main(int argc, char** argv) {

   uint64_t n_buckets;



   if (argc < 2){
      n_buckets = 1000000;
   } else {
      n_buckets = std::stoull(argv[1]);
   }


   //start of tests.
   test_slabhash(n_buckets);

   test_table<hashing_project::tables::md_p2_generic, 4, 32>(n_buckets);

   test_table<hashing_project::tables::p2_ext_generic, 8, 32>(n_buckets);

   test_table<hashing_project::tables::p2_int_generic, 8, 32>(n_buckets);

   test_table<hashing_project::tables::double_generic, 4, 8>(n_buckets);

   test_table<hashing_project::tables::iht_p2_generic, 8, 32>(n_buckets);


   init_global_allocator(16ULL*1024*1024*1024, 111);

   test_table<hashing_project::tables::chaining_generic, 4, 8>(n_buckets);

   free_global_allocator();




   cudaDeviceReset();
   return 0;

}
