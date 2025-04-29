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
#include <warpSpeed/tables/p2_hashing.cuh>
#include <warpSpeed/tables/double_hashing.cuh>
#include <warpSpeed/tables/iht_p2.cuh>
#include <warpSpeed/tables/chaining.cuh>
#include <warpSpeed/tables/p2_hashing_metadata.cuh>
#include <warpSpeed/tables/cuckoo.cuh>
#include <warpSpeed/tables/double_hashing_metadata.cuh>
#include <warpSpeed/tables/iht_metadata.cuh>

#include <gpu_hash_table.cuh>

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
__global__ void slabhash_IQD(HT table, uint32_t n_buckets, uint64_t * misses){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<32> my_tile = cg::tiled_partition<32>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid != 0) return;

   bool to_be_inserted = my_tile.thread_rank() == 0;

   //slabAlloc specific setup.
   uint32_t stid = threadIdx.x + blockIdx.x * blockDim.x;
   uint32_t slaneId = threadIdx.x & 0x1F;

   AllocatorContextT local_allocator_ctx(table.getAllocatorContext());
   local_allocator_ctx.initAllocator(stid, slaneId);



   for (int i = 0; i < 10000; i++){

      uint32_t my_key = 15102315094359844531ULL;

      uint32_t my_val = 0;

      uint32_t myBucket = table.computeBucket(my_key);


      table.insertPairUnique(to_be_inserted, slaneId, my_key, my_val, myBucket, local_allocator_ctx);

      to_be_inserted = my_tile.thread_rank() == 0;

      bool deleted = table.deleteKey(to_be_inserted, slaneId, my_key, myBucket);

      if (!my_tile.ballot(deleted) && my_tile.thread_rank() == 0){
         atomicAdd(&misses[0], 1);
      }

   }

   




}

template <typename HT>
__global__ void iterative_IQD_workload_warpcore(HT * table, uint32_t n_buckets, uint64_t * misses){

   using namespace warpcore;
   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<8> my_tile = cg::tiled_partition<8>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid != 0) return;

   uint32_t my_key = 15102315094359844531ULL;

   
   for (uint i = 0; i < 10000; i++){

      uint32_t my_val = 0;

      auto insert_status = table->insert(my_key, i, my_tile);
      
      if (insert_status != Status::none()){

         printf("Insert failed\n");

         if (my_tile.thread_rank() == 0){
            atomicAdd((unsigned long long int *)&misses[0], 1ULL);
         }

         
      }

      if (insert_status == Status::probing_length_exceeded()){
         printf("Probe exceeded passed external!\n");
      }

      auto query_status = table->retrieve(my_key, my_val, my_tile);

      if (query_status != Status::none()){

         printf("Query failed\n");

         if (my_tile.thread_rank() == 0){
            atomicAdd((unsigned long long int *)&misses[1], 1ULL);
         }


      }

      if (my_val != i){
         printf("Bad read %u != %u\n", my_val, i);
      }

      auto delete_status = table->erase(my_key, my_tile);


      if (delete_status != Status::none()){

         printf("Delete failed\n");

         if (my_tile.thread_rank() == 0){
            atomicAdd((unsigned long long int *)&misses[2], 1ULL);
         }
      }

      auto second_query_status = table->retrieve(my_key, my_val, my_tile);

      if (query_status == Status::none()){

         //printf("Query suceeded on deleted key, val is %u\n", my_val);

         if (my_tile.thread_rank() == 0){
            atomicAdd((unsigned long long int *)&misses[3], 1ULL);
         }


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

   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*5);

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;


   int gen_round = 0;

   cudaDeviceSynchronize();


   iterative_IQD_workload_warpcore<warpcore_type><<<(n_buckets*8-1)/256+1,256>>>(table, n_buckets, misses);

   cudaDeviceSynchronize();


   bool failed = false;
   for (int i =0; i < 4; i++){
      if (misses[i] != 0){
         failed = true;
      }
   }


   if (failed){
      printf("Table Warpcore FAILED:\n- Inserts: %lu\n- Queries: %lu\n- Deletions: %lu\n- Queries on Deleted %lu\n", misses[0], misses[1], misses[2], misses[3]);
   } else {
      printf("Table Warpcore PASSED\n");
   }



   //free
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

   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*5);

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;

   cudaDeviceSynchronize();


   slabhash_IQD<contextType><<<(n_buckets*32-1)/256+1,256>>>(slabhash_context, n_buckets, misses);

   cudaDeviceSynchronize();


   bool failed = false;
   for (int i =0; i < 5; i++){
      if (misses[i] != 0){
         failed = true;
      }
   }



   if (failed){
      printf("Table SlabHash FAILED:\n- Inserts: %lu\n- Queries: %lu\n- Deletions: %lu\n", misses[0], misses[1], misses[2]);
   } else {
      printf("Table SlabHash PASSED\n");
   }



}






template <typename HT, uint tile_size>
__global__ void IQD_generic(HT * table, uint32_t n_buckets, uint64_t * misses){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid != 0) return;

   uint32_t my_key = 15102315094359844531ULL;

   uint32_t my_val = 0;


   for (int i = 0; i < 10000; i++){

      if (!table->upsert_replace(my_tile, my_key, i) && my_tile.thread_rank() == 0){
         atomicAdd(&misses[0], 1);
      }

      if (!table->find_with_reference(my_tile, my_key, my_val) && my_tile.thread_rank() == 0){
         atomicAdd(&misses[1],1);
      }

      if (!table->remove(my_tile, my_key) && my_tile.thread_rank() == 0){
         atomicAdd(&misses[2], 1);
      }

      if (table->find_with_reference(my_tile, my_key, my_val) && my_tile.thread_rank() == 0){
         atomicAdd(&misses[3],1);
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

   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*5);

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;


   int gen_round = 0;

   //generate new batches of data until all buckets have 2 keys.

   //printf("Setup done\n");

   

   cudaDeviceSynchronize();


   IQD_generic<ht_type, tile_size><<<(n_buckets*tile_size-1)/256+1,256>>>(table, n_buckets, misses);

   cudaDeviceSynchronize();

   bool failed = false;
   for (int i =0; i < 5; i++){
      if (misses[i] != 0){
         failed = true;
      }
   }


   if (failed){
      printf("Table %s FAILED:\n- Inserts: %lu\n- Queries: %lu\n- Deletions: %lu\n- Queries after delete: %lu\n", ht_type::get_name().c_str(), misses[0], misses[1], misses[2], misses[3]);
   } else {
      printf("Table %s PASSED\n", ht_type::get_name().c_str());
   }



   //free resources.

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

      test_table<warpSpeed::tables::iht_metadata_generic, 4, 32>(table_capacity);

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
