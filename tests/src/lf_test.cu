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
#include <hashing_project/tables/iht_double_hashing.cuh>
#include <hashing_project/tables/double_hashing.cuh>
#include <hashing_project/tables/iht_p2.cuh>
#include <hashing_project/tables/chaining.cuh>
#include <hashing_project/tables/p2_hashing_metadata.cuh>
#include <hashing_project/tables/cuckoo.cuh>

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

   printf("Generation done\n");
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


template <typename ht_type, uint tile_size>
__global__ void insert_kernel(ht_type * table, DATA_TYPE * insert_buffer, uint64_t n_keys, uint64_t * misses){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys) return;

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }


   uint64_t my_key = insert_buffer[tid];

   if (!table->upsert_generic(my_tile, my_key, my_key)){

      //table->upsert_generic(my_tile, my_key, my_key);


      #if MEASURE_FAILS
      if (my_tile.thread_rank() == 0){

         atomicAdd((unsigned long long int *)&misses[0], 1ULL);
         //printf("Init upsert failed for %lu\n", my_key);
      }
      #endif
      
   } else {

      DATA_TYPE my_val;

      // if (!table->find_with_reference(my_tile, my_key, my_val)){




      //    table->find_with_reference(my_tile, my_key, my_val);
      //    //printf("Failed query\n");
      // }

   }


}


template <typename ht_type, uint tile_size>
__global__ void query_kernel(ht_type * table, DATA_TYPE * insert_buffer, uint64_t n_keys, uint64_t * misses){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys) return;

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }


   DATA_TYPE my_key = insert_buffer[tid];
   DATA_TYPE my_val;



   if (!table->find_with_reference(my_tile, my_key, my_val)){

      //table->upsert_generic(my_tile, my_key, my_key);


      #if MEASURE_FAILS
      if (my_tile.thread_rank() == 0){

         atomicAdd((unsigned long long int *)&misses[1], 1ULL);
         //printf("Init upsert failed for %lu\n", my_key);
      }
      #endif
      
   } else {

      if (my_val != my_key){


         table->find_with_reference(my_tile, my_key, my_val);

         atomicAdd((unsigned long long int *)&misses[1], 1ULL);
      }
   }


}


template <template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ void lf_test(uint64_t n_indices, DATA_TYPE * access_pattern){



   using ht_type = hash_table_type<DATA_TYPE, DATA_TYPE, tile_size, bucket_size>;


   //generate table and buffers
   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*4);

   cudaDeviceSynchronize();

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;


   #if COUNT_PROBES

   std::string filename = "results/lf_probe/";

   filename = filename + ht_type::get_name() + ".txt";


   printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query\n";


   #else

   std::string filename = "results/lf/";

   filename = filename + ht_type::get_name() + ".txt";


   printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query\n";

   #endif

  




   for (int i = 1; i < 19; i++){

      

      double lf = .05*i;

      ht_type * table = ht_type::generate_on_device(n_indices, 42);

      helpers::get_num_probes();

      uint64_t items_to_insert = lf*n_indices;

      DATA_TYPE * device_data = gallatin::utils::get_device_version<DATA_TYPE>(items_to_insert);

      //set original buffer
      cudaMemcpy(device_data, access_pattern, sizeof(DATA_TYPE)*items_to_insert, cudaMemcpyHostToDevice);

      cudaDeviceSynchronize();

      gallatin::utils::timer insert_timer;

      insert_kernel<ht_type, tile_size><<<(items_to_insert*tile_size-1)/256+1,256>>>(table, device_data, items_to_insert, misses);

      insert_timer.sync_end();

      uint64_t insert_probes = helpers::get_num_probes();

      cudaDeviceSynchronize();

      gallatin::utils::timer query_timer;

      query_kernel<ht_type, tile_size><<<(items_to_insert*tile_size-1)/256+1,256>>>(table, device_data, items_to_insert, misses);

      query_timer.sync_end();

      uint64_t query_probes = helpers::get_num_probes();


      //free tables and generate results
      cudaFree(device_data);

      ht_type::free_on_device(table);

      insert_timer.print_throughput("Inserted", items_to_insert);
      query_timer.print_throughput("Queried", items_to_insert);

      #if COUNT_PROBES

      printf("Probes %llu %llu\n", insert_probes, query_probes);
    
      myfile << lf << "," << std::setprecision(12) << 1.0*insert_probes/items_to_insert << "," << 1.0*query_probes/items_to_insert << "\n";

      #else

      myfile << lf << "," << std::setprecision(12) << 1.0*items_to_insert/(insert_timer.elapsed()*1000000) << "," << 1.0*items_to_insert/(query_timer.elapsed()*1000000) << "\n";

      #endif

      printf("Misses: %lu %lu\n", misses[0], misses[1]);

      misses[0] = 0;
      cudaDeviceSynchronize();

      //cuckoo is not leaking memory oon device.
      //gallatin::allocators::print_global_stats();


   }


   myfile.close();
 
  
   cudaFree(misses);
   cudaDeviceSynchronize();

}


int main(int argc, char** argv) {

   uint64_t table_capacity;



   if (argc < 2){
      table_capacity = 1000000;
   } else {
      table_capacity = std::stoull(argv[1]);
   }

   // if (argc < 3){
   //    n_rounds = 1000;
   // } else {
   //    n_rounds = std::stoull(argv[2]);
   // }


   if(fs::create_directory("results")){
    std::cout << "Created a directory\n";
   } else {
    std::cerr << "Failed to create a directory\n";
   }

   if(fs::create_directory("results/lf")){
    std::cout << "Created a directory\n";
   } else {
    std::cerr << "Failed to create a directory\n";
   }


   #if COUNT_PROBES

   if(fs::create_directory("results/lf_probe")){
    std::cout << "Created a directory\n";
   } else {
    std::cerr << "Failed to create a directory\n";
   }

   #endif

   auto access_pattern = generate_data<DATA_TYPE>(table_capacity);
   

   // lf_test<hashing_project::tables::p2_int_generic, 8, 32>(table_capacity, access_pattern);

   // lf_test<hashing_project::tables::p2_ext_generic, 8, 32>(table_capacity, access_pattern);

   // lf_test<hashing_project::tables::p2_ext_generic, 1, 32>(table_capacity, access_pattern);
   // lf_test<hashing_project::tables::p2_ext_generic, 2, 32>(table_capacity, access_pattern);
   // lf_test<hashing_project::tables::p2_ext_generic, 4, 32>(table_capacity, access_pattern);
   // lf_test<hashing_project::tables::p2_ext_generic, 8, 32>(table_capacity, access_pattern);
   // lf_test<hashing_project::tables::p2_ext_generic, 16, 32>(table_capacity, access_pattern);
   // lf_test<hashing_project::tables::p2_ext_generic, 32, 32>(table_capacity, access_pattern);

   // lf_test<hashing_project::tables::double_generic, 4, 8>(table_capacity, access_pattern);

   lf_test<hashing_project::tables::md_p2_generic, 4, 32>(table_capacity, access_pattern);

   
   init_global_allocator(15ULL*1024*1024*1024, 111);

   //lf_test<hashing_project::tables::chaining_generic, 4, 8>(table_capacity, access_pattern);

   //lf_test<hashing_project::tables::cuckoo_generic, 32, 32>(table_capacity, access_pattern);

   free_global_allocator();

   cudaDeviceSynchronize();
   

   //lf_test<hashing_project::tables::iht_p2_generic, 8, 32>(table_capacity, access_pattern);
   
  

   // lf_test<hashing_project::wrappers::warpcore_wrapper, 8, 8>(table_capacity, access_pattern);


   cudaFreeHost(access_pattern);




   cudaDeviceReset();
   return 0;

}
