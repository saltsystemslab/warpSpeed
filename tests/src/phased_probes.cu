/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#define COUNT_PROBES 1

#define LOAD_CHEAP 1

#include <argparse/argparse.hpp>

#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <warpcore/single_value_hash_table.cuh>
#include <gallatin/allocators/timer.cuh>

#include <bght/p2bht.hpp>
#include <bght/bcht.hpp>
#include <bght/iht.hpp>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <chrono>
#include <openssl/rand.h>

#include <filesystem>

namespace fs = std::filesystem;




#include <warpSpeed/tables/p2_hashing.cuh>

#include <warpSpeed/tables/double_hashing.cuh>
#include <warpSpeed/tables/iht_p2.cuh>
#include <warpSpeed/tables/chaining.cuh>
#include <warpSpeed/tables/p2_hashing_metadata.cuh>

#include <warpSpeed/tables/iht_p2_metadata_full.cuh>
#include <warpSpeed/tables/cuckoo.cuh>
#include <warpSpeed/tables/double_hashing_metadata.cuh>



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


#define LARGE_MD_LOAD 1
#define LARGE_BUCKET_MODS 0


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

      //printf("Generated %lu/%lu\n", to_fill, nitems);

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

   if (!table->upsert_replace(my_tile, my_key, my_key)){

      //table->upsert_replace(my_tile, my_key, my_key);


      #if MEASURE_FAILS
      if (my_tile.thread_rank() == 0){

         atomicAdd((unsigned long long int *)&misses[0], 1ULL);
         //printf("Init upsert failed for %lu\n", my_key);
      }
      #endif
      
   } else {
      

      // DATA_TYPE my_val;

      // if (!table->find_with_reference(my_tile, my_key, my_val)){

      //    table->upsert_replace(my_tile, my_key, my_key);

      //    table->find_with_reference(my_tile, my_key, my_val);
      //    //printf("Failed query\n");
      // }

   }


}

template <typename ht_type, uint tile_size>
__global__ void remove_kernel(ht_type * table, DATA_TYPE * insert_buffer, uint64_t n_keys, uint64_t * misses){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys) return;

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }


   uint64_t my_key = insert_buffer[tid];

   if (!table->remove(my_tile, my_key)){

      // table->upsert_replace(my_tile, my_key, my_key);

      // table->remove(my_tile, my_key);


      #if MEASURE_FAILS
      if (my_tile.thread_rank() == 0){

         atomicAdd((unsigned long long int *)&misses[2], 1ULL);
         //printf("Init upsert failed for %lu\n", my_key);
      }
      #endif
      
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

      //table->upsert_replace(my_tile, my_key, my_key);

      //table->find_with_reference(my_tile, my_key, my_val);


      #if MEASURE_FAILS
      if (my_tile.thread_rank() == 0){

         atomicAdd((unsigned long long int *)&misses[1], 1ULL);
         //printf("Init upsert failed for %lu\n", my_key);
      }
      #endif
      
   } else {

      if (my_val != my_key){


        //table->find_with_reference(my_tile, my_key, my_val);


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

   #if LOAD_CHEAP
      std::string filename = "results/lf_probe_bght/";
   #else
      std::string filename = "results/lf_probe/";
   #endif

   filename = filename + ht_type::get_name() + ".txt";


   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query,remove\n";


   #else

   #if LOAD_CHEAP
      std::string filename = "results/lf_bght/";
   #else
      std::string filename = "results/lf/";
   #endif

   filename = filename + ht_type::get_name() + ".txt";


   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query,remove\n";

   #endif

  


   double avg_insert_throughput = 0;
   double avg_query_throughput = 0;
   double avg_delete_throughput = 0;


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

      cudaDeviceSynchronize();



      gallatin::utils::timer remove_timer;
      
      remove_kernel<ht_type, tile_size><<<(items_to_insert*tile_size-1)/256+1,256>>>(table, device_data, items_to_insert, misses);

      remove_timer.sync_end();

      uint64_t remove_probes = helpers::get_num_probes();

      cudaDeviceSynchronize();



      //free tables and generate results
      cudaFree(device_data);

      //table->print_chain_stats();

      ht_type::free_on_device(table);

      // insert_timer.print_throughput("Inserted", items_to_insert);
      // query_timer.print_throughput("Queried", items_to_insert);
      // remove_timer.print_throughput("Removed", items_to_insert);

      #if COUNT_PROBES

      //printf("Probes %lu %lu %lu\n", insert_probes, query_probes, remove_probes);
    
      myfile << lf << "," << std::setprecision(12) << 1.0*insert_probes/items_to_insert << "," << 1.0*query_probes/items_to_insert << "," << 1.0*remove_probes/items_to_insert << "\n";

      #else

      myfile << lf << "," << std::setprecision(12) << 1.0*items_to_insert/(insert_timer.elapsed()*1000000) << "," << 1.0*items_to_insert/(query_timer.elapsed()*1000000) << "," << 1.0*items_to_insert/(remove_timer.elapsed()*1000000) << "\n";

      #endif

      //printf("Misses: %lu %lu %lu\n", misses[0], misses[1], misses[2]);

      // misses[0] = 0;
      // misses[1] = 0;
      // misses[2] = 0;
      cudaDeviceSynchronize();

      //cuckoo is not leaking memory oon device.
      //gallatin::allocators::print_global_stats();

      avg_insert_throughput = 1.0*items_to_insert/(insert_timer.elapsed()*1000000);
      avg_query_throughput = 1.0*items_to_insert/(query_timer.elapsed()*1000000);
      avg_delete_throughput = 1.0*items_to_insert/(remove_timer.elapsed()*1000000);



   }

   double avg_throughput = (avg_insert_throughput+avg_query_throughput+avg_delete_throughput)/3;

   //printf("%u-%u Avg operations throughput %f\n", bucket_size, tile_size, avg_throughput);

   //printf("Misses: %lu %lu %lu\n", misses[0], misses[1], misses[2]);

   myfile.close();
 
  
   cudaFree(misses);
   cudaDeviceSynchronize();

}

template <template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ void lf_test_combo_cuckoo(uint64_t n_indices, DATA_TYPE * access_pattern){



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

   #if LOAD_CHEAP
      std::string filename = "results/lf_probe_bght/";
   #else
      std::string filename = "results/lf_probe/";
   #endif

   filename = filename + ht_type::get_name() + ".txt";


   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query,remove\n";


   #else

   #if LOAD_CHEAP
      std::string filename = "results/lf_bght/";
   #else
      std::string filename = "results/lf/";
   #endif

   filename = filename + ht_type::get_name() + ".txt";


   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query,remove\n";

   #endif

  


   double avg_insert_throughput = 0;
   double avg_query_throughput = 0;
   double avg_delete_throughput = 0;


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

      insert_kernel<ht_type, tile_size><<<(items_to_insert*tile_size-1)/64+1,64>>>(table, device_data, items_to_insert, misses);

      insert_timer.sync_end();

      uint64_t insert_probes = helpers::get_num_probes();

      cudaDeviceSynchronize();



      gallatin::utils::timer query_timer;

      query_kernel<ht_type, tile_size><<<(items_to_insert*tile_size-1)/64+1,64>>>(table, device_data, items_to_insert, misses);

      query_timer.sync_end();

      uint64_t query_probes = helpers::get_num_probes();

      cudaDeviceSynchronize();



      gallatin::utils::timer remove_timer;
      
      remove_kernel<ht_type, tile_size><<<(items_to_insert*tile_size-1)/64+1,64>>>(table, device_data, items_to_insert, misses);

      remove_timer.sync_end();

      uint64_t remove_probes = helpers::get_num_probes();

      cudaDeviceSynchronize();



      //free tables and generate results
      cudaFree(device_data);

      //table->print_chain_stats();

      ht_type::free_on_device(table);

      // insert_timer.print_throughput("Inserted", items_to_insert);
      // query_timer.print_throughput("Queried", items_to_insert);
      // remove_timer.print_throughput("Removed", items_to_insert);

      #if COUNT_PROBES

      //printf("Probes %lu %lu %lu\n", insert_probes, query_probes, remove_probes);
    
      myfile << lf << "," << std::setprecision(12) << 1.0*insert_probes/items_to_insert << "," << 1.0*query_probes/items_to_insert << "," << 1.0*remove_probes/items_to_insert << "\n";

      #else

      myfile << lf << "," << std::setprecision(12) << 1.0*items_to_insert/(insert_timer.elapsed()*1000000) << "," << 1.0*items_to_insert/(query_timer.elapsed()*1000000) << "," << 1.0*items_to_insert/(remove_timer.elapsed()*1000000) << "\n";

      #endif

      //printf("Misses: %lu %lu %lu\n", misses[0], misses[1], misses[2]);

      // misses[0] = 0;
      // misses[1] = 0;
      // misses[2] = 0;
      cudaDeviceSynchronize();

      //cuckoo is not leaking memory oon device.
      //gallatin::allocators::print_global_stats();

      avg_insert_throughput = 1.0*items_to_insert/(insert_timer.elapsed()*1000000);
      avg_query_throughput = 1.0*items_to_insert/(query_timer.elapsed()*1000000);
      avg_delete_throughput = 1.0*items_to_insert/(remove_timer.elapsed()*1000000);



   }

   double avg_throughput = (avg_insert_throughput+avg_query_throughput+avg_delete_throughput)/3;

   // printf("%u-%u Avg operations throughput %f\n", bucket_size, tile_size, avg_throughput);
   // printf("Misses: %lu %lu %lu\n", misses[0], misses[1], misses[2]);


   myfile.close();
 
  
   cudaFree(misses);
   cudaDeviceSynchronize();

}


template<template<typename, typename, uint, uint> typename hash_table_type>
__host__ void test_all_combinations(uint64_t n_indices, DATA_TYPE * access_pattern){

   //fill out arguments with junk to get static type
   std::cout << "Table: " << hash_table_type<uint64_t, uint64_t, 1,1>::get_name() << std::endl;
   lf_test<hash_table_type, 1,4>(n_indices, access_pattern);
   lf_test<hash_table_type, 2,4>(n_indices, access_pattern);
   lf_test<hash_table_type, 4,4>(n_indices, access_pattern);
   lf_test<hash_table_type, 1,8>(n_indices, access_pattern);
   lf_test<hash_table_type, 2,8>(n_indices, access_pattern);
   lf_test<hash_table_type, 4,8>(n_indices, access_pattern);
   lf_test<hash_table_type, 8,8>(n_indices, access_pattern);
   lf_test<hash_table_type, 1,16>(n_indices, access_pattern);
   lf_test<hash_table_type, 2,16>(n_indices, access_pattern);
   lf_test<hash_table_type, 4,16>(n_indices, access_pattern);
   lf_test<hash_table_type, 8,16>(n_indices, access_pattern);
   lf_test<hash_table_type, 16,16>(n_indices, access_pattern);
   lf_test<hash_table_type, 1,32>(n_indices, access_pattern);
   lf_test<hash_table_type, 2,32>(n_indices, access_pattern);
   lf_test<hash_table_type, 4,32>(n_indices, access_pattern);
   lf_test<hash_table_type, 8,32>(n_indices, access_pattern);
   lf_test<hash_table_type, 16,32>(n_indices, access_pattern);
   lf_test<hash_table_type, 32,32>(n_indices, access_pattern);

}

template<template<typename, typename, uint, uint> typename hash_table_type>
__host__ void test_all_combinations_cuckoo(uint64_t n_indices, DATA_TYPE * access_pattern){

   //fill out arguments with junk to get static type
   std::cout << "Table: " << hash_table_type<uint64_t, uint64_t, 1,1>::get_name() << std::endl;
   lf_test_combo_cuckoo<hash_table_type, 1,4>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 2,4>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 4,4>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 1,8>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 2,8>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 4,8>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 8,8>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 1,16>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 2,16>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 4,16>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 8,16>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 16,16>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 1,32>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 2,32>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 4,32>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 8,32>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 16,32>(n_indices, access_pattern);
   lf_test_combo_cuckoo<hash_table_type, 32,32>(n_indices, access_pattern);

}

template<template<typename, typename, uint, uint> typename hash_table_type>
__host__ void test_all_combinations_md(uint64_t n_indices, DATA_TYPE * access_pattern){

   //fill out arguments with junk to get static type
   std::cout << "Table: " << hash_table_type<uint64_t, uint64_t, 1,1>::get_name() << std::endl;
   // lf_test<hash_table_type, 1,4>(n_indices, access_pattern);
   // lf_test<hash_table_type, 2,4>(n_indices, access_pattern);
   // lf_test<hash_table_type, 4,4>(n_indices, access_pattern);
   lf_test<hash_table_type, 1,8>(n_indices, access_pattern);
   //lf_test<hash_table_type, 2,8>(n_indices, access_pattern);
   // lf_test<hash_table_type, 4,8>(n_indices, access_pattern);
   // lf_test<hash_table_type, 8,8>(n_indices, access_pattern);
   lf_test<hash_table_type, 1,16>(n_indices, access_pattern);
   lf_test<hash_table_type, 2,16>(n_indices, access_pattern);
   // lf_test<hash_table_type, 4,16>(n_indices, access_pattern);
   // lf_test<hash_table_type, 8,16>(n_indices, access_pattern);
   // lf_test<hash_table_type, 16,16>(n_indices, access_pattern);
   lf_test<hash_table_type, 1,32>(n_indices, access_pattern);
   lf_test<hash_table_type, 2,32>(n_indices, access_pattern);
   lf_test<hash_table_type, 4,32>(n_indices, access_pattern);
   // lf_test<hash_table_type, 8,32>(n_indices, access_pattern);
   // lf_test<hash_table_type, 16,32>(n_indices, access_pattern);
   // lf_test<hash_table_type, 32,32>(n_indices, access_pattern);

}


template <typename data>
__global__ void count_duplicates(data * data_array, uint64_t n_pairs, uint64_t * misses){

   uint64_t tid = gallatin::utils::get_tid();

   uint64_t first = tid/n_pairs;
   uint64_t second = tid % n_pairs;

   if (first >= n_pairs || second >= n_pairs) return;


   if (data_array[first] == data_array[second]){
      atomicAdd((unsigned long long int *)&misses, 1ULL);
   }

}

template <typename data>
__host__ void print_duplicates(data * data_array, uint64_t n_pairs){


   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   misses[0] = 0;


   data * device_data = gallatin::utils::get_device_version<data>(n_pairs);

   cudaMemcpy(device_data, data_array, sizeof(data)*n_pairs, cudaMemcpyHostToDevice);

   cudaDeviceSynchronize();

   count_duplicates<<<(n_pairs*n_pairs-1)/512+1,512>>>(device_data, n_pairs, misses);

   cudaDeviceSynchronize();

   //printf("System has %lu duplicates\n", misses[0]);

   cudaFree(misses);
   cudaFree(device_data);

}


template<typename data, typename pair_type>
__global__ void setup_bght_pair(data * data_array, pair_type * pair_array, uint64_t n_pairs){


   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= n_pairs) return;

   pair_array[tid].first = data_array[tid];
   pair_array[tid].second = data_array[tid];

} 


template <typename ht_type, typename data, uint tile_size>
__global__ void bght_query(ht_type table, data * query_data, uint64_t n_keys){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys) return;

   data key = query_data[tid];

   auto result = table.find(key, my_tile);

   //doesn't trigger?
   // if (result != key){

   //    //if (my_tile)
   //    printf("Missed\n");
   // }



}

template <template<typename, typename> typename hash_table_type, uint tile_size>
__host__ void lf_test_BGHT(uint64_t n_indices, DATA_TYPE * access_pattern, std::string ht_name){



   using ht_type = hash_table_type<DATA_TYPE, DATA_TYPE>;

   using pair_type = bght::padded_pair<DATA_TYPE, DATA_TYPE>;


   #if COUNT_PROBES

   std::string filename = "results/lf_probe_bght/";

   filename = filename + ht_name + ".txt";


   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query\n";


   #else

   std::string filename = "results/lf_bght/";

   filename = filename + ht_name + ".txt";


   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query\n";

   #endif


   for (int i = 1; i < 19; i++){

      

      double lf = .05*i;

      ht_type table(n_indices, 0U, ~0U);

      uint64_t items_to_insert = lf*n_indices;

      DATA_TYPE * device_data = gallatin::utils::get_device_version<DATA_TYPE>(items_to_insert);

      DATA_TYPE * query_data = gallatin::utils::get_device_version<DATA_TYPE>(items_to_insert);

      pair_type * device_pairs = gallatin::utils::get_device_version<pair_type>(items_to_insert);

      //set original buffer
      cudaMemcpy(device_data, access_pattern, sizeof(DATA_TYPE)*items_to_insert, cudaMemcpyHostToDevice);

      cudaDeviceSynchronize();

      setup_bght_pair<DATA_TYPE,pair_type><<<(items_to_insert-1)/256+1,256>>>(device_data, device_pairs, items_to_insert);


      #if COUNT_PROBES

      cudaDeviceSynchronize();

      bght::get_num_probes();

      #endif

      cudaDeviceSynchronize();




      gallatin::utils::timer insert_timer;

      bool inserted = table.insert(device_pairs, device_pairs+items_to_insert, 0);

      if (!inserted){
         printf("Failed to insert BGHT keys\n");
      }

      insert_timer.sync_end();

      #if COUNT_PROBES
      auto insert_probes = bght::get_num_probes();
      #endif


      //query
      gallatin::utils::timer query_timer;

      //bght_query<ht_type, DATA_TYPE, tile_size><<<(items_to_insert*tile_size-1)/256+1,256>>>(table, device_data, items_to_insert);

      table.find(device_data, device_data+items_to_insert, query_data, 0);


      query_timer.sync_end();


      #if COUNT_PROBES
      auto query_probes = bght::get_num_probes();
      #endif

      DATA_TYPE * output = gallatin::utils::get_device_version<DATA_TYPE>(items_to_insert);

      // gallatin::utils::timer self_query_timer;

      // table.find(device_data, device_data+items_to_insert, output);


      // self_query_timer.sync_end();

      #if COUNT_PROBES

      myfile << lf << "," << std::setprecision(12) << 1.0*insert_probes/items_to_insert << "," << 1.0*query_probes/items_to_insert  << "\n";

      #else

      myfile << lf << "," << std::setprecision(12) << 1.0*items_to_insert/(insert_timer.elapsed()*1000000) << "," << 1.0*items_to_insert/(query_timer.elapsed()*1000000) << "\n";

      #endif

      insert_timer.print_throughput("Inserted", items_to_insert);
      query_timer.print_throughput("Queried", items_to_insert);
      //self_query_timer.print_throughput("Self Queried", items_to_insert);

      cudaFree(device_pairs);
      cudaFree(device_data);
      cudaFree(query_data);
      cudaFree(output);




   }

   myfile.close();


}



__host__ void execute_test(std::string table, uint64_t table_capacity){


   auto access_pattern = generate_data<DATA_TYPE>(table_capacity);

   if (table == "p2"){

      lf_test<warpSpeed::tables::p2_ext_generic, 8, 32>(table_capacity, access_pattern);

      //p2 p2MD double doubleMD iceberg icebergMD cuckoo chaining bght_p2 bght_cuckoo");


   } else if (table == "p2MD"){

      lf_test<warpSpeed::tables::md_p2_generic, 4, 32>(table_capacity, access_pattern);

   } else if (table == "double"){
      lf_test<warpSpeed::tables::double_generic, 8, 8>(table_capacity, access_pattern);

   } else if (table == "doubleMD"){

      lf_test<warpSpeed::tables::md_double_generic, 4, 32>(table_capacity,access_pattern);


   } else if (table == "iceberg"){

      lf_test<warpSpeed::tables::iht_p2_generic, 8, 32>(table_capacity, access_pattern);
     
   } else if (table == "icebergMD"){

      lf_test<warpSpeed::tables::iht_p2_metadata_full_generic, 4, 32>(table_capacity, access_pattern);

   } else if (table == "cuckoo") {
       lf_test<warpSpeed::tables::cuckoo_generic, 4, 8>(table_capacity, access_pattern);
   
   } else if (table == "chaining"){

      init_global_allocator(30ULL*1024*1024*1024, 111);

      lf_test<warpSpeed::tables::chaining_generic, 4, 8>(table_capacity, access_pattern);

      free_global_allocator();
   } else if (table == "bght_p2"){

      lf_test_BGHT<bght::bcht8, 8>(table_capacity, access_pattern, "bcht_8");

   } else if (table == "bght_cuckoo"){

      lf_test_BGHT<bght::p2bht32, 32>(table_capacity, access_pattern, "p2bht_32");

   } else {
      throw std::runtime_error("Unknown table");
   }



   cudaFreeHost(access_pattern);
}



int main(int argc, char** argv) {


   argparse::ArgumentParser program("phased_probes");

   // program.add_argument("square")
   // .help("display the square of a given integer")
   // .scan<'i', int>();

   program.add_argument("--table", "-t")
   .required()
   .help("Specify table type. Options [p2 p2MD double doubleMD iceberg icebergMD cuckoo chaining bght_p2 bght_cuckoo");

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

   // uint64_t table_capacity;


   std::cout << "Running phased probe test with table " << table << " and " << table_capacity << " slots." << std::endl;

   // if (argc < 2){
   //    table_capacity = 100000000;
   // } else {
   //    table_capacity = std::stoull(argv[1]);
   // }

   // if (argc < 3){
   //    n_rounds = 1000;
   // } else {
   //    n_rounds = std::stoull(argv[2]);
   // }


   if(fs::create_directory("results")){
    //std::cout << "Created a directory\n";
   } else {
    //std::cerr << "Failed to create a directory\n";
   }

   if(fs::create_directory("results/lf")){
    //std::cout << "Created a directory\n";
   } else {
    //std::cerr << "Failed to create a directory\n";
   }


   #if COUNT_PROBES

   if(fs::create_directory("results/lf_probe")){
    //std::cout << "Created a directory\n";
   } else {
    //std::cerr << "Failed to create a directory\n";
   }

   #endif

   if(fs::create_directory("results/lf_bght")){
    //std::cout << "Created a directory\n";
   } else {
    //std::cerr << "Failed to create a directory\n";
   }


   #if COUNT_PROBES

   if(fs::create_directory("results/lf_probe_bght")){
    //std::cout << "Created a directory\n";
   } else {
    //std::cerr << "Failed to create a directory\n";
   }

   #endif


   execute_test(table, table_capacity);



   //print_duplicates<DATA_TYPE>(access_pattern, table_capacity);
   



   

   // lf_test<warpSpeed::tables::p2_ext_generic, 4, 32>(table_capacity, access_pattern);
   // lf_test<warpSpeed::tables::p2_ext_generic, 8, 32>(table_capacity, access_pattern);



   // lf_test<warpSpeed::tables::md_p2_generic, 4, 32>(table_capacity, access_pattern);
   
   // lf_test<warpSpeed::tables::md_double_generic, 4, 32>(table_capacity, access_pattern);



   
   // init_global_allocator(15ULL*1024*1024*1024, 111);

   // lf_test<warpSpeed::tables::chaining_generic, 4, 8>(table_capacity, access_pattern);

   

   // free_global_allocator();

   // cudaDeviceSynchronize();

   // lf_test<warpSpeed::tables::cuckoo_generic, 4, 8>(table_capacity, access_pattern);
   
   // lf_test<warpSpeed::tables::cuckoo_generic, 8, 8>(table_capacity, access_pattern);
   

   // lf_test<warpSpeed::tables::iht_p2_generic, 8, 32>(table_capacity, access_pattern);
   

   // lf_test<warpSpeed::tables::iht_p2_metadata_full_generic, 4, 32>(table_capacity, access_pattern);

  


   // printf("Starting BGHT tests\n");

   // lf_test_BGHT<bght::bcht8, 8>(table_capacity, access_pattern, "bcht_8");

   // lf_test_BGHT<bght::bcht16, 16>(table_capacity, access_pattern, "bcht_16");
   
   // lf_test_BGHT<bght::bcht32, 32>(table_capacity, access_pattern, "bcht_32");


   // lf_test_BGHT<bght::p2bht8, 8>(table_capacity, access_pattern, "p2bht_8");
   
   // lf_test_BGHT<bght::p2bht16, 16>(table_capacity, access_pattern, "p2bht_16");
   
   // lf_test_BGHT<bght::p2bht32, 32>(table_capacity, access_pattern, "p2bht_32");



   //testing for all possible configurations.

   //test_all_combinations<warpSpeed::tables::p2_ext_generic>(table_capacity, access_pattern);
   // test_all_combinations_md<warpSpeed::tables::md_p2_generic>(table_capacity, access_pattern);

   //test_all_combinations<warpSpeed::tables::double_generic>(table_capacity, access_pattern);
  
   // test_all_combinations_md<warpSpeed::tables::md_double_generic>(table_capacity, access_pattern);

   // test_all_combinations_cuckoo<warpSpeed::tables::cuckoo_generic>(table_capacity, access_pattern);
   
   // test_all_combinations<warpSpeed::tables::iht_p2_generic>(table_capacity, access_pattern);
  
   // test_all_combinations_md<warpSpeed::tables::iht_p2_metadata_full_generic>(table_capacity, access_pattern);

   // init_global_allocator(15ULL*1024*1024*1024, 111);

   // test_all_combinations<warpSpeed::tables::chaining_generic>(table_capacity, access_pattern);
   

   // free_global_allocator();
   
  

   

   cudaDeviceReset();
   return 0;

}
