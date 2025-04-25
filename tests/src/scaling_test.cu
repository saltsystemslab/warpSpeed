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

#include <argparse/argparse.hpp>


#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <warpcore/single_value_hash_table.cuh>
#include <gallatin/allocators/timer.cuh>

#include <bght/p2bht.hpp>



#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <chrono>
#include <openssl/rand.h>

#include <filesystem>

namespace fs = std::filesystem;



// 
#include <warpSpeed/tables/p2_hashing.cuh>

#include <warpSpeed/tables/double_hashing.cuh>
#include <warpSpeed/tables/iht_p2.cuh>
#include <warpSpeed/tables/iht_metadata.cuh>
#include <warpSpeed/tables/chaining.cuh>
#include <warpSpeed/tables/p2_hashing_metadata.cuh>
#include <warpSpeed/tables/cuckoo.cuh>
#include <warpSpeed/tables/double_hashing_metadata.cuh>

#include <iostream>
#include <locale>


//thrust stuff.
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <math.h>

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

      //table->upsert_replace(my_tile, my_key, my_key);


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
__host__ void lf_test(uint64_t n_indices, DATA_TYPE * access_pattern, std::ofstream & myfile){



   using ht_type = hash_table_type<DATA_TYPE, DATA_TYPE, tile_size, bucket_size>;


   //generate table and buffers
   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*4);

   cudaDeviceSynchronize();

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;




   for (int i = 18; i < 19; i++){

      

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

      ht_type::free_on_device(table);

      // insert_timer.print_throughput("Inserted", items_to_insert);
      // query_timer.print_throughput("Queried", items_to_insert);
      // remove_timer.print_throughput("Removed", items_to_insert);

      #if COUNT_PROBES

      printf("Probes %lu %lu %lu\n", insert_probes, query_probes, remove_probes);
    
      myfile << n_indices << "," << std::setprecision(12) << 1.0*insert_probes/items_to_insert << "," << 1.0*query_probes/items_to_insert << "," << 1.0*remove_probes/items_to_insert << "\n";

      #else

      myfile << n_indices << "," << std::setprecision(12) << 1.0*items_to_insert/(insert_timer.elapsed()*1000000) << "," << 1.0*items_to_insert/(query_timer.elapsed()*1000000) << "," << 1.0*items_to_insert/(remove_timer.elapsed()*1000000) << "\n";

      #endif

      //printf("Misses: %lu %lu %lu\n", misses[0], misses[1], misses[2]);

      misses[0] = 0;
      misses[1] = 0;
      misses[2] = 0;
      cudaDeviceSynchronize();

      //cuckoo is not leaking memory on device.
      //gallatin::allocators::print_global_stats();


   }


   //myfile.close();
 
  
   cudaFree(misses);
   cudaDeviceSynchronize();

}


template <template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ void run_scaling(uint64_t n_items_start, uint32_t n_rounds, uint32_t scaling_factor, DATA_TYPE * access_pattern){

   using ht_type = hash_table_type<DATA_TYPE, DATA_TYPE, tile_size, bucket_size>;

   if (n_rounds == 0) return;

   uint64_t max_items = n_items_start * std::pow(scaling_factor, n_rounds-1);

   //auto access_pattern = generate_data<DATA_TYPE>(max_items);



   #if COUNT_PROBES

   std::string filename = "results/scaling_probe/";

   filename = filename + ht_type::get_name() + ".txt";


   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query\n";


   #else

   std::string filename = "results/scaling/";

   filename = filename + ht_type::get_name() + ".txt";


   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "size,insert,query,remove\n";

   #endif

  


   for (uint64_t i = 0; i < n_rounds; i++){

      uint64_t items_in_round = n_items_start * std::pow(scaling_factor, i);

      lf_test<hash_table_type, tile_size, bucket_size>(items_in_round, access_pattern, myfile);

   }

   myfile.close();


}


__host__ void execute_test(std::string table, uint64_t table_capacity, uint32_t n_rounds, uint32_t scaling_factor){

   uint64_t max_items = table_capacity * std::pow(scaling_factor, n_rounds-1);

   auto access_pattern = generate_data<DATA_TYPE>(max_items);

   if (table == "p2"){

      run_scaling<warpSpeed::tables::p2_ext_generic, 8, 32>(table_capacity, n_rounds, scaling_factor, access_pattern);

      //p2 p2MD double doubleMD iceberg icebergMD cuckoo chaining bght_p2 bght_cuckoo");


   } else if (table == "p2MD"){

      run_scaling<warpSpeed::tables::md_p2_generic, 4, 32>(table_capacity, n_rounds, scaling_factor, access_pattern);

   } else if (table == "double"){
      run_scaling<warpSpeed::tables::double_generic, 8, 8>(table_capacity, n_rounds, scaling_factor, access_pattern);

   } else if (table == "doubleMD"){

      run_scaling<warpSpeed::tables::md_double_generic, 4, 32>(table_capacity, n_rounds, scaling_factor, access_pattern);


   } else if (table == "iceberg"){

      run_scaling<warpSpeed::tables::iht_p2_generic, 8, 32>(table_capacity, n_rounds, scaling_factor, access_pattern);
     
   } else if (table == "icebergMD"){

      run_scaling<warpSpeed::tables::iht_metadata_generic, 4, 32>(table_capacity, n_rounds, scaling_factor, access_pattern);

   } else if (table == "cuckoo") {
       run_scaling<warpSpeed::tables::cuckoo_generic, 4, 8>(table_capacity, n_rounds, scaling_factor, access_pattern);
   
   } else if (table == "chaining"){

      init_global_allocator(30ULL*1024*1024*1024, 111);

      run_scaling<warpSpeed::tables::chaining_generic, 4, 8>(table_capacity, n_rounds, scaling_factor, access_pattern);

      free_global_allocator();
   } else {
      throw std::runtime_error("Unknown table");
   }



   cudaFreeHost(access_pattern);
}

int main(int argc, char** argv) {


   argparse::ArgumentParser program("scaling_test");

   // program.add_argument("square")
   // .help("display the square of a given integer")
   // .scan<'i', int>();

   program.add_argument("--table", "-t")
   .required()
   .help("Specify table type. Options [p2 p2MD double doubleMD iceberg icebergMD cuckoo chaining");

   program.add_argument("--capacity", "-c").required().scan<'u', uint64_t>().help("Number of slots in the table.");

   program.add_argument("--rounds", "-r").required().scan<'u', uint32_t>().help("Number of rounds to execute. Each round increases number of items by scaling factor");

   program.add_argument("--scaling_factor", "-s").required().scan<'u', uint32_t>().help("Multiplicative increase in size of table per round.");


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

   uint32_t n_rounds = program.get<uint32_t>("--rounds");

   uint32_t scaling_factor = program.get<uint32_t>("--scaling_factor");

   std::cout << "Running scaling test with table " << table << " and " << table_capacity << " slots and " << n_rounds << " rounds with scaling factor " << scaling_factor << std::endl;

   // int n_rounds;

   // int scaling_factor;

   // if (argc < 2){
   //    table_capacity = 1000000;
   // } else {
   //    table_capacity = std::stoull(argv[1]);
   // }

   // if (argc < 3){
   //    n_rounds = 3;
   // } else {
   //    n_rounds = std::stoull(argv[2]);
   // }


   // if (argc < 4){
   //    scaling_factor = 10;
   // } else {
   //    scaling_factor = std::stoull(argv[3]);
   // }


   if(fs::create_directory("results")){
    //std::cout << "Created a directory\n";
   } else {
    //std::cerr << "Failed to create a directory\n";
   }

   if(fs::create_directory("results/scaling")){
    //std::cout << "Created a directory\n";
   } else {
    //std::cerr << "Failed to create a directory\n";
   }


   #if COUNT_PROBES

   if(fs::create_directory("results/scaling_probe")){
    //std::cout << "Created a directory\n";
   } else {
    //std::cerr << "Failed to create a directory\n";
   }

   #endif

   execute_test(table, table_capacity, n_rounds, scaling_factor);

  //  uint64_t max_items = table_capacity * std::pow(scaling_factor, n_rounds-1);

  //  auto access_pattern = generate_data<DATA_TYPE>(max_items);
   
  //  run_scaling<warpSpeed::tables::md_p2_generic, 4, 32>(table_capacity, n_rounds, scaling_factor, access_pattern);

  //  run_scaling<warpSpeed::tables::p2_ext_generic, 8, 32>(table_capacity, n_rounds, scaling_factor, access_pattern);
  //  run_scaling<warpSpeed::tables::double_generic, 4, 8>(table_capacity, n_rounds, scaling_factor, access_pattern);

  //  run_scaling<warpSpeed::tables::iht_p2_generic, 8, 32>(table_capacity, n_rounds, scaling_factor, access_pattern);

  //  run_scaling<warpSpeed::tables::iht_metadata_generic, 4, 32>(table_capacity, n_rounds, scaling_factor, access_pattern);


  //  run_scaling<warpSpeed::tables::cuckoo_generic, 4, 8>(table_capacity, n_rounds, scaling_factor, access_pattern);


  //  run_scaling<warpSpeed::tables::md_double_generic, 4, 32>(table_capacity, n_rounds, scaling_factor, access_pattern);

   
  //  init_global_allocator(30ULL*1024*1024*1024, 111);

  //  run_scaling<warpSpeed::tables::chaining_generic, 4, 8>(table_capacity, n_rounds, scaling_factor, access_pattern);

  //  free_global_allocator();

  //  cudaDeviceSynchronize();
   

  // // lf_test<warpSpeed::tables::iht_p2_generic, 8, 32>(table_capacity, access_pattern);
   
  

  //  //lf_test<warpSpeed::wrappers::warpcore_wrapper, 8, 8>(table_capacity, access_pattern);


  //  cudaFreeHost(access_pattern);



   cudaDeviceReset();
   return 0;

}
