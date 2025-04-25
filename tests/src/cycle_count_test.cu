/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




//macro to turn on cache benchmarking
#define COUNT_CACHE_CYCLES 1


#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <warpcore/single_value_hash_table.cuh>
#include <gallatin/allocators/timer.cuh>

#include <bght/p2bht.hpp>

#include <warpSpeed/helpers/cache.cuh>
#include <warpSpeed/helpers/host_cache.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <openssl/rand.h>

#include <fstream>
#include <locale>
#include <filesystem>

namespace fs = std::filesystem;



#include <warpSpeed/tables/p2_hashing_metadata.cuh>

#include <warpSpeed/tables/chaining.cuh>
#include <warpSpeed/tables/double_hashing.cuh>
#include <warpSpeed/tables/iht_p2.cuh>
#include <warpSpeed/tables/p2_hashing.cuh>

#include <warpSpeed/tables/iht_p2_metadata_full.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

using namespace gallatin::allocators;


#if GALLATIN_DEBUG_PRINTS
   #define TEST_BLOCK_SIZE 256
#else
   #define TEST_BLOCK_SIZE 256
#endif


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

   printf("Generation done\n");
   return vals;
}


template <typename cache_type, uint tile_size>
__global__ void cache_read_kernel(cache_type * cache, uint64_t n_indices, uint64_t * access_pattern, uint64_t n_ops){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_ops) return;

   uint64_t my_access = access_pattern[tid] % n_indices;

   //if (my_access == 0) my_access = 1;

   cache->read_item(my_tile, my_access);

   //if (my_tile.thread_rank() == 0 && tid % 100000 == 0) printf("Done with %lu\n", tid);

   //if (my_tile.thread_rank() == 0) printf("Done with %lu\n", tid);

   //cache->write_back_host<tile_size>(my_tile, my_access, 1);


   //if (my_tile.thread_rank() == 0 && tid % 1000000 == 0) printf("Done with %lu\n", tid);




}

//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
//works on actual pointers instead of uint64_t
//The correctness check is done by treating each allocation as a uint64_t and writing the tid
// if TID is not what is expected, we know that a double malloc has occurred.
template <template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ void cache_test(uint64_t host_items, uint64_t n_ops, uint64_t * data_pattern){


   using cache_type = warpSpeed::ht_fifo_cache<hash_table_type, tile_size, bucket_size>;

   uint64_t * dev_data = gallatin::utils::get_device_version<uint64_t>(n_ops);

   cudaMemcpy(dev_data, data_pattern, sizeof(uint64_t)*n_ops, cudaMemcpyHostToDevice);

   //uint64_t * dev_data = gallatin::utils::copy_to_device<uint64_t>(access_data, n_ops);



   std::string filename = "results/cycle_counts/";

   filename += cache_type::get_name();

   filename += ".txt";

   //std::string filename = "results/cache/" + "test" + ".txt";

   //std::string filename = "results/cache/test.txt" + "booga";


   printf("Writing to %s\n", filename.c_str());

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "fill total fast host queue\n";

   cudaDeviceSynchronize();



   //for (int i = 10; i <= 20; i++){
   for (int i = 20; i <= 90; i+=20){

      uint64_t capacity = host_items*(.01*i);

      cache_type * cache = cache_type::generate_on_device(host_items, capacity, .85);

      cudaDeviceSynchronize();

      gallatin::utils::timer cache_timing;

      cache_read_kernel<cache_type, tile_size><<<(n_ops*tile_size -1)/256+1,256>>>(cache, host_items, dev_data, n_ops);

      cache_timing.sync_end();

      double duration = cache_timing.elapsed();

      myfile << .01*i << " " << helpers::print_total_cycle_data() << " " << helpers::print_fast_cycle_data() << " " << helpers::print_host_cycle_data() << " " << helpers::print_queue_cycle_data() << "\n";

      //cache->print_space_usage();
   
      cache_type::free_on_device(cache);



   }

   myfile.close();

}


//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
//works on actual pointers instead of uint64_t
//The correctness check is done by treating each allocation as a uint64_t and writing the tid
// if TID is not what is expected, we know that a double malloc has occurred.
template<uint tile_size>
__host__ void host_cache_test(uint64_t host_items, uint64_t n_ops, uint64_t * data_pattern){


   using cache_type = warpSpeed::host_cache<tile_size>;

   uint64_t * dev_data = gallatin::utils::get_device_version<uint64_t>(n_ops);

   cudaMemcpy(dev_data, data_pattern, sizeof(uint64_t)*n_ops, cudaMemcpyHostToDevice);

   //uint64_t * dev_data = gallatin::utils::copy_to_device<uint64_t>(access_data, n_ops);



   std::string filename = "results/cycle_counts/";

   filename += cache_type::get_name();

   filename += ".txt";

   //std::string filename = "results/cache/" + "test" + ".txt";

   //std::string filename = "results/cache/test.txt" + "booga";


   printf("Writing to %s\n", filename.c_str());

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "fill total fast host\n";

   cudaDeviceSynchronize();



   //for (int i = 10; i <= 20; i++){
   for (int i = 20; i <= 90; i+=20){

      uint64_t capacity = host_items*(.01*i);

      cache_type * cache = cache_type::generate_on_device(host_items, capacity, .85);

      cudaDeviceSynchronize();

      gallatin::utils::timer cache_timing;

      cache_read_kernel<cache_type, tile_size><<<(n_ops*tile_size -1)/256+1,256>>>(cache, host_items, dev_data, n_ops);

      cache_timing.sync_end();

      double duration = cache_timing.elapsed();

      myfile << .01*i << " " << helpers::print_total_cycle_data() << "\n";

      //cache->print_space_usage();
   
      cache_type::free_on_device(cache);



   }

   myfile.close();

}

int main(int argc, char** argv) {

   uint64_t host_items;

   uint64_t n_ops;


   if (argc < 2){
      host_items = 1000000;
   } else {
      host_items = std::stoull(argv[1]);
   }


   if (argc < 3){
      n_ops = 1000;
   } else {
      n_ops = std::stoull(argv[2]);
   }

   if(fs::create_directory("results")){
    std::cout << "Created a directory\n";
   } else {
    std::cerr << "Failed to create a directory\n";
   }

   if(fs::create_directory("results/cycle_counts")){
    std::cout << "Created a directory\n";
   } else {
    std::cerr << "Failed to create a directory\n";
   }


   uint64_t * access_data = generate_data<uint64_t>(n_ops);


   //can't give up this space.
   // init_global_allocator(16ULL*1024*1024*1024, 111);

   
   
   // cache_test<warpSpeed::tables::chaining_generic, 4, 8>(host_items, n_ops, access_data);

   // free_global_allocator();

   host_cache_test<4>(host_items, n_ops, access_data);


   cache_test<warpSpeed::tables::md_p2_generic, 4, 32>(host_items, n_ops, access_data);

   cache_test<warpSpeed::tables::iht_p2_metadata_full_generic, 4, 32>(host_items, n_ops, access_data);


   //cache_test<warpSpeed::tables::p2_int_generic, 8, 32>(host_items, n_ops, access_data);
   
   cache_test<warpSpeed::tables::double_generic, 4, 8>(host_items, n_ops, access_data);
   cache_test<warpSpeed::tables::iht_p2_generic, 8, 32>(host_items, n_ops, access_data);
   //cache_test<warpSpeed::tables::p2_ext_generic, 8, 32>(host_items, n_ops, access_data);

   



   //free_global_allocator();


   cudaDeviceReset();
   return 0;

}
